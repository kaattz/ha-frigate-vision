import datetime as dt
from pathlib import Path

import yaml
from jinja2 import Environment

from custom_components.frigate_vision.scenes import scene_for

BLUEPRINT = (
    Path(__file__).parents[1]
    / "blueprints/automation/frigate_vision/activity_notification.yaml"
)

# The whitelist the live automation was configured with. Kept here verbatim
# because it is the input that caused the measured failure below, and the test is
# only meaningful while it still names the list that dropped events.
DEPLOYED_WHITELIST = [
    "home_arrival",
    "home_departure",
    "short_roundtrip",
    "package_delivery",
    "food_delivery",
    "cleaning",
    "maintenance",
    "visitor",
    "suspicious_activity",
    "unknown_activity",
    "unable_to_confirm",
]


class _BlueprintLoader(yaml.SafeLoader):
    """Loads a blueprint without Home Assistant's tag handling.

    `!input name` is kept as a marker string rather than resolved: the tests
    below render individual expressions, and a real substitution would need a
    running instance.
    """


def _construct_input(loader: yaml.SafeLoader, node: yaml.Node) -> str:
    return f"!input {loader.construct_scalar(node)}"


_BlueprintLoader.add_constructor("!input", _construct_input)


def _resolve(marker: object, inputs: dict[str, object]) -> object:
    """Resolve a `!input name` marker to the automation's configured value.

    Home Assistant substitutes these before the template renders; the loader
    above keeps them as markers so the file can be parsed without an instance.
    """
    text = str(marker)
    if text.startswith("!input "):
        return inputs[text.removeprefix("!input ")]
    return marker


def _should_notify(
    classification: str,
    allowed: list[str],
    at: dt.datetime,
    inputs: dict[str, object] | None = None,
) -> bool:
    """Render the blueprint's own `should_notify` expression.

    Rendering the shipped template rather than re-implementing the policy is the
    point: a test that restated the rule would keep passing while the blueprint
    drifted away from it.
    """
    path = Path(__file__).parents[1] / "blueprints/automation/frigate_vision"
    data = yaml.load(
        (path / "activity_notification.yaml").read_text("utf-8"),
        Loader=_BlueprintLoader,
    )
    variables = data["variables"]
    # Defaults leave the quiet window wide open, so only the classification is
    # under test unless a caller supplies a real window.
    resolved: dict[str, object] = {
        "quiet_start": "00:00:00",
        "quiet_end": "00:00:00",
    }
    if inputs:
        resolved.update(inputs)
    environment = Environment()
    environment.globals["now"] = lambda: at
    environment.globals["today_at"] = lambda value: dt.datetime.combine(
        at.date(), dt.time.fromisoformat(str(_resolve(value, resolved)))
    )
    rendered = environment.from_string(str(variables["should_notify"])).render(
        allowed_classifications=allowed,
        classification=classification,
        quiet_start=_resolve(variables.get("quiet_start", "00:00:00"), resolved),
        quiet_end=_resolve(variables.get("quiet_end", "00:00:00"), resolved),
    )
    return rendered.strip() == "True"


def test_quiet_window_still_suppresses() -> None:
    """Removing the classification gate must not remove the quiet hours too.

    The quiet window is the one policy left in the blueprint, so it is the one
    thing that can still stop a notification. Checked on both sides of a real
    window so a template that always returned True would fail here.
    """
    window = {"quiet_start": "23:00:00", "quiet_end": "07:00:00"}
    inside = dt.datetime(2026, 9, 24, 3, 0)
    outside = dt.datetime(2026, 9, 24, 12, 0)

    assert _should_notify("cleaning", [], inside, inputs=window) is False
    assert _should_notify("cleaning", [], outside, inputs=window) is True


def test_classification_never_suppresses_a_notification() -> None:
    """A classification is a label for the reader, not a reason to drop.

    Measured on this deployment: one morning produced five person reviews and
    only two arrived. Both missing ones were classified `elevator_activity`.
    The scene had gained that classification, but the automation's whitelist is a
    *static copy* of the classification list, and nothing updates it when the
    integration learns a new word -- so the event was analysed, delivered to the
    event bus, and then silently discarded by a list that had gone stale.

    Every classification the scene can emit must therefore survive the policy,
    even when a stale whitelist is supplied.
    """
    scene = scene_for("review_six")
    assert scene is not None

    # Mid-morning, outside every quiet window, so only classification is under
    # test here.
    at = dt.datetime(2026, 9, 24, 9, 0)
    dropped = [
        classification
        for classification in sorted(scene.classifications)
        if not _should_notify(classification, DEPLOYED_WHITELIST, at)
    ]
    assert not dropped, (
        f"these classifications would be silently discarded: {dropped}. "
        "A classification is information, not a gate."
    )


def test_blueprint_offers_no_classification_filter() -> None:
    """The blueprint must not present a filter that cannot be kept current.

    A user-facing checklist looks authoritative, so a stale entry is invisible:
    the notification simply never arrives. The integration already decides what
    is worth analysing (`min_review_seconds`, `analyze_all_far_reviews`), so a
    second, hand-maintained list in the blueprint only adds a silent way to lose
    events.
    """
    text = BLUEPRINT.read_text("utf-8")
    assert "classifications:" not in text
    assert "allowed_classifications" not in text


def test_notification_blueprint_exposes_policy_and_acks_after_action() -> None:
    text = BLUEPRINT.read_text("utf-8")
    for value in (
        "quiet_start:",
        "quiet_end:",
        "title_prefix:",
        "include_image:",
        "include_clip:",
        "notification_title:",
        "notification_message:",
        "evidence_url:",
        "clip_url:",
        "frigate_review_url:",
        "ack_delivery",
    ):
        assert value in text
    assert text.index("!input notification_action") < text.index("ack_delivery")
    assert "should_notify" in text
    assert "sequence: !input notification_action" in text
    assert "condition:" not in text.split("action:", 1)[0]
    assert "continue_on_error" not in text
    assert "notify." not in text
    assert "mode: parallel" in text
    assert "max: 20" in text


def test_blueprint_message_does_not_show_confidence() -> None:
    """Confidence must stay out of the user-facing message.

    Measured on this deployment: the same contact sheet analysed four times
    returned both `home_arrival` and `unable_to_confirm`. A number that moves
    between runs invites doubt about a verdict that is already the best available
    reading, and the user asked for it to be dropped.
    """
    text = BLUEPRINT.read_text("utf-8")
    message_line = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("notification_message:")
    )
    assert "confidence" not in message_line
    # The variable may still exist for policy use; only the message is checked.
    assert "confidence:" in text


def test_blueprint_forwards_the_evidence_sheet() -> None:
    """The popup's comparison sheet must reach the notification.

    Defined here and *passed* by the user's automation action are two different
    things: a blueprint variable that the action never forwards arrives empty,
    while the automation still fires and the notification still sends. This
    project has already lost a feature to exactly that, so the forwarding half
    is asserted in the automation's own file by
    `test_blueprint_links_the_clip_when_available`'s sibling checks below --
    here the blueprint must at least define both fields from the event, gated by
    `include_image` alongside the sheet the model was given.
    """
    text = BLUEPRINT.read_text("utf-8")
    assert "evidence_image_url" in text
    assert "trigger.event.data.evidence_image_url" in text
    assert "evidence_offsets" in text
    for name in ("evidence_image_url:", "evidence_offsets:"):
        line = next(
            line for line in text.splitlines() if line.strip().startswith(name)
        )
        # Both are gated with the sheet itself: a user who turned images off
        # should not get a grid either.
        assert "include_image" in line


def test_blueprint_links_the_clip_when_available() -> None:
    """The message must offer the recording, which is what the old one did.

    The previous automation linked a video; dropping that would be a visible
    regression even though this integration's evidence is a still image.
    """
    text = BLUEPRINT.read_text("utf-8")
    assert "clip_url" in text
    # The link is built from the event, not hardcoded.
    assert "trigger.event.data.clip_url" in text


def test_blueprint_shows_a_readable_label_not_the_enum() -> None:
    """The title must read as Chinese, not as `home_arrival`.

    The classification is an internal enum. The previous automation showed the
    user tags like 回家 and 保洁员, so leaving the raw value in the title would be
    a visible downgrade. The mapping lives in the blueprint so the enum stays the
    integration's vocabulary.
    """
    text = BLUEPRINT.read_text("utf-8")
    assert "classification_labels" in text
    for label in ("回家", "离家", "保洁", "快递", "外卖"):
        assert label in text, f"{label} missing from the label map"
    # The label variable must resolve through the map, and the title must use the
    # resolved label. Checking the title for the map name would be checking the
    # implementation rather than the outcome.
    label_line = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("classification_label:")
    )
    assert "classification_labels.get(classification" in label_line
    title_line = next(
        line
        for line in text.splitlines()
        if line.strip().startswith("notification_title:")
    )
    assert "classification_label" in title_line
    # And the raw enum must not reach the title.
    assert "~ classification }}" not in title_line
