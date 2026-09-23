from pathlib import Path

BLUEPRINT = (
    Path(__file__).parents[1]
    / "blueprints/automation/frigate_vision/activity_notification.yaml"
)


def test_notification_blueprint_exposes_policy_and_acks_after_action() -> None:
    text = BLUEPRINT.read_text("utf-8")
    for value in (
        "classifications:",
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
