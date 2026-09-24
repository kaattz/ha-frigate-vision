"""Tests for the scene interface.

Layer ② of this integration: the general vision core fetches frames and calls a
model, while a *scene* says what those frames mean. Keeping the two apart is what
lets another scenario (a driveway, a back garden) be added without touching the
core, and it is what stops a scene from quietly depending on another scene's
context.
"""

from __future__ import annotations

import pytest

from custom_components.frigate_vision.scenes import (
    SCENES,
    SceneRequest,
    scene_for,
    scene_prompt_version,
)

# Digest of the review_six prompt body as of PROMPT_VERSION = "prompt_4".
#
# Pinned so an edit to the wording cannot ride along on an unchanged version:
# `prompt_version` is half the analysis cache key, so leaving it alone after a
# text change makes the deployment keep serving the old answers. Bump the version
# and this digest in the same commit.
REVIEW_SIX_PROMPT_DIGEST = "c1ecfc13b5c9"


def test_every_scene_declares_what_it_can_answer() -> None:
    """The registry must be the single source of allowed answers.

    The model has no schema to enforce them -- the endpoint rejects
    `response_format: json_schema` -- so the set travels in the prompt and is
    checked again on the reply. Two tables would drift.
    """
    assert SCENES, "the registry must not be empty"
    for mode, scene in SCENES.items():
        assert scene.mode == mode
        assert scene.classifications, f"{mode} declares no classifications"
        assert scene.prompt_version, f"{mode} has no prompt version"


def test_a_scene_version_changes_only_its_own_cache_key() -> None:
    """Versions must be per scene, not global.

    The version is part of the analysis cache key. With one shared version, a
    wording change for the door invalidates every unrelated analysis -- and a
    scene's cached result would survive a change to its own prompt if some other
    scene's version happened to cover it.

    Distinct *current* values are not required; what matters is that each scene
    owns its version, so the two can diverge independently.
    """
    from custom_components.frigate_vision.vision import analysis_key

    base = scene_for("review_six")
    other = scene_for("door_single")
    assert base is not None and other is not None

    # Bumping a scene's own version must invalidate just that scene's analyses.
    bumped = analysis_key("activity", base.mode, "bumped_version")
    assert bumped != analysis_key("activity", base.mode, base.prompt_version)
    assert bumped.endswith("bumped_version")


def test_the_cache_key_separates_scenes() -> None:
    """One activity can be analysed by more than one scene.

    The key must identify the question as well as the record, or a result
    produced under one scene's rules would be reused for another's -- and two
    scenes that happen to share a prompt version would collide outright.
    """
    from custom_components.frigate_vision.vision import analysis_key

    review = analysis_key("activity", "review_six", "prompt_3")
    door = analysis_key("activity", "door_single", "prompt_3")
    assert review != door, "the same activity under two scenes must not collide"
    assert "review_six" in review
    assert "door_single" in door


def test_every_scene_owns_its_version_field() -> None:
    """Each scene carries its own version, so the two can move independently."""
    for mode, scene in SCENES.items():
        assert scene.prompt_version, f"{mode} has no prompt version"
    versions = {mode: scene.prompt_version for mode, scene in SCENES.items()}
    assert len(versions) == len(SCENES)


def test_unknown_mode_is_rejected_rather_than_guessed() -> None:
    """An unregistered mode must fail loudly.

    A silently-chosen default would analyse frames under the wrong rules, which
    is worse than refusing: the classification would look valid.
    """
    assert scene_for("no_such_scene") is None
    assert scene_for("review_six") is not None


def test_a_scene_declares_the_signals_it_consumes() -> None:
    """Auxiliary inputs are declared, so the seam is explicit.

    The door scenes read door-lock state; the general review scene does not.
    Declaring it makes the dependency visible and reviewable rather than buried
    in a prompt string.
    """
    review = scene_for("review_six")
    assert review is not None
    assert review.signals == frozenset(), "the general scene needs no extra input"

    roundtrip = scene_for("door_roundtrip")
    assert roundtrip is not None
    assert "door_remained_open" in roundtrip.signals
    assert "opening_side" in roundtrip.signals


def test_a_scene_receives_only_the_signals_it_declared() -> None:
    """Undeclared inputs must not reach the renderer.

    This is the isolation that makes a new scene safe to add: it cannot come to
    depend on another scene's context by accident, because that context is never
    handed to it.
    """
    general = scene_for("review_six")
    assert general is not None
    request = SceneRequest(
        language="中文",
        allowed=general.classifications,
        signals={"door_remained_open": True, "opening_side": "inside"},
    )
    prompt = general.render(request)
    # The door state is present in the request but the scene declared none, so
    # none of it may appear in the prompt.
    assert "持续开启" not in prompt
    assert "inside" not in prompt

    door = scene_for("door_roundtrip")
    assert door is not None
    door_prompt = door.render(
        SceneRequest(
            language="中文",
            allowed=door.classifications,
            signals={"door_remained_open": True, "opening_side": "inside"},
        )
    )
    assert "持续开启" in door_prompt


def test_rendering_lists_the_allowed_classifications() -> None:
    """The contract travels in the prompt, so every allowed value must appear."""
    scene = scene_for("review_six")
    assert scene is not None
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed=frozenset({"cleaning", "visitor"}),
            signals={},
        )
    )
    assert "cleaning" in prompt
    assert "visitor" in prompt
    assert "package_delivery" not in prompt


def test_the_glossary_never_teaches_a_disallowed_answer() -> None:
    """Definitions must follow the contract, not the scene's full vocabulary.

    The allowed set travels per call, so a caller may offer a subset. A glossary
    written from the scene's own list would then define answers the model is not
    permitted to give -- naming `package_delivery` in a call that forbids it
    invites exactly the out-of-contract answer the contract exists to prevent.

    Scoped to the glossary, not the whole prompt: the template always states the
    arrival/departure rules by name, and that predates this. What must follow
    `allowed` is the extra definitions.
    """
    scene = scene_for("review_six")
    assert scene is not None
    glossary = scene._glossary(frozenset({"cleaning", "visitor"}))
    for label in scene.classifications - {"cleaning", "visitor"}:
        assert label not in glossary, (
            f"{label} is defined although this call disallows it"
        )
    assert "visitor" in glossary


def test_the_glossary_is_empty_when_nothing_is_offered() -> None:
    """An empty contract must not carry definitions for absent answers."""
    scene = scene_for("review_six")
    assert scene is not None
    assert scene._glossary(frozenset()) == ""


def test_changing_the_prompt_body_changes_the_cache_key() -> None:
    """The version must move whenever the wording does.

    `prompt_version` is half of the analysis cache key, and the other half is the
    activity. So an edit to `template` that leaves the version alone makes every
    already-analysed activity keep its old answer -- the change ships, the cache
    hides it, and the deployment looks unchanged.

    This cannot detect the edit by itself: the version is a hand-written string
    and nothing ties it to the text. What it can do is fail loudly when the two
    are known to disagree, by pinning the digest the version was last bumped
    against. Update the digest *and* bump the version together, or this fails.
    """
    import hashlib

    scene = scene_for("review_six")
    assert scene is not None
    # Hash what actually reaches the model for one representative call, so the
    # digest covers the template, the glossary and the contract order alike.
    rendered = scene.render(
        SceneRequest(
            language="zh-CN",
            allowed=set(scene.classifications),
            signals={},
            scene_description="",
        )
    )
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    assert digest == REVIEW_SIX_PROMPT_DIGEST, (
        "the review_six prompt body changed:\n"
        f"  version  : {scene.prompt_version}\n"
        f"  digest   : {digest} (expected {REVIEW_SIX_PROMPT_DIGEST})\n"
        "Bump PROMPT_VERSION, then update REVIEW_SIX_PROMPT_DIGEST to match. "
        "Leaving the version alone would let the cache serve answers produced by "
        "the old wording."
    )

    """Every definition must name its label, or the model cannot bind it.

    Repeating a label the template already defines would also show two versions
    of one rule, so the two sets must not overlap.
    """
    from custom_components.frigate_vision.scenes import (
        CLASSIFICATION_GLOSSARY,
        SCENES,
    )

    # The map is shared across scenes, so a label is fair game if *any* scene
    # offers it -- `short_roundtrip` belongs to the door scene, not this one.
    offered = {label for scene in SCENES.values() for label in scene.classifications}
    for label, text in CLASSIFICATION_GLOSSARY.items():
        assert text.startswith("指"), f"{label}'s definition does not read as one"
        assert label in offered, f"{label} is defined but no scene offers it"
    # The template's own three rules must not be restated.
    for already_defined in ("cleaning", "home_arrival", "home_departure"):
        assert already_defined not in CLASSIFICATION_GLOSSARY, (
            f"{already_defined} is already defined by the template"
        )


def test_a_narrowed_call_still_explains_what_it_does_offer() -> None:
    """Filtering the glossary must not filter it away entirely."""
    scene = scene_for("review_six")
    assert scene is not None
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed=frozenset({"cleaning", "visitor"}),
            signals={},
        )
    )
    # `visitor` is one of the labels the template never used to define, so a
    # narrowed call must still carry its meaning.
    assert "访客" in prompt, "visitor was offered but not explained"


def test_rendering_without_a_classification_set_omits_the_contract() -> None:
    """An empty set means "no contract", not "an empty list of options"."""
    scene = scene_for("review_six")
    assert scene is not None
    prompt = scene.render(
        SceneRequest(language="中文", allowed=frozenset(), signals={})
    )
    assert "confidence" not in prompt


@pytest.mark.parametrize("mode", sorted(SCENES))
def test_every_scene_renders_a_non_empty_prompt(mode: str) -> None:
    """Each registered scene must be usable, not merely declared."""
    scene = SCENES[mode]
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed=scene.classifications,
            signals=dict.fromkeys(scene.signals, "unknown"),
        )
    )
    assert prompt.strip()
    assert "JSON" in prompt


def test_prompt_version_lookup_reports_unknown_modes() -> None:
    """Callers cache on this value, so an unknown mode must not get one."""
    assert scene_prompt_version("review_six")
    assert scene_prompt_version("no_such_scene") is None


def _render(mode: str, **signals: object) -> str:
    scene = SCENES[mode]
    return scene.render(
        SceneRequest(
            language="中文",
            allowed=set(scene.classifications),
            signals=signals,
        )
    )


def test_every_offered_label_is_explained_to_the_model() -> None:
    """A label the prompt never defines is a label the model must guess.

    Measured on this deployment: the review prompt offered twelve answers and
    explained five. `elevator_activity`, `package_delivery`, `food_delivery`,
    `visitor`, `maintenance`, `suspicious_activity` and `short_roundtrip`
    appeared only in the allowed-answers list, while `unknown_activity` and
    `unable_to_confirm` were each named three times. The model was told, in six
    separate prohibitions, what it must not say, and never what the remaining
    words meant.

    The cost was measurable and large: across 41 real activities the deployment
    abstained on 55% of them, and adding one sentence per label took that to 11%
    while *raising* accuracy on the owner's labelled sheets from 53% to 60%.

    A count of one means "only in the contract", so this fails for any label
    added later without a definition -- which is the failure it exists to catch.
    """
    prompt = _render("review_six")
    scene = SCENES["review_six"]
    unexplained = sorted(
        label for label in scene.classifications if prompt.count(label) <= 1
    )
    assert not unexplained, (
        f"these answers are offered but never explained: {unexplained}. "
        "A model that does not know what a label means cannot choose it on "
        "evidence, and will abstain instead."
    )


def test_the_prompt_offers_guidance_and_not_only_prohibitions() -> None:
    """Rules about what not to say cannot, alone, produce an answer.

    The measured prompt had six prohibitions ("不得"/"不要"/"不能") and no
    sentence of the form "判为/应判断" for the non-directional labels. Every
    classification that is not arrival or departure was therefore reachable
    only by elimination.
    """
    prompt = _render("review_six")
    prohibitions = sum(prompt.count(word) for word in ("不得", "不要", "不能"))
    assertions = prompt.count("指") + prompt.count("判为")
    assert assertions > 0, "the prompt gives no positive guidance at all"
    assert assertions >= prohibitions / 3, (
        f"the prompt is overwhelmingly prohibitive: {assertions} defining "
        f"statements against {prohibitions} prohibitions"
    )


def test_review_scene_states_the_arrival_rules() -> None:
    """The general scene must define direction by movement across the door.

    No zone telemetry reaches this path, so the only defensible evidence is
    visible movement. The prompt has to rule out the tempting shortcuts, or the
    model decides from dwell time or frame order and produces confident guesses.
    """
    prompt = _render("review_six")
    assert "home_arrival" in prompt
    assert "home_departure" in prompt
    assert "门内走出" in prompt
    assert "走近该门并进入门内" in prompt
    # The shortcuts must be named as insufficient.
    assert "停留时长" in prompt
    assert "出现顺序" in prompt
    assert "只是经过" in prompt


def test_review_scene_does_not_claim_door_lock_knowledge() -> None:
    """This path receives no lock input, so it must not imply otherwise.

    The door scenes may reason about the lock because they declare it as a
    signal; the general scene declares none.
    """
    scene = SCENES["review_six"]
    assert scene.signals == frozenset()
    prompt = _render("review_six")
    # The rules may name the door itself; what they must not do is claim to
    # know its lock state.
    assert "门磁" not in prompt
    assert "持续开启" not in prompt


def test_door_scene_treats_frames_as_candidates_not_conclusions() -> None:
    """The lock bounds the activity; it does not name the person's intent."""
    prompt = _render("door_single", opening_side="inside")
    assert "候选" in prompt
    assert "门锁" in prompt
    assert "不能" in prompt
    assert "离家候选" in prompt and "不是结论" in prompt


def test_roundtrip_scene_reports_the_lock_evidence_it_was_given() -> None:
    """Each lock state must produce a different, honest statement."""
    open_prompt = _render(
        "door_roundtrip", door_remained_open=True, opening_side="inside"
    )
    assert "上排" in open_prompt and "下排" in open_prompt
    assert "持续开启" in open_prompt

    conflict = _render(
        "door_roundtrip", door_remained_open=False, opening_side="inside"
    )
    assert "冲突" in conflict and "不能确认" in conflict

    unknown = _render("door_roundtrip", door_remained_open=None, opening_side="inside")
    assert "未知" in unknown


# --- deployment-supplied scene description (diagnosis defect C) -------------
# The general prompt's arrival/departure rules speak of "the front door", but
# nothing told the model which door in frame that is. Measured on this
# deployment: a person stepping out of a lift and walking away was classified
# `home_departure`, because the nearest door was read as the front door.


def test_an_empty_description_leaves_the_prompt_byte_for_byte_unchanged() -> None:
    """A deployment that configures nothing must see no change at all.

    This is the whole safety argument for shipping the option: the prompt is a
    cache key input and a behaviour input, so "empty means untouched" has to
    hold exactly, not approximately.
    """
    scene = SCENES["review_six"]
    without = scene.render(
        SceneRequest(language="中文", allowed=scene.classifications, signals={})
    )
    for empty in ("", "   ", "\n\t "):
        assert (
            scene.render(
                SceneRequest(
                    language="中文",
                    allowed=scene.classifications,
                    signals={},
                    scene_description=empty,
                )
            )
            == without
        ), "whitespace-only descriptions must count as unset"


def test_the_description_is_injected_before_the_rules() -> None:
    """Context first, rules second.

    The template's rules refer to "the front door" and to movement relative to
    it. If the description arrived after them, the model would read the rules
    without knowing which door they mean -- which is the defect itself.
    """
    scene = SCENES["review_six"]
    marker = "电梯1门通往大堂，入户门在画面左侧画外"
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed=scene.classifications,
            signals={},
            scene_description=marker,
        )
    )
    assert marker in prompt
    assert prompt.index(marker) < prompt.index("六格依次是")


def test_the_description_carries_no_authority_of_its_own() -> None:
    """Layout is context; it must not license a classification by itself.

    A description that merely said "the front door is off-camera" would still
    let a model conclude a departure from it. The injected text therefore says
    the description only identifies what is in frame.
    """
    scene = SCENES["review_six"]
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed=scene.classifications,
            signals={},
            scene_description="入户门在画外",
        )
    )
    assert "只是摄像头视野的说明" in prompt
    assert "不得据布局猜测" in prompt


def test_only_the_general_scene_accepts_a_description() -> None:
    """Declared, not assumed -- the same isolation `signals` uses.

    The door scenes carry their own preamble about the lift and the off-camera
    door; a description written for the general scene would duplicate or
    contradict it. A scene that did not ask for the text must never see it.
    """
    assert SCENES["review_six"].accepts_scene_description is True
    assert SCENES["door_single"].accepts_scene_description is False
    assert SCENES["door_roundtrip"].accepts_scene_description is False

    marker = "入户门在画面左侧画外"
    for mode in ("door_single", "door_roundtrip"):
        scene = SCENES[mode]
        prompt = scene.render(
            SceneRequest(
                language="中文",
                allowed=scene.classifications,
                signals=dict.fromkeys(scene.signals, "unknown"),
                scene_description=marker,
            )
        )
        assert marker not in prompt, (
            f"{mode} must ignore a description it did not declare"
        )


def test_editing_the_description_changes_the_cache_key() -> None:
    """Otherwise the edit looks like it did nothing.

    The version is a module constant, so it cannot see a deployment's option. A
    user who rewrote their description would be served the previous prompt's
    stored answer, and the change would appear to have had no effect -- the same
    shape of silent failure this project has already hit twice.
    """
    from custom_components.frigate_vision.scenes import effective_prompt_version

    first = effective_prompt_version("review_six", "入户门在画外左侧")
    second = effective_prompt_version("review_six", "入户门在画外右侧")
    assert first != second, "different descriptions must not share a cache key"
    assert first == effective_prompt_version("review_six", "入户门在画外左侧")


def test_an_unset_description_keeps_the_plain_version() -> None:
    """No description means no key change, so existing caches stay valid."""
    from custom_components.frigate_vision.scenes import effective_prompt_version

    base = SCENES["review_six"].prompt_version
    assert effective_prompt_version("review_six", "") == base
    assert effective_prompt_version("review_six", "   ") == base
    # A scene that never accepts a description is unaffected even if handed one.
    door = SCENES["door_single"].prompt_version
    assert effective_prompt_version("door_single", "任何描述") == door


def test_the_effective_version_is_a_valid_cache_key_component() -> None:
    """It reaches `analysis_key`, which validates against a strict alphabet.

    A digest that happened to contain a colon or a space would raise at analysis
    time -- after frames were fetched and possibly after the provider was
    charged.
    """
    from custom_components.frigate_vision.models import SAFE_ID
    from custom_components.frigate_vision.scenes import effective_prompt_version
    from custom_components.frigate_vision.vision import analysis_key

    for description in (
        "入户门在画外左侧",
        "a" * 500,
        "多行\n描述\t带特殊字符 :/?#[]@!$&'()*+,;=",
    ):
        version = effective_prompt_version("review_six", description)
        assert version is not None
        assert SAFE_ID.fullmatch(version), f"not a safe key component: {version!r}"
        # The real call site must accept it.
        assert analysis_key("activity_1", "review_six", version)

    assert effective_prompt_version("no_such_scene", "x") is None

