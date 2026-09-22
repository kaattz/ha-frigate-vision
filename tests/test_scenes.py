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
