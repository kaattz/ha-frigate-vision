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
REVIEW_SIX_PROMPT_DIGEST = "e6c70a758822"


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


def test_the_cell_description_does_not_hardcode_a_count() -> None:
    """The prompt must describe the layout, not assert a cell count.

    The sheet is 2x3 when no hole is wide enough to probe and 3x3 when one is, so
    a prompt saying "六格" is wrong half the time -- and wrong in the direction
    that matters, because it tells the model the last cell is the emptied scene
    when three probed frames have been inserted before it. A model counting six
    cells across nine would misread which frame is which.

    The structure is identical in both sizes: the first cell is the person's first
    frame, the last is the emptied scene, the one before it is the person's final
    frame, and everything between is candidates, ordered in time. Describing that
    keeps the prompt true at either size.
    """
    prompt = _render("review_six")
    for wrong in ("六格", "九格", "六个格", "9格"):
        assert wrong not in prompt, (
            f"the prompt asserts a fixed cell count ({wrong}) that only holds for "
            "one of the two sheet sizes"
        )
    # The structural anchors must still be stated, or the model loses the meaning
    # of the first and last cells.
    assert "首帧" in prompt and "末帧" in prompt and "后置" in prompt
    # And the order must be given, since the grid is read as a sequence.
    assert "顺序" in prompt or "依次" in prompt


def test_the_prompt_asks_for_a_recognisable_description_of_the_person() -> None:
    """The description is the notification body, so it must identify who it was.

    `description` is what the household actually reads: the blueprint sends it as
    the notification message. A description that says only "a person walked past"
    cannot be acted on -- the reader needs to know whether it was someone they
    know, and for a delivery whether to expect a parcel.

    The guidance is about *writing*, not about concluding, so it is checked for
    the three things a useful description needs and nothing more.
    """
    prompt = _render("review_six")
    assert "外貌" in prompt, "no guidance on describing the person's appearance"
    assert "性别" in prompt and "发型" in prompt, (
        "appearance guidance must name what to look at, or it is too vague to act on"
    )
    assert "来" in prompt and "去" in prompt, (
        "the description must state where the person came from and went to"
    )


def test_the_prompt_forbids_guessing_an_age_without_a_face() -> None:
    """A guessed age is a fabrication, and it reaches the reader as fact.

    Infrared frames and distant figures carry no facial detail, so an age is not
    observable from most of this deployment's evidence. An earlier prompt already
    forbade guessing a *profession* from a uniform for the same reason; age is the
    same failure with a different attribute.

    The neutral wording is asserted too, because "do not guess" without an
    alternative leaves the model with nothing to write.
    """
    prompt = _render("review_six")
    assert "年龄" in prompt, "no guidance about inferring age"
    assert "男子" in prompt or "人员" in prompt, (
        "the rule must offer neutral wording to use instead"
    )
    # And the prohibition must be tied to the absence of a face, not absolute:
    # with a clear face an age estimate is a reasonable observation.
    assert "面部" in prompt or "正面" in prompt, (
        "the rule must say when it applies, or it forbids an observable fact"
    )


def test_appearance_guidance_cannot_license_a_classification() -> None:
    """Writing guidance must not read as a rule that permits a label.

    The prompt has already been broken once by a rule that collapsed the answers
    it was meant to sharpen. This addition constrains only how the `description`
    field is written, so it must not read as narrowing or widening which
    classifications are available -- and it is placed before the glossary, which
    is where conclusions are defined.
    """
    prompt = _render("review_six")
    addition = prompt.index("外貌")
    glossary = prompt.index("其余标签按实际可见动作选择")
    assert addition < glossary, (
        "the writing guidance must come before the label definitions, so it "
        "cannot read as one of them"
    )
    # The addition must not name any classification; a label mentioned here would
    # attach a conclusion to a statement about prose.
    for label in SCENES["review_six"].classifications:
        segment = prompt[addition:glossary]
        assert label not in segment, (
            f"{label} appears inside the description-writing guidance, which "
            "would tie a classification to how the text is written"
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
    # Anchored on a rule from the template, not on its opening words: the layout
    # sentence now describes structure rather than a fixed cell count, so a
    # phrase-level anchor would break every time the wording is improved.
    assert prompt.index(marker) < prompt.index("才可判断cleaning")


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


def test_scene_labels_parses_name_and_definition_pairs() -> None:
    """每行一条「标签: 定义」——一个字段同时给出枚举和解释。

    两者必须同源：契约要枚举、模型要定义，分开写就会出现「提示词教了一个
    校验器不认的标签」，答案被静默丢弃。
    """
    from custom_components.frigate_vision.scenes import parse_scene_labels

    parsed = parse_scene_labels("宠物: 画面中只有宠物\n无人: 画面中没有任何人物")
    assert parsed == (("宠物", "画面中只有宠物"), ("无人", "画面中没有任何人物"))


def test_scene_labels_rejects_an_empty_definition() -> None:
    from custom_components.frigate_vision.scenes import parse_scene_labels

    with pytest.raises(ValueError, match="label_definition_missing"):
        parse_scene_labels("宠物:")


def test_scene_labels_rejects_duplicates() -> None:
    from custom_components.frigate_vision.scenes import parse_scene_labels

    with pytest.raises(ValueError, match="label_duplicate"):
        parse_scene_labels("宠物: 甲\n宠物: 乙")


def test_scene_labels_rejects_a_line_without_a_colon() -> None:
    from custom_components.frigate_vision.scenes import parse_scene_labels

    with pytest.raises(ValueError, match="label_malformed"):
        parse_scene_labels("宠物 画面中只有宠物")


def test_scene_labels_enforces_the_limits() -> None:
    from custom_components.frigate_vision.scenes import parse_scene_labels

    too_many = "\n".join(f"label_{i}: 定义" for i in range(31))
    with pytest.raises(ValueError, match="too_many_labels"):
        parse_scene_labels(too_many)
    with pytest.raises(ValueError, match="label_definition_too_long"):
        parse_scene_labels("宠物: " + "很长" * 101)
    with pytest.raises(ValueError, match="label_name_too_long"):
        parse_scene_labels("x" * 193 + ": 定义")


def test_scene_labels_ignores_blank_lines() -> None:
    """用户会在段落之间留空行；空行不该报错，也不该产生空标签。"""
    from custom_components.frigate_vision.scenes import parse_scene_labels

    parsed = parse_scene_labels("宠物: 只有宠物\n\n\n无人: 没有任何人物\n  \n")
    assert parsed == (("宠物", "只有宠物"), ("无人", "没有任何人物"))


def test_scene_labels_accepts_a_definition_containing_a_colon() -> None:
    """定义里可能有冒号（例如英文解释）。只有第一个冒号是分隔符。"""
    from custom_components.frigate_vision.scenes import parse_scene_labels

    parsed = parse_scene_labels("pet: a cat: or a dog")
    assert parsed == (("pet", "a cat: or a dog"),)


def test_empty_custom_config_reproduces_todays_prompt_byte_for_byte() -> None:
    """两个选项都空时，提示词必须与今天逐字节相同——零回归。

    这是整个改动的地基：既有部署不配置任何东西，行为就不能变。
    """
    scene = SCENES["review_six"]
    today = scene.render(
        SceneRequest(language="中文", allowed=scene.classifications, signals={})
    )
    with_empty = scene.render(
        SceneRequest(
            language="中文",
            allowed=scene.classifications,
            signals={},
            scene_labels=(),
            prompt_override="",
        )
    )
    assert with_empty == today


def test_custom_labels_replace_the_builtin_glossary() -> None:
    """自定义标签的定义必须进提示词，内置定义必须消失。"""
    scene = SCENES["review_six"]
    labels = (("宠物", "画面中只有宠物"), ("无人", "画面中没有任何人物"))
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed={name for name, _ in labels},
            signals={},
            scene_labels=labels,
        )
    )
    assert "宠物" in prompt and "画面中只有宠物" in prompt
    assert "elevator_activity" not in prompt, "内置标签不该出现在自定义配置里"


def test_the_contract_lists_exactly_the_custom_labels() -> None:
    """契约枚举必须恰好是自定义标签，一个不多一个不少。"""
    scene = SCENES["review_six"]
    labels = (("宠物", "画面中只有宠物"), ("无人", "画面中没有任何人物"))
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed={name for name, _ in labels},
            signals={},
            scene_labels=labels,
        )
    )
    contract = prompt.split("必须严格取以下之一：")[1].split("。")[0]
    assert set(contract.split("、")) == {"宠物", "无人"}


def test_prompt_override_replaces_the_rules_but_keeps_contract_and_layout() -> None:
    """覆盖只换「规则+标签定义」那一段；契约和布局必须仍在，否则答案无法解析。"""
    scene = SCENES["review_six"]
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed=scene.classifications,
            signals={},
            scene_description="现场布局：测试布局",
            prompt_override="自定义规则：只按可见动作判断。",
        )
    )
    assert "自定义规则：只按可见动作判断。" in prompt
    assert "JSON必须且只能包含三个字段" in prompt, "契约必须仍在"
    assert "测试布局" in prompt, "布局必须仍在"
    assert "关于回家与离家" not in prompt, "内置规则必须消失"


def test_override_and_custom_labels_can_be_used_together() -> None:
    """用户同时写规则和标签时，两者都要生效，且内置定义不出现。"""
    scene = SCENES["review_six"]
    labels = (("宠物", "画面中只有宠物"),)
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed={"宠物"},
            signals={},
            prompt_override="自定义规则：只判断有没有宠物。",
            scene_labels=labels,
        )
    )
    assert "自定义规则：只判断有没有宠物。" in prompt
    assert "宠物" in prompt
    assert "画面中只有宠物" in prompt
    assert "elevator_activity" not in prompt


def test_a_definition_without_a_terminator_gets_one() -> None:
    """用户手写的定义通常不带结尾标点，渲染时要补上，否则句子会黏在一起。"""
    scene = SCENES["review_six"]
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed={"宠物", "无人"},
            signals={},
            scene_labels=(("宠物", "画面中只有宠物"), ("无人", "画面中没有任何人物")),
        )
    )
    assert "画面中只有宠物；" in prompt
    assert "画面中没有任何人物；" in prompt


def test_a_definition_that_already_ends_in_punctuation_is_not_doubled() -> None:
    """已经带结尾标点的定义不能再补一个，否则出现「。。；」这样的重复标点。

    内置 glossary 的值自带结尾标点（elevator_activity 以「；」结尾，
    short_roundtrip 以「。」结尾），而 _glossary 不追加标点。用户手写时两种
    都可能出现，所以两条路径都要正确。
    """
    scene = SCENES["review_six"]
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed={"宠物", "无人"},
            signals={},
            scene_labels=(
                ("宠物", "画面中只有宠物。"),
                ("无人", "画面中没有任何人物；"),
            ),
        )
    )
    assert "画面中只有宠物。；" not in prompt, "结尾是「。」时不该再补「；」"
    assert "画面中没有任何人物；；" not in prompt, "结尾是「；」时不该再补「；」"
    assert "画面中只有宠物。" in prompt
    assert "画面中没有任何人物；" in prompt


def test_an_override_alone_still_states_the_custom_labels() -> None:
    """只写规则、不写标签时，内置标签定义仍应附加（否则模型不知道标签含义）。

    实测本部署：不解释标签含义时弃权率 55%，解释后降到 11%。标签定义与
    「用什么规则判断」是正交的两件事——覆盖规则不该顺带删掉定义。

    断言的是**定义正文**而不是标签名：标签名也出现在契约的枚举列表里，所以
    断言名字即使在定义缺失时也会通过，测不出这个行为。
    """
    from custom_components.frigate_vision.scenes import CLASSIFICATION_GLOSSARY

    scene = SCENES["review_six"]
    prompt = scene.render(
        SceneRequest(
            language="中文",
            allowed=scene.classifications,
            signals={},
            prompt_override="自定义规则：只按可见动作判断。",
        )
    )
    definition = CLASSIFICATION_GLOSSARY["elevator_activity"]
    assert definition in prompt, (
        "自定义规则不该移除标签定义：模型仍需知道每个标签的含义"
    )


def test_changing_the_labels_changes_the_cache_key() -> None:
    """改了标签就必须换键，否则旧答案会被当成新配置的结果返回。

    本项目已因「改了文本但没换 key」丢过功能两次，所以版本必须内容派生。
    """
    from custom_components.frigate_vision.scenes import effective_prompt_version

    base = effective_prompt_version("review_six", "")
    with_labels = effective_prompt_version(
        "review_six", "", scene_labels=(("宠物", "画面中只有宠物"),)
    )
    assert with_labels != base
    assert with_labels.startswith("prompt_6")


def test_changing_the_prompt_override_changes_the_cache_key() -> None:
    from custom_components.frigate_vision.scenes import effective_prompt_version

    base = effective_prompt_version("review_six", "")
    a = effective_prompt_version("review_six", "", prompt_override="规则甲")
    b = effective_prompt_version("review_six", "", prompt_override="规则乙")
    assert a != b, "不同的规则必须得到不同的键"
    assert a != base, "有覆盖时必须与无覆盖不同"


def test_changing_only_a_label_definition_changes_the_cache_key() -> None:
    """只改定义文字（标签名不变）也要换键——模型看到的内容变了。"""
    from custom_components.frigate_vision.scenes import effective_prompt_version

    a = effective_prompt_version(
        "review_six", "", scene_labels=(("宠物", "只有宠物"),)
    )
    b = effective_prompt_version(
        "review_six", "", scene_labels=(("宠物", "只有宠物或小孩"),)
    )
    assert a != b


def test_all_three_inputs_compose_into_the_key() -> None:
    """三个输入各自贡献一段摘要，任一改变都改变整个键。"""
    from custom_components.frigate_vision.scenes import effective_prompt_version

    combined = effective_prompt_version(
        "review_six",
        "布局",
        scene_labels=(("宠物", "只有宠物"),),
        prompt_override="规则甲",
    )
    assert combined.startswith("prompt_6-")
    assert combined.count("-") == 3, f"三段摘要应各自出现：{combined}"


def test_the_key_stays_a_safe_id() -> None:
    """analysis_key 会校验字母表，键必须落在 SAFE_ID 内。"""
    from custom_components.frigate_vision.models import SAFE_ID
    from custom_components.frigate_vision.scenes import effective_prompt_version

    key = effective_prompt_version(
        "review_six",
        "布局",
        scene_labels=(("宠物", "只有宠物"),),
        prompt_override="规则",
    )
    assert key is not None
    assert SAFE_ID.fullmatch(key), key


def test_nothing_configured_keeps_the_bare_base_version() -> None:
    """什么都没配时必须返回裸的基础版本，既有部署的缓存与行为完全不变。"""
    from custom_components.frigate_vision.scenes import effective_prompt_version

    assert effective_prompt_version("review_six", "") == "prompt_6"
    assert effective_prompt_version("review_six", "", scene_labels=()) == "prompt_6"
    assert effective_prompt_version("review_six", "", prompt_override="") == "prompt_6"
    blank = effective_prompt_version("review_six", "", prompt_override="   ")
    assert blank == "prompt_6"


def test_door_scenes_still_key_on_custom_labels() -> None:
    """门锁场景不接受「现场布局」，但仍必须让标签参与缓存键。

    早返回 `if not scene.accepts_scene_description: return base` 的语义是
    「这个场景是否接受现场布局描述」，不代表「是否接受自定义标签」。若把新输入
    接在那个早返回之后，门锁场景改标签就不会换键，用户会拿到上一次提示词的
    存储结果——这正是本项目已踩过两次的失败形状。
    """
    from custom_components.frigate_vision.scenes import effective_prompt_version

    for mode in ("door_single", "door_roundtrip"):
        base = effective_prompt_version(mode, "")
        with_labels = effective_prompt_version(
            mode, "", scene_labels=(("短暂外出", "出门后很快返回"),)
        )
        assert with_labels != base, f"{mode} 的标签没有参与缓存键"
        with_override = effective_prompt_version(mode, "", prompt_override="自定义规则")
        assert with_override != base, f"{mode} 的提示词覆盖没有参与缓存键"

