"""Scene definitions: what a set of frames means, and what may be concluded.

This is layer ② of the integration. The core (frame selection, repair, contact
sheets, the model call, delivery) is scenario-agnostic; a *scene* supplies the
scenario-specific parts:

  * the classifications that may be returned,
  * the prompt that explains what the frames show and what may not be inferred,
  * which auxiliary signals it is allowed to read,
  * the prompt version, which keys the analysis cache.

Adding a scenario means registering a scene here. Nothing in the core needs to
change, and a new scene cannot accidentally depend on another's context because
it only ever receives the signals it declares.

Why the isolation matters: the door scenes read door-lock state, and the general
review scene must not. Measured on this deployment, no zone telemetry reaches the
standalone-review path at all (0 of 90 reviews carried `detection_zone_updates`,
while door cycles did), so a general prompt that implied door knowledge would be
asking the model to invent evidence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field

# Bumped whenever any scene's wording changes in a way that could alter its
# answer. It is part of the analysis cache key, so a stale result is never
# reused after the prompt that produced it has changed.
#
# A deployment-supplied scene description changes the prompt without changing
# this constant -- see `effective_prompt_version`, which folds that description
# into the key so a user editing it is not silently served a cached answer.
PROMPT_VERSION = "prompt_6"


@dataclass(frozen=True, slots=True)
class SceneRequest:
    """Everything a scene may use to render its prompt."""

    language: str
    # The classifications the caller will accept. Empty means "do not state a
    # contract", which is what the unit tests of the renderer use.
    allowed: frozenset[str] | set[str]
    # Auxiliary evidence, e.g. door-lock state. A scene reads only the keys it
    # declared, so an undeclared signal can never reach its prompt.
    signals: Mapping[str, object] = field(default_factory=dict)
    # The deployment's own description of what the camera looks at. Injected
    # only into scenes that declare `accepts_scene_description`.
    scene_description: str = ""


@dataclass(frozen=True, slots=True)
class Scene:
    """One scenario: its answers, its auxiliary inputs, and its prompt."""

    mode: str
    classifications: frozenset[str]
    signals: frozenset[str]
    prompt_version: str
    template: str
    # Extra text appended per declared signal, keyed by signal name. Kept
    # separate from `template` so a scene's wording stays readable.
    signal_text: Mapping[str, Mapping[object, str]] = field(default_factory=dict)
    preamble: str = ""
    # Whether the deployment's own description of the camera's view may be
    # injected. Declared rather than assumed, for the same reason `signals` is:
    # a scene only ever receives input it asked for, so a description written
    # for one layout cannot leak into a scene that would misread it.
    accepts_scene_description: bool = False

    def render(self, request: SceneRequest) -> str:
        """Return the prompt for this scene.

        The response contract is appended rather than embedded, because the
        allowed set is supplied per call: the endpoint rejects
        `response_format: json_schema`, so the model only knows the permitted
        values if the prompt lists them.
        """
        parts = [self._common(request.language)]
        if request.allowed:
            parts.append(self._contract(request.allowed))
        # Before the template, not after: the template's rules refer to "the
        # front door" and to movement relative to it, so the model has to be
        # told which door that is before it is told how to reason about it.
        if self.accepts_scene_description:
            parts.append(_scene_context(request.scene_description))
        parts.append(self.template)
        # After the template: these are definitions of what the remaining words
        # mean, so they belong with the rules rather than before them.
        parts.append(self._glossary(request.allowed))
        for name in sorted(self.signals):
            # Only declared signals are consulted; anything else in the request
            # is deliberately ignored.
            for value, text in self.signal_text.get(name, {}).items():
                if request.signals.get(name) == value:
                    parts.append(text)
        parts.append(self.preamble)
        return "".join(part for part in parts if part)

    @staticmethod
    def _common(language: str) -> str:
        return (
            f"请使用{language}，严格返回符合给定结构的JSON对象。"
            "画面、zone与门锁都只是时间和路线候选，不能直接当成身份或行为结论。"
            "只描述实际可见动作；证据不足必须选择unknown_activity或unable_to_confirm。"
        )

    @staticmethod
    def _contract(allowed: frozenset[str] | set[str]) -> str:
        values = "、".join(sorted(allowed))
        return (
            "只输出一个JSON对象，不要输出任何其他文字、解释或代码块标记。"
            'JSON必须且只能包含三个字段："classification"、"description"、"confidence"。'
            f'"classification"必须严格取以下之一：{values}。'
            '"description"是字符串，只描述实际可见动作，不超过500字。'
            '"confidence"是0到100的整数。'
        )

    def _glossary(self, allowed: frozenset[str] | set[str]) -> str:
        """Definitions for the labels this call offers but the rules leave open.

        Measured on this deployment: the prompt offered twelve answers and
        explained five, while naming `unknown_activity` three times and
        `unable_to_confirm` three times. Across 41 real activities the
        deployment abstained on 55%; defining these words took that to 11% and
        raised accuracy on the owner's labelled sheets from 53% to 60%. Being
        told what not to say does not tell the model what to say.

        Filtered by `allowed`, not by the scene's own list: the allowed set
        travels per call and a caller may offer a subset, so a definition for an
        unoffered label would invite an out-of-contract answer. Labels the
        template already defines are omitted rather than repeated.
        """
        defined = {label: text for label, text in CLASSIFICATION_GLOSSARY.items()}
        # `elevator_activity` and `visitor` sit in the glossary rather than the
        # template because their wording is a definition, not a rule.
        offered = sorted(label for label in allowed if label in defined)
        if not offered:
            return ""
        return "其余标签按实际可见动作选择：" + "".join(
            f"{label}{defined[label]}" for label in offered
        )


# One sentence per label the template's rules leave undefined. Kept as data
# rather than prose inside `template` because the definitions must follow the
# per-call `allowed` set: teaching a label the caller disallowed would invite an
# out-of-contract answer.
#
# `cleaning`, `home_arrival` and `home_departure` are absent deliberately -- the
# template already states their conditions, and repeating them here would show
# the model two versions of the same rule.
CLASSIFICATION_GLOSSARY: Mapping[str, str] = {
    "elevator_activity": "指人物在电梯间内活动但没有跨越入户门；",
    "package_delivery": "指放下快递包裹且离开后包裹仍留在原处；",
    "food_delivery": "指放下餐饮外卖且离开后餐食仍留在原处；",
    "visitor": "指访客到访，例如敲门、在门外等候或被迎入；",
    "maintenance": "指维修人员对楼道设施进行作业；",
    "suspicious_activity": "指试探门锁、反复徘徊或窥探等可疑行为；",
    "short_roundtrip": "指短暂外出后随即返回。",
    "unknown_activity": "指画面确实无法支持任何其他判断；",
    "unable_to_confirm": "指证据不足或画面质量导致无法判断。",
}


# Wraps the deployment's description so the model reads it as context rather
# than as another rule. The closing sentence matters: a description alone tells
# the model what it is looking at, while this tells it what may *not* be
# concluded from that -- which is the part that actually prevents the
# "saw someone leave a lift, called it a departure" error.
def _scene_context(description: str) -> str:
    text = description.strip()
    if not text:
        return ""
    return (
        f"现场布局：{text}"
        "以上只是摄像头视野的说明，用于判断画面中的门和通道分别是什么；"
        "仍须按下面的规则根据实际可见动作判断，不得据布局猜测。"
    )


# The general scene. Its phrasing carries the arrival/departure rules because no
# zone telemetry reaches this path (measured: 0 of 90 standalone reviews carried
# detection_zone_updates, while door cycles did), so direction has to be seen in
# the frames rather than inferred from timing or frame order.
_REVIEW_SIX = Scene(
    mode="review_six",
    classifications=frozenset(
        {
            "home_arrival",
            "home_departure",
            "package_delivery",
            "food_delivery",
            "cleaning",
            "maintenance",
            "visitor",
            "elevator_activity",
            "suspicious_activity",
            "unknown_activity",
            "unable_to_confirm",
        }
    ),
    signals=frozenset(),
    prompt_version=PROMPT_VERSION,
    # Measured on this deployment: without a layout description the model read a
    # person stepping out of a lift and walking away down the corridor as
    # `home_departure`, because the prompt told it to judge movement relative to
    # "the front door" without ever saying that the door in frame is a lift and
    # the front door is off-camera. The rule was right; the missing context was
    # the door's identity.
    accepts_scene_description=True,
    template=(
        # Structure, not a count. The sheet is 2x3 when no hole is wide enough to
        # probe and 3x3 when one is, so naming a number would be wrong half the
        # time -- and wrong in the direction that matters, since it would place the
        # emptied scene three cells early.
        "图按时间顺序排列，从左到右、再从上到下读取。"
        "第一格是人物首帧，最后一格是人物离开后的后置现场，倒数第二格是人物末帧，"
        "中间各格是这段时间内的变化候选。"
        "只有连续清楚出现清扫、扫地、拖地或擦拭动作才可判断cleaning。"
        "只有清楚看到放下包裹或外卖且离开后物品仍留下，才可判断配送。"
        "不能根据制服、携带物或停留时间猜测职业。"
        "关于回家与离家：只有画面中能看出人物相对入户门的移动方向时才可判断——"
        "人物从门内走出并远离该门，判为home_departure；"
        "人物从门外走近该门并进入门内，判为home_arrival。"
        "无法看出门的开合或人物的进出方向时，不得据停留时长、出现顺序或"
        "是否携带物品推断方向，应选择unknown_activity或unable_to_confirm。"
        "若人物只是经过、在门前停留或整理物品而未跨越门，也不要判断回家或离家。"
        # Guidance on writing the `description` field, which is the notification
        # body the household reads. Placed after the classification rules and
        # before the glossary so it cannot be read as one of them: it constrains
        # prose, not conclusions. The "write that you cannot tell" clause matters
        # because most of this deployment's night evidence is infrared and a
        # distant figure has no visible clothing -- without it, "must describe the
        # appearance" invites a fabricated outfit.
        "描述人物时须写明可辨认的外貌（性别、发型、上下身衣着的颜色与款式）；"
        "看不清就写明看不清，不得补足。"
        "画面中没有清晰正面面部时，不得推断年龄，改用中性说法"
        "（一名男子、一名女士或一名人员）。"
        "描述须交代人物从何处来、往何处去，以及门与电梯的状态变化，动作要连贯。"
    ),
)

# Door scenes read the lock, so they declare it. `door_single` uses the opening
# side; `door_roundtrip` uses both.
_DOOR_SINGLE = Scene(
    mode="door_single",
    classifications=frozenset(
        {"home_arrival", "home_departure", "unknown_activity", "unable_to_confirm"}
    ),
    signals=frozenset({"opening_side"}),
    prompt_version=PROMPT_VERSION,
    template="门锁只给出开门和关门边界，各格是近端、路径和远端候选。",
    signal_text={
        "opening_side": {
            "inside": "室内侧开门支持离家候选，但不是结论。",
            "outside": "室外侧开门支持回家候选，但不是结论。",
            "unknown": "开门操作侧未知，不能据此判断回家或离家。",
        }
    },
    preamble="不能仅凭电梯方向断言进入可见电梯；右侧画外还有另一部电梯。",
)

_DOOR_ROUNDTRIP = Scene(
    mode="door_roundtrip",
    classifications=frozenset(
        {"short_roundtrip", "unknown_activity", "unable_to_confirm"}
    ),
    signals=frozenset({"door_remained_open", "opening_side"}),
    prompt_version=PROMPT_VERSION,
    template=(
        "这是六帧两阶段联系图：上排三格是出去候选，下排三格是返回候选，"
        "最终在返回候选后关门。"
    ),
    signal_text={
        "door_remained_open": {
            True: "门磁支持活动期间持续开启。",
            False: "门磁记录与持续开启候选冲突，不能确认短时往返。",
            None: "没有门磁连续开启证据，门是否持续开启未知。",
        }
    },
    preamble="不能猜测同一人物或倒垃圾目的。",
)


SCENES: dict[str, Scene] = {
    scene.mode: scene for scene in (_REVIEW_SIX, _DOOR_SINGLE, _DOOR_ROUNDTRIP)
}


def scene_for(mode: str) -> Scene | None:
    """Return the scene for a mode, or None when it is not registered."""
    return SCENES.get(mode)


def scene_prompt_version(mode: str) -> str | None:
    """Return the prompt version for a mode, or None when unregistered."""
    scene = SCENES.get(mode)
    return scene.prompt_version if scene is not None else None


def effective_prompt_version(
    mode: str, scene_description: str = ""
) -> str | None:
    """Return the version to key an analysis on, for a mode and a description.

    The base version is a module constant, so it cannot reflect a description
    the deployment supplies from its own options. Without folding that
    description in, editing it would leave the cache key untouched and the next
    analysis of an activity would be answered from the *previous* prompt's
    stored result -- the change would look like it had no effect at all. This
    project has already lost a feature to exactly that shape of failure twice
    (a blueprint silently rewritten by Home Assistant, and a template value
    that stopped being a string), so the version is derived from the content
    rather than trusted to be bumped by hand.

    Returns None for an unregistered mode, matching `scene_prompt_version`.

    The digest is eight hex characters, which keeps the result inside the
    `SAFE_ID` alphabet that `analysis_key` validates. Eight characters is far
    more than enough to distinguish the handful of descriptions a deployment
    would ever write, and it is a cache key rather than a security boundary.
    """
    scene = SCENES.get(mode)
    if scene is None:
        return None
    base = scene.prompt_version
    if not scene.accepts_scene_description:
        return base
    text = scene_description.strip()
    if not text:
        # Nothing was configured, so the prompt is exactly the base one and the
        # key must stay exactly the base key. Deployments that never set a
        # description keep their existing cache and behaviour untouched.
        return base
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return f"{base}-{digest}"
