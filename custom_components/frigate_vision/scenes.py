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

from collections.abc import Mapping
from dataclasses import dataclass, field

# Bumped whenever any scene's wording changes in a way that could alter its
# answer. It is part of the analysis cache key, so a stale result is never
# reused after the prompt that produced it has changed.
PROMPT_VERSION = "prompt_3"


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
        parts.append(self.template)
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
    template=(
        "六格依次是人物首帧、三张变化候选、人物末帧和后置现场。"
        "只有连续清楚出现清扫、扫地、拖地或擦拭动作才可判断cleaning。"
        "只有清楚看到放下包裹或外卖且离开后物品仍留下，才可判断配送。"
        "不能根据制服、携带物或停留时间猜测职业。"
        "关于回家与离家：只有画面中能看出人物相对入户门的移动方向时才可判断——"
        "人物从门内走出并远离该门，判为home_departure；"
        "人物从门外走近该门并进入门内，判为home_arrival。"
        "无法看出门的开合或人物的进出方向时，不得据停留时长、出现顺序或"
        "是否携带物品推断方向，应选择unknown_activity或unable_to_confirm。"
        "若人物只是经过、在门前停留或整理物品而未跨越门，也不要判断回家或离家。"
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
