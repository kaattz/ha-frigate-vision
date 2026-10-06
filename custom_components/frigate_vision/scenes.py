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
from dataclasses import dataclass, field, replace
from functools import cache
from pathlib import Path

from .const import MAX_LABEL_DEFINITION_LENGTH, MAX_SCENE_LABELS
from .models import MAX_CLASSIFICATION_LENGTH, SAFE_CLASSIFICATION

# Where the built-in prompt text lives.
#
# It used to be a Python string literal, and it had to be kept in step with a
# second literal in `const.py` (`DEFAULT_PROMPT_OVERRIDE`) by a test that
# asserted the two were equal. That test was the only thing holding them
# together: nothing stopped an edit to one, and the failure it produced said
# "the prefill disagrees with the built-in rules" rather than "you edited one of
# two copies". Reading one file from both places removes the second copy, so the
# property is now structural rather than asserted.
#
# Data, not code, because the wording is something the household tunes: it names
# the doors this camera sees and the words its answers are allowed to use.
PROMPTS_DIR = Path(__file__).parent / "prompts"


class PromptDataError(RuntimeError):
    """A built-in prompt file is missing or unreadable.

    Raised rather than falling back to an empty prompt: a scene with no rules
    would still call the model, still be billed, and answer from the contract
    alone -- which is exactly the failure that looks like "the model got worse".
    """


@cache
def load_prompt_text(mode: str, name: str) -> str:
    """Return one built-in prompt file's contents, cached per process.

    Cached because `Scene.render` runs per analysis and the files cannot change
    while the integration is loaded; re-reading them would put disk I/O on the
    request path for no benefit. A restart picks up an edit, which is the same
    contract the rest of the integration's configuration has.
    """
    path = PROMPTS_DIR / mode / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PromptDataError(f"prompt_file_unreadable:{mode}/{name}") from exc
    if not text.strip():
        raise PromptDataError(f"prompt_file_empty:{mode}/{name}")
    return text


def load_scene_labels(mode: str) -> Mapping[str, str]:
    """Return the built-in label definitions for a scene, parsed from its file.

    Reuses `parse_scene_labels` so the file format is the same one the options
    field accepts -- a user can copy a line out of `labels.txt` into their own
    configuration and it means the same thing.
    """
    text = load_prompt_text(mode, "labels.txt")
    try:
        return dict(parse_scene_labels(text))
    except ValueError as exc:
        raise PromptDataError(f"prompt_labels_invalid:{mode}:{exc}") from exc


def load_prefill_labels(mode: str) -> tuple[tuple[str, str], ...]:
    """Return the ordered label set the options form prefills.

    Deliberately separate from `load_scene_labels`, and not derivable from it.
    The glossary explains only the labels the rules leave undefined; the form
    prefill has to carry *every* answer the scene can give, including the three
    whose conditions are stated in `rules.txt` instead. Measured on this
    deployment: the glossary holds 8, the prefill needs 11, and `scene_labels`
    replaces the allowed set wholesale -- so a prefill missing one label makes
    that answer permanently unreachable, with no error to notice it by.

    Order is the file's, which is alphabetical, matching `_glossary`'s `sorted()`.
    """
    text = load_prompt_text(mode, "prefill_labels.txt")
    try:
        return parse_scene_labels(text)
    except ValueError as exc:
        raise PromptDataError(f"prompt_prefill_labels_invalid:{mode}:{exc}") from exc


BUILTIN_SCENE_LABELS: tuple[tuple[str, str], ...] = ()


# The base prompt version is DERIVED from the rendered prompt rather than typed,
# and is computed at the bottom of this module (`PROMPT_VERSION`) because the
# digest needs the scene to exist.
#
# It is part of the analysis cache key, so a stale result is never reused after
# the prompt that produced it has changed. It used to be a hand-written string
# ("prompt_6") guarded by a test that pinned the digest of the rendered prompt --
# a test whose failure message had to tell the next person to bump the version
# *and* update the digest. Deriving it removes that ritual: edit the wording and
# the key moves by construction.
#
# A deployment-supplied scene description changes the prompt without changing
# this value -- see `effective_prompt_version`, which folds that description into
# the key so a user editing it is not silently served a cached answer.


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
    # 该 entry 自定义的标签与定义，有序。空表示用场景内置的标签集。
    scene_labels: tuple[tuple[str, str], ...] = ()
    # 该 entry 自定义的规则全文。空表示用场景内置的 template。
    prompt_override: str = ""
    # 图上是否多了一栏人物特写。默认 False，既有的构造调用与渲染逐字节不变。
    has_person_highlight: bool = False


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

        契约与布局总是自动附加：前者是解析器依赖的格式，去掉答案就无法解析；
        后者是该摄像头视野的事实，规则里引用「入户门」等词依赖它。

        自定义配置只替换**规则那一段**——内置的 template + 标签定义，或用户的
        覆盖文本。这是「整段覆盖」的含义：用户写自己的判断依据，而不是在默认
        规则上追加。

        覆盖与自定义标签是正交的：只写覆盖时，内置标签定义仍会附加（不解释
        标签含义会让弃权率从 11% 升到 55%）；只写标签时，内置 template 仍会
        附加。两者都没写时，行为与加这个功能之前完全一致。
        """
        parts = [self._common(request.language)]
        if request.allowed:
            parts.append(self._contract(request.allowed))
        # Before the template, not after: the template's rules refer to "the
        # front door" and to movement relative to it, so the model has to be
        # told which door that is before it is told how to reason about it.
        if self.accepts_scene_description:
            parts.append(_scene_context(request.scene_description))
        override = request.prompt_override.strip()
        if override:
            parts.append(override)
        else:
            parts.append(self.template)
        # 只加在规则之后、标签定义之前：先让模型读完判断规则，再告诉它图上
        # 有两栏。不启用时一个字都不加，既有部署的提示词逐字节不变。
        if request.has_person_highlight:
            parts.append(_PERSON_HIGHLIGHT)
        # 标签定义：自定义优先，否则用内置的。覆盖不影响这一选择——只写规则
        # 时内置定义仍要出现：定义与「用什么规则判断」是正交的两件事。
        if request.scene_labels:
            parts.append(_custom_glossary(request.scene_labels))
        else:
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


# One sentence per label the template's rules leave undefined. Loaded from
# `prompts/review_six/labels.txt` rather than written here, so the wording lives
# with the rules it complements and a household can read both in one place.
#
# Kept as data rather than prose inside the template because the definitions must
# follow the per-call `allowed` set: teaching a label the caller disallowed would
# invite an out-of-contract answer.
#
# `cleaning`, `home_arrival` and `home_departure` are absent deliberately -- the
# template already states their conditions, and repeating them here would show the
# model two versions of the same rule. The form prefill is a *different* set and
# does include them; see `load_prefill_labels`.
CLASSIFICATION_GLOSSARY: Mapping[str, str] = {}


def parse_scene_labels(text: str) -> tuple[tuple[str, str], ...]:
    """把「标签: 定义」的每行文本解析成有序对。

    一个字段同时给出枚举和定义，因为契约要枚举、模型要定义，而两者必须同源：
    分开配置就会出现「提示词教了一个校验器不认的标签」，答案被静默丢弃。

    定义不能省。实测本部署，不解释标签含义时弃权率 55%，解释后降到 11%——
    只给标签名不给定义，等于让模型猜。

    用第一个冒号作为分隔符，这样定义里可以再出现冒号（英文解释常见）。
    空白行忽略：用户在段落之间留空行是正常的。

    标签名用 `SAFE_CLASSIFICATION` 校验，而不是只数长度：这个标签会进 allowed
    集合、进契约枚举，模型照契约回它，最后写入 `ActivityRecord.classification`。
    解析器比存储层宽就会漂移，而漂移在这里不报错——它要到写入时才炸，那时
    provider 已经计费，异常还不是 `VisionError`，于是变成不可重试的
    `analysis_outcome_unknown`。两边同源，就不会有这段距离。

    抛出 ValueError，其消息就是错误码，供配置流程直接展示给用户。
    """
    labels: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" not in line:
            raise ValueError("label_malformed")
        name, _, definition = line.partition(":")
        name = name.strip()
        definition = definition.strip()
        if not name:
            raise ValueError("label_malformed")
        # 长度先报，再报字符集：193 个 ASCII 字符同时违反两条，但「太长」是
        # 用户更能直接照做的那个提示。
        if len(name) > MAX_CLASSIFICATION_LENGTH:
            raise ValueError("label_name_too_long")
        if not SAFE_CLASSIFICATION.fullmatch(name):
            raise ValueError("label_name_invalid")
        if not definition:
            raise ValueError("label_definition_missing")
        if len(definition) > MAX_LABEL_DEFINITION_LENGTH:
            raise ValueError("label_definition_too_long")
        if name in seen:
            raise ValueError("label_duplicate")
        seen.add(name)
        labels.append((name, definition))
    if len(labels) > MAX_SCENE_LABELS:
        raise ValueError("too_many_labels")
    return tuple(labels)


def _custom_glossary(labels: tuple[tuple[str, str], ...]) -> str:
    """用该 entry 自己的定义渲染标签说明。

    与 `_glossary` 同形，但不过滤：自定义标签集就是用户为这个摄像头写的全部
    答案，没有「场景定义了但这次没提供」的情况。

    定义末尾没有标点时补一个「；」——用户手写时通常不带，不补的话相邻两条定义
    会黏成一句。已经有标点（「。」或「；」）时不重复添加。
    """
    if not labels:
        return ""
    parts: list[str] = []
    for name, definition in labels:
        text = definition.strip()
        if text and text[-1] not in "。；;.":
            text += "；"
        parts.append(f"{name}{text}")
    return "其余标签按实际可见动作选择：" + "".join(parts)


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


# Tells the model what the extra column on the right is.
#
# Without this the model reads that column as a tenth time step, which reverses
# the direction it reports -- and direction is what this scene's arrival and
# departure rules rest on. Those rules already have to read direction off the
# frames rather than off timing (see `_REVIEW_SIX`: no zone telemetry reaches
# this path), so a column that displaces the frame order costs the scene its only
# remaining evidence. A column merely *described* as a close-up still invites the
# model to place it somewhere in the sequence, so the exclusion sentence is the
# payload here rather than a caveat: it is what stops the column from being read
# as evidence about order.
#
# Says "各帧" rather than naming a count, for the reason the template does: the
# grid is 2x3 when no hole is wide enough to probe and 3x3 when one is, so
# "九宫格" would be wrong half the time -- and wrong in the direction that
# matters, since it would place the emptied scene three cells early.
_PERSON_HIGHLIGHT = (
    "画面分成两栏。左侧是本次事件过程的各帧网格图，按时间从左到右、"
    "再从上到下读取，用于判断人物是靠近还是远离。"
    "右侧单独一栏是同一事件中人物最清晰的一帧放大特写，"
    "用于辨认衣着、携带物等外观特征。"
    "它不是时间序列的一部分，不要据它判断先后顺序。"
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
    # Filled in below, once the scene's own text is available to hash. A scene
    # cannot hash itself while it is being constructed.
    prompt_version="",
    # Measured on this deployment: without a layout description the model read a
    # person stepping out of a lift and walking away down the corridor as
    # `home_departure`, because the prompt told it to judge movement relative to
    # "the front door" without ever saying that the door in frame is a lift and
    # the front door is off-camera. The rule was right; the missing context was
    # the door's identity.
    accepts_scene_description=True,
    # Read from `prompts/review_six/rules.txt`. The structure-not-a-count rule
    # that used to be a comment here still holds and is worth keeping in mind
    # when editing that file: the sheet is 2x3 when no hole is wide enough to
    # probe and 3x3 when one is, so naming a cell count would be wrong half the
    # time -- and wrong in the direction that matters, since it would place the
    # emptied scene three cells early.
    #
    # The description-field guidance stays in the same file, after the
    # classification rules so it cannot be read as one of them: it constrains
    # prose, not conclusions. The "write that you cannot tell" clause matters
    # because most of this deployment's night evidence is infrared and a distant
    # figure has no visible clothing -- without it, "must describe the
    # appearance" invites a fabricated outfit.
    template=load_prompt_text("review_six", "rules.txt"),
)

SCENES: dict[str, Scene] = {scene.mode: scene for scene in (_REVIEW_SIX,)}


def scene_for(mode: str) -> Scene | None:
    """Return the scene for a mode, or None when it is not registered."""
    return SCENES.get(mode)


def scene_prompt_version(mode: str) -> str | None:
    """Return the prompt version for a mode, or None when unregistered."""
    scene = SCENES.get(mode)
    return scene.prompt_version if scene is not None else None


def effective_prompt_version(
    mode: str,
    scene_description: str = "",
    *,
    scene_labels: tuple[tuple[str, str], ...] = (),
    prompt_override: str = "",
    has_person_highlight: bool = False,
) -> str | None:
    """Return the version to key an analysis on.

    三个可配置输入都要折进键里。基础版本是模块常量，反映不了部署自己的配置；
    不折进去的话，改了配置而键不变，下一次分析会直接返回**上一次提示词**的存储
    结果，改动看起来完全没生效。本项目已因这种形状丢过功能两次，所以版本内容
    派生，不靠手改版本号。

    未配置任何东西时返回裸的基础版本，既有部署的缓存和行为完全不变。

    注意早返回只针对**现场布局**：`accepts_scene_description` 的语义是「这个
    场景是否接受布局描述」，不代表「是否接受自定义标签」。门锁场景不接受布局，
    但它的标签和规则改了同样必须换键，否则会拿到旧答案。

    Returns None for an unregistered mode, matching `scene_prompt_version`.

    Each digest is eight hex characters, which keeps the result inside the
    `SAFE_ID` alphabet that `analysis_key` validates. Eight characters is far
    more than enough to distinguish the handful of inputs a deployment would
    ever write, and it is a cache key rather than a security boundary.
    """
    scene = SCENES.get(mode)
    if scene is None:
        return None
    base = scene.prompt_version
    parts: list[str] = []
    # A description only counts for a scene that declares it accepts one; the
    # labels and the override below are not gated on that flag.
    if scene.accepts_scene_description:
        text = scene_description.strip()
        if text:
            parts.append(text)
    if scene_labels:
        # 名字和定义都参与：只改定义也会改变模型看到的内容。
        parts.append(
            "\n".join(f"{name}:{definition}" for name, definition in scene_labels)
        )
    override = prompt_override.strip()
    if override:
        parts.append(override)
    # 与 `render()` 用同一个条件：图上多一栏时提示词就多一段，键也必须多一段。
    # 两个函数各用各的条件，claim 键就会与 return 键不同，抛
    # `side_effect_key_mismatch`——而那发生在 provider 已经计费之后。
    if has_person_highlight:
        parts.append("person_highlight")
    if not parts:
        # Nothing was configured, so the prompt is exactly the base one and the
        # key must stay exactly the base key. Deployments that never configure
        # anything keep their existing cache and behaviour untouched.
        return base
    digests = [
        hashlib.sha256(part.encode("utf-8")).hexdigest()[:8] for part in parts
    ]
    return base + "".join(f"-{digest}" for digest in digests)


def prompt_version_for(mode: str) -> str:
    """Return the cache-key version for a scene, derived from its prompt body.

    The digest covers what actually reaches the model for a representative call:
    the template, the contract and the glossary, in their rendered order. Hashing
    the rendered text rather than the template alone means a change to any of the
    three moves the key, which is the property the key exists for.

    Twelve hex characters, prefixed with `p`: long enough that two prompt
    revisions cannot collide in practice, and inside the `SAFE_ID` alphabet
    `analysis_key` validates. The prefix keeps a derived version recognisable
    beside the hand-written ones already stored on existing records -- those are
    `prompt_6`, and `SafeId` accepts both, so no migration is needed.

    Raises `PromptDataError` for an unregistered mode rather than inventing a
    version, matching `scene_prompt_version` returning None.
    """
    scene = SCENES.get(mode)
    if scene is None:
        raise PromptDataError(f"prompt_scene_unregistered:{mode}")
    rendered = scene.render(
        SceneRequest(language="zh-CN", allowed=set(scene.classifications))
    )
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:12]
    return f"p{digest}"


# Populate the glossary first: `Scene.render` reads it through the module global
# and `_glossary` looks it up at call time, so deriving the version before this
# line would hash a prompt that has no label definitions in it -- a real ordering
# bug, caught by the derived version disagreeing with a fresh render.
#
# Reuses the same helper the options field uses, so a line copied out of
# `labels.txt` into a user's own configuration still means the same thing.
CLASSIFICATION_GLOSSARY = load_scene_labels("review_six")

# The form's prefilled label set: every answer the scene can give, which is a
# superset of the glossary -- `cleaning`, `home_arrival` and `home_departure`
# have their conditions in `rules.txt`, so they carry no glossary entry.
BUILTIN_SCENE_LABELS = load_prefill_labels("review_six")

# The derived base version, bound now that the scene, its glossary and
# `prompt_version_for` all exist. Every scene shares it because the registry
# holds one scene; a second scene would need its own derived value on its
# `Scene`.
PROMPT_VERSION = prompt_version_for("review_six")

# Bind the derived version onto the registered scene, replacing the placeholder
# that had to stand in while the scene was being built.
SCENES["review_six"] = replace(_REVIEW_SIX, prompt_version=PROMPT_VERSION)
