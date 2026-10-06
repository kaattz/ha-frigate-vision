"""Constants for Frigate Vision."""

from __future__ import annotations

from typing import Any

DOMAIN = "frigate_vision"

AUTH_NONE = "none"
AUTH_NATIVE = "native"

CONF_NAME = "name"
CONF_BASE_URL = "base_url"
CONF_AUTH_MODE = "auth_mode"
CONF_MQTT_TOPIC_PREFIX = "mqtt_topic_prefix"
CONF_CAMERA = "camera"
CONF_NEAR_ZONES = "near_zones"
CONF_TRANSITION_ZONES = "transition_zones"
CONF_FAR_ZONES = "far_zones"

# Zones whose change is worth an extra evidence cell.
#
# Separate from the near/transition/far roles above, and deliberately so: those
# three describe where a zone sits on a route, while these say "the picture
# changing HERE matters more than the picture changing on a blank wall". A zone
# can be both, neither, or one of the two.
#
# Names, not coordinates. The regions are drawn in Frigate and read back from
# `GET /api/config`, so the household edits one drawing rather than keeping two
# in step -- and a region that Frigate uses to raise the review is exactly the
# region whose change is worth a cell.
#
# Stored as a LIST in `entry.options` and empty by default: empty means the
# change analysis runs on the whole frame alone, which is the behaviour that
# exists before anyone configures a region.
CONF_FOCUS_ZONES = "focus_zones"

# Vision provider settings. Held by this integration rather than borrowed from
# another component, so the request body is fully under its control -- notably
# the reasoning toggle, which dominates cost.
CONF_LLM_BASE_URL = "llm_base_url"
CONF_LLM_API_KEY = "llm_api_key"
CONF_LLM_MODEL = "llm_model"
CONF_LLM_THINKING = "llm_thinking"
CONF_LLM_REASONING_EFFORT = "llm_reasoning_effort"
# 第二组供应商。只重复「谁来回话」的字段；提示词、图片宽度、语言等共享，因为
# 换供应商不该改变问题本身。
CONF_FALLBACK_LLM_BASE_URL = "fallback_llm_base_url"
CONF_FALLBACK_LLM_API_KEY = "fallback_llm_api_key"
CONF_FALLBACK_LLM_MODEL = "fallback_llm_model"
CONF_FALLBACK_LLM_THINKING = "fallback_llm_thinking"
CONF_FALLBACK_LLM_REASONING_EFFORT = "fallback_llm_reasoning_effort"
CONF_LLM_PROVIDER = "llm_provider"
CONF_LLM_BASE_URL_DEFAULT = "https://api.deepseek.com/v1"
CONF_LLM_PROVIDER_DEFAULT = "deepseek"

# 证据图右侧是否再附一栏人物放大特写。九宫格里的人物只有 59x74 px，辨认衣着
# 与携带物基本不可能；右侧特写栏能到 255-415 px，模型才真的看得清。
#
# 默认**关**，因为这是一个行为变更：它改变发给模型的图，并且会改变缓存键
# （`effective_prompt_version` 把开关折进键里）。默认开着会让每一个既有部署在
# 升级后立刻换键、缓存全失效、每次分析都多发一张图——而用户并没有要求这个。
# 默认关则提示词与键都逐字节不变（Task 3 已验证不传 highlight 时拼图输出不变），
# 只有显式打开的人才付出那点额外费用。
#
# 与 `analyze_night_unknown` 这类既有选项一样取保守默认：新功能不改变既有行为。
CONF_PERSON_HIGHLIGHT = "person_highlight"
CONF_PERSON_HIGHLIGHT_DEFAULT = False

# There is deliberately no width constant for the person close-up.
#
# There used to be (`PERSON_HIGHLIGHT_WIDTH = 448`), from when the close-up was a
# COLUMN beside the grid and the provider budget had to add its width. The close-up is
# now a strip UNDER the grid, so it adds height and never width: a composed sheet is
# exactly `target_width` wide, `evidence_width_budget` returns `target_width`, and
# `build_contact_sheet` takes no width for it.
#
# The constant was removed rather than left unused because the provider budget spent it
# (`target_width + this`), which is what let a sheet with no close-up be delivered 448
# pixels wider than configured -- 2.5x the pixel area, billed, for a strip that was not
# in the image. See `evidence_width_budget` and the test that pins it.

# Blank pixels between the frame grid and the close-up strip below it. The two are
# different scenes, and a model reading straight down across the boundary would
# otherwise see the grid's last row as continuous with the crop.
HIGHLIGHT_SEAM = 1

# Optional face-detection service, used to choose *which* frame the close-up comes
# from. Empty by default, and empty means the close-up behaves exactly as it did
# before this option existed.
#
# It is a URL rather than a switch on purpose: the service is a separate container
# the user may or may not be running, so "where is it" and "is it on" are the same
# question, and an empty string answers it without a second setting to keep in step.
#
# The value is an improvement, never a dependency. The integration's own rule --
# the largest detection box -- always runs first and is what gets used when this is
# unset, unreachable, slow, or wrong. Verified against a real activity: the largest
# box was a frame of the person's back, and the face service moved the choice to a
# frontal frame six frames earlier. With the service stopped, the same activity
# must produce the same sheet it produces today.
CONF_FACE_SERVICE_URL = "face_service_url"
CONF_FACE_SERVICE_URL_DEFAULT = ""

# How long to wait for one face check. The service measures ~25 ms on a real crop;
# this is generous enough to absorb a cold start or a busy host, and short enough
# that a wedged service cannot hold an activity's analysis open. A timeout is an
# ordinary answer -- the largest-box choice stands.
FACE_SERVICE_TIMEOUT = 4.0

# Candidates sent for a face check. The close-up is only built once per activity
# and these are HTTP round trips to another host, so this is deliberately small:
# enough to cover a person walking through the frame, not the whole clip.
FACE_SERVICE_CANDIDATES = 6

# Known OpenAI-compatible endpoints, so the base URL does not have to be typed by
# hand. That URL is the field most easily got wrong: it must be the
# OpenAI-compatible *root*, and providers disagree about the shape. Google's is the
# clearest example -- it carries both a version segment and an `openai` segment, so
# the `https://host/v1` form that suits every other provider here returns 404.
#
# Only `base_url` and a list of model names to suggest are recorded. The key is
# the user's, and the model is a cost decision, so a preset supplies neither.
PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ("deepseek-v4.1-flash", "deepseek-v4-pro"),
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "models": ("gemini-3.8-flash",),
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ("glm-5.3-flash",),
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "models": ("gpt-4o-mini",),
    },
}
# There is deliberately no "custom" entry with an empty URL: the base URL field is
# free text and a preset only pre-fills it, so anything unlisted is typed directly.
PROVIDER_CUSTOM = "custom"


def provider_for_url(base_url: str) -> str:
    """The provider a base URL belongs to, or `custom` when none matches.

    Reads the URL rather than the stored provider name because the URL is what
    actually decides which endpoint is spoken to. An entry created before the
    provider field existed has no stored name, and a user may pick a preset and
    then edit the URL; both would make a name-based check answer the wrong
    question.
    """
    text = base_url.strip().rstrip("/")
    if not text:
        return CONF_LLM_PROVIDER_DEFAULT
    for name, preset in PROVIDER_PRESETS.items():
        # Trailing slash ignored: both forms are reasonable things to have typed.
        if preset["base_url"].rstrip("/") == text:
            return name
    return PROVIDER_CUSTOM


# Providers that reject the `thinking` body field outright.
#
# Google's OpenAI-compatible endpoint answers a hard HTTP 400:
#
#   Invalid JSON payload received. Unknown name "thinking": Cannot find field.
#
# That endpoint is the `gemini` preset -- this deployment's own -- and the UI
# offers "disabled" as a Reasoning mode for every provider. Choosing it therefore
# made *every* analysis fail with `provider_http_400`, and a 400 is not retried
# (correctly: the request is malformed, so retrying cannot help). The option was a
# trap rather than a cost control.
#
# The field is omitted for these providers instead of removing the option, so a
# user who picks it still gets a working analysis -- at the provider's default
# reasoning cost -- rather than a lost activity.
#
# Only `thinking` is listed here. `reasoning_effort` is accepted by the same
# endpoint -- see `REASONING_EFFORT_TRANSLATIONS` below for which of its *values*
# are, which is a different question this set does not answer.
THINKING_UNSUPPORTED_PROVIDERS = frozenset({"gemini"})

# `reasoning_effort` values a provider rejects VERBATIM, and what to send instead.
#
# The paragraph above says the *key* is accepted by Google's endpoint. That is
# true and it is also not the whole question: the key is accepted, some of the
# values are not. The original probe tested acceptance of the key and stopped
# there, so `max` sat in the UI's option list while never having been sent by any
# exercised path.
#
# Probed against this deployment's own key (2026-10-06), one text-only request per
# value, on `gemini-3.5-flash` (the model this deployment actually configures) and
# `gemini-3.8-flash`:
#
#   low, medium, none   -> HTTP 200 on both
#   high                -> HTTP 200 on 3.5 (503, i.e. load, on 3.8)
#   max, auto, HIGH     -> HTTP 400 both:
#       Invalid reasoning_effort: max. Valid values are:
#       high, low, medium, minimal, none
#
# A 400 is not retried -- the request is malformed, so retrying cannot help --
# and `is_failover_eligible` treats any `provider_http_*` as worth another
# provider. So picking `max` cost the activity outright when no fallback was
# configured, which is the same trap `thinking` was, one field over.
#
# The UI no longer offers `max`. This table still exists because a value already
# stored in an entry outlives the option list: a deployment that never reopens the
# form would otherwise keep 400ing on every activity. `max` is translated to
# `high`, the endpoint's own maximum, rather than dropped -- the same choice
# `THINKING_UNSUPPORTED_PROVIDERS` makes, one step narrower: honour the intent at
# the provider's ceiling instead of discarding the control entirely.
REASONING_EFFORT_TRANSLATIONS: dict[str, dict[str, str]] = {
    "gemini": {"max": "high"},
}


def reasoning_effort_for(base_url: str, value: str) -> str:
    """The `reasoning_effort` to send this provider, or "" to send no field.

    A value the provider is known to reject is translated to one it accepts; an
    empty result means "leave the field out and let the provider's own default
    apply". The decision lives here, next to `provider_for_url`, because both the
    analysis request and the connectivity probe must make it identically -- a
    probe that sent a rejected value would report HTTP 400 for a configuration
    that works perfectly.
    """
    text = value.strip() if isinstance(value, str) else ""
    if not text or text == "default":
        return ""
    return REASONING_EFFORT_TRANSLATIONS.get(provider_for_url(base_url), {}).get(
        text, text
    )


# The preset whose endpoint was measured to reject the field. Kept next to the set
# above so the two cannot drift: a rename in `PROVIDER_PRESETS` would otherwise
# silently un-gate the provider, and the only symptom would be activities that
# stop being analysed.
GEMINI_PRESET = "gemini"

# The deployment's own description of what the camera looks at: which door in
# frame is a lift, where the front door is, which way a corridor leads. Optional
# and empty by default. The general prompt's rules are camera-agnostic, but its
# arrival/departure rules speak of "the front door", and a model that cannot
# tell which door that is will read the nearest one as it -- measured on this
# deployment, that turned "stepped out of the lift and walked away" into
# `home_departure`.
CONF_SCENE_DESCRIPTION = "scene_description"

# Bounded so a pasted essay cannot crowd out the frames' own description; a
# couple of sentences is what the prompt is written to expect.
MAX_SCENE_DESCRIPTION_LENGTH = 500

# 每个 entry 可自定义的标签集与提示词。上限的目的是拦截误粘的超长文本，
# 不是设计约束——现默认 11 个标签、提示词约 1400 字。
CONF_SCENE_LABELS = "scene_labels"
CONF_PROMPT_OVERRIDE = "prompt_override"
MAX_SCENE_LABELS = 30
MAX_LABEL_DEFINITION_LENGTH = 200
MAX_PROMPT_OVERRIDE_LENGTH = 4000


# 配置界面里两个可自定义字段的预填内容：出厂默认就是电梯厅场景。预填是为了让
# 用户看到默认值并在此基础上修改，而不是从零写。
#
# 这两个常量只做表单默认值。存储里没有值时仍然回退到内置场景
# （`vision_config_from` 用 `pick(..., "")`），所以既有 entry 的提示词逐字节
# 不变——预填不会把内置场景复制进它们的配置里。
#
# 文本本身**不在这里**：它从 `prompts/review_six/` 的数据文件读出来，和内置场景
# 读的是同一份。此前是两份手写副本，靠 `tests/test_config_flow.py` 里一条
# `assert DEFAULT_PROMPT_OVERRIDE == SCENES["review_six"].template` 维系同步——
# 那条断言能发现不一致，却说不清是谁改了哪一份。同源之后这个失败模式消失。
#
# 标签必须列**全场景的并集**：scene_labels 非空时会**全局替换** allowed 集合，
# 不区分场景。漏掉某个标签它就永远无法报出，且没有任何报错——预填是默认值，
# 用户不会意识到自己删掉了一个标签。并集 = 11 个（只剩 review_six）：8 个来自
# `labels.txt`，3 个（cleaning / home_arrival / home_departure）的定义写在
# `rules.txt` 正文里，术语表刻意不重复它们。
def _prefilled_scene_labels() -> str:
    """The form's prefilled label set, built from the prompt data files.

    Imported lazily: `scenes.py` imports from `const.py`, so a module-level
    import here would be circular. The call happens once, when the options
    schema is built, so the indirection costs nothing measurable.
    """
    from .scenes import BUILTIN_SCENE_LABELS

    return "".join(
        f"{name}: {definition}\n" for name, definition in BUILTIN_SCENE_LABELS
    )


def _prefilled_prompt_override() -> str:
    """The form's prefilled judgement rules, read from the prompt data file."""
    from .scenes import load_prompt_text

    return load_prompt_text("review_six", "rules.txt")


DEFAULT_SCENE_LABELS = _prefilled_scene_labels()

DEFAULT_PROMPT_OVERRIDE = _prefilled_prompt_override()
