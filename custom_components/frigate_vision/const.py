"""Constants for Frigate Vision."""

from __future__ import annotations

from typing import Any

DOMAIN = "frigate_vision"

AUTH_NONE = "none"
AUTH_NATIVE = "native"
PROCESSING_MODES = ("observe", "shadow", "live")

CONF_NAME = "name"
CONF_BASE_URL = "base_url"
CONF_AUTH_MODE = "auth_mode"
CONF_MQTT_TOPIC_PREFIX = "mqtt_topic_prefix"
CONF_CAMERA = "camera"
CONF_NEAR_ZONES = "near_zones"
CONF_TRANSITION_ZONES = "transition_zones"
CONF_FAR_ZONES = "far_zones"

# Vision provider settings. Held by this integration rather than borrowed from
# another component, so the request body is fully under its control -- notably
# the reasoning toggle, which dominates cost.
CONF_LLM_BASE_URL = "llm_base_url"
CONF_LLM_API_KEY = "llm_api_key"
CONF_LLM_MODEL = "llm_model"
CONF_LLM_THINKING = "llm_thinking"
CONF_LLM_REASONING_EFFORT = "llm_reasoning_effort"
CONF_LLM_PROVIDER = "llm_provider"
CONF_LLM_BASE_URL_DEFAULT = "https://api.deepseek.com/v1"
CONF_LLM_PROVIDER_DEFAULT = "deepseek"

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
# 标签必须列**全场景的并集**，不能只列电梯厅那 11 个：scene_labels 非空时会
# **全局替换** allowed 集合，不区分场景。door_roundtrip 的 short_roundtrip 只属于
# 它自己，漏掉的话门锁场景的「短暂外出」永远无法报出，且没有任何报错——预填是
# 默认值，用户不会意识到自己删掉了一个标签。
#
# 并集 = 12 个。其中 8 个的定义来自 `CLASSIFICATION_GLOSSARY`，
# cleaning / home_arrival / home_departure 的定义写在 `review_six` 的 template
# 正文里（术语表刻意不重复它们），short_roundtrip 的定义同样取自术语表。
# 顺序按标签名字母序，与 `_glossary` 里的 `sorted()` 一致。
DEFAULT_SCENE_LABELS = (
    "cleaning: 指连续清楚出现清扫、扫地、拖地或擦拭动作；\n"
    "elevator_activity: 指人物在电梯间内活动但没有跨越入户门；\n"
    "food_delivery: 指放下餐饮外卖且离开后餐食仍留在原处；\n"
    "home_arrival: 指人物从门外走近入户门并进入门内；\n"
    "home_departure: 指人物从入户门内走出并远离该门；\n"
    "maintenance: 指维修人员对楼道设施进行作业；\n"
    "package_delivery: 指放下快递包裹且离开后包裹仍留在原处；\n"
    "short_roundtrip: 指短暂外出后随即返回。\n"
    "suspicious_activity: 指试探门锁、反复徘徊或窥探等可疑行为；\n"
    "unable_to_confirm: 指证据不足或画面质量导致无法判断；\n"
    "unknown_activity: 指画面确实无法支持任何其他判断；\n"
    "visitor: 指访客到访，例如敲门、在门外等候或被迎入；\n"
)

# `review_six` 场景的 template 原文，逐字节相同。由该场景的 template 生成，
# 不是手抄——预填与内置规则必须是同一套规则，否则就成了两个会各自漂移的版本。
DEFAULT_PROMPT_OVERRIDE = (
    "图按时间顺序排列，从左到右、再从上到下读取。第一格是人物首帧，最后一格是人物离开后的后置现场，倒数第二格是人物末帧，中间各格是这段时间内的变化候选。"
    "只有连续清楚出现清扫、扫地、拖地或擦拭动作才可判断cleaning。只有清楚看到放下包裹或外卖且离开后物品仍留下，才可判断配送。"
    "不能根据制服、携带物或停留时间猜测职业。"
    "关于回家与离家：只有画面中能看出人物相对入户门的移动方向时才可判断——人物从门内走出并远离该门，判为home_departure；"
    "人物从门外走近该门并进入门内，判为home_arrival。"
    "无法看出门的开合或人物的进出方向时，不得据停留时长、出现顺序或是否携带物品推断方向，应选择unknown_activity或unable_to_confirm。"
    "若人物只是经过、在门前停留或整理物品而未跨越门，也不要判断回家或离家。描述人物时须写明可辨认的外貌（性别、发型、上下身衣着的颜色与款式）；"
    "看不清就写明看不清，不得补足。画面中没有清晰正面面部时，不得推断年龄，改用中性说法（一名男子、一名女士或一名人员）。"
    "描述须交代人物从何处来、往何处去，以及门与电梯的状态变化，动作要连贯。"
)
