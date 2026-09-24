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
