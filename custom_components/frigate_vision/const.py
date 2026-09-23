"""Constants for Frigate Vision."""

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
CONF_LLM_BASE_URL_DEFAULT = "https://api.deepseek.com/v1"

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
