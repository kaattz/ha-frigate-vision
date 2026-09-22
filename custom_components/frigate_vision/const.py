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
