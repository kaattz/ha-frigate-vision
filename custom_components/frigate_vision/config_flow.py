"""Config flow for Frigate Vision."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import voluptuous as vol
from aiohttp import (
    ClientError,
    ClientSession,
    ClientTimeout,
    CookieJar,
    InvalidURL,
)
from homeassistant import config_entries
from homeassistant.config_entries import ConfigFlowResult
from homeassistant.core import HomeAssistant
from homeassistant.helpers import selector
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from yarl import URL

from .const import (
    AUTH_NATIVE,
    AUTH_NONE,
    CONF_AUTH_MODE,
    CONF_BASE_URL,
    CONF_CAMERA,
    CONF_FAR_ZONES,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_BASE_URL_DEFAULT,
    CONF_LLM_MODEL,
    CONF_LLM_REASONING_EFFORT,
    CONF_LLM_THINKING,
    CONF_MQTT_TOPIC_PREFIX,
    CONF_NAME,
    CONF_NEAR_ZONES,
    CONF_SCENE_DESCRIPTION,
    CONF_TRANSITION_ZONES,
    DOMAIN,
    PROCESSING_MODES,
)
from .vision import (
    REASONING_EFFORTS,
    THINKING_MODES,
    VisionError,
    async_test_connection,
    vision_config_from,
)


def _text() -> selector.TextSelector:
    return selector.TextSelector(selector.TextSelectorConfig())


def _split_values(value: str, *, allow_empty: bool = False) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if (not values and not allow_empty) or len(values) != len(set(values)):
        raise vol.Invalid("invalid_list")
    return values


def _parse_base_url(value: str) -> URL:
    try:
        parsed = URL(value)
        origin = parsed.origin()
    except ValueError as exc:
        raise InvalidURL(value) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.host
        or parsed.user is not None
        or parsed.password is not None
    ):
        raise InvalidURL(value)
    if not str(origin):
        raise InvalidURL(value)
    return parsed


def _options_schema() -> vol.Schema:
    return vol.Schema(
        {
            vol.Required("processing_mode", default="observe"): selector.SelectSelector(
                selector.SelectSelectorConfig(options=list(PROCESSING_MODES))
            ),
            # Provider settings live in options rather than data so they can be
            # changed from the UI at any time; changing the credential should
            # not require removing and re-adding the integration.
            vol.Optional(CONF_LLM_BASE_URL, default=CONF_LLM_BASE_URL_DEFAULT): _text(),
            vol.Optional(CONF_LLM_API_KEY): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
            ),
            vol.Optional(CONF_LLM_MODEL): _text(),
            # Reasoning is the cost control that does not break reasoning.
            # Measured: disabling thinking entirely returned unrelated output
            # when a task needed multiple steps, while `low` stayed correct at
            # roughly 60% of the default's reasoning tokens.
            vol.Optional(
                CONF_LLM_REASONING_EFFORT, default="default"
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(options=list(REASONING_EFFORTS))
            ),
            vol.Optional(CONF_LLM_THINKING, default="default"): selector.SelectSelector(
                selector.SelectSelectorConfig(options=list(THINKING_MODES))
            ),
            vol.Required("target_width", default=768): vol.All(
                int, vol.Range(512, 1920)
            ),
            vol.Required("max_tokens", default=4000): vol.All(int, vol.Range(1, 20000)),
            vol.Required("output_language", default="zh-CN"): _text(),
            # Optional. Empty keeps the prompt exactly as it was, so a
            # deployment that does not need this is unaffected.
            vol.Optional(CONF_SCENE_DESCRIPTION, default=""): selector.TextSelector(
                selector.TextSelectorConfig(multiline=True)
            ),
            vol.Required("history_retention_days", default=30): vol.All(
                int, vol.Range(1, 365)
            ),
            vol.Required("media_retention_days", default=7): vol.All(
                int, vol.Range(1, 90)
            ),
            vol.Required("queue_size", default=10): vol.All(int, vol.Range(1, 100)),
            # Reviews shorter than this are dropped before any work is done, so
            # a person merely crossing the frame neither uses tokens nor reaches
            # the user. Measured over 233 stored reviews, 9% run under 10s.
            vol.Required("min_review_seconds", default=10): vol.All(
                int, vol.Range(0, 120)
            ),
            vol.Required("analyze_night_unknown", default=True): bool,
            vol.Required("analyze_all_far_reviews", default=True): bool,
        }
    )


def _door_schema() -> vol.Schema:
    """Optional lock mapping; an empty lock entity means review-only."""
    return vol.Schema(
        {
            vol.Optional("event_entity_id"): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="event")
            ),
            vol.Optional("action_attribute"): _text(),
            vol.Optional("open_values"): _text(),
            vol.Optional("close_values"): _text(),
            vol.Optional("side_attribute"): _text(),
            vol.Optional("inside_values"): _text(),
            vol.Optional("outside_values"): _text(),
            vol.Optional("contact_entity_id"): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="binary_sensor")
            ),
            vol.Optional("doorbell_event_entity_id"): selector.EntitySelector(
                selector.EntitySelectorConfig(domain="event")
            ),
        }
    )


def _parse_door_mapping(
    user_input: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Return the stored door mapping, or the error key that rejected it.

    An absent lock entity is the review-only case: it yields an empty mapping
    so the runtime skips the whole door-cycle branch (REQ-2). Contact and
    doorbell context cannot stand alone, so they are rejected without a lock.
    """
    event_entity_id = user_input.get("event_entity_id")
    contact_entity_id = user_input.get("contact_entity_id")
    doorbell_event_entity_id = user_input.get("doorbell_event_entity_id")
    if not event_entity_id:
        if contact_entity_id or doorbell_event_entity_id:
            return None, "door_lock_required"
        return {}, None
    try:
        open_values = _split_values(user_input.get("open_values") or "")
        close_values = _split_values(user_input.get("close_values") or "")
        inside_values = _split_values(user_input.get("inside_values") or "")
        outside_values = _split_values(user_input.get("outside_values") or "")
        action_attribute = str(user_input.get("action_attribute") or "").strip()
        side_attribute = str(user_input.get("side_attribute") or "").strip()
        if not action_attribute or not side_attribute:
            raise vol.Invalid("invalid_door_mapping")
        if set(open_values) & set(close_values) or set(inside_values) & set(
            outside_values
        ):
            raise vol.Invalid("invalid_door_mapping")
    except vol.Invalid:
        return None, "invalid_door_mapping"
    return {
        "event_entity_id": event_entity_id,
        "action_attribute": action_attribute,
        "open_values": open_values,
        "close_values": close_values,
        "side_attribute": side_attribute,
        "inside_values": inside_values,
        "outside_values": outside_values,
        "contact_entity_id": contact_entity_id,
        "doorbell_event_entity_id": doorbell_event_entity_id,
    }, None


def _door_suggestions(door: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a stored mapping or a submitted form payload into form values.

    Called with two shapes: the persisted mapping (lists) and, when a
    submission was rejected, the payload the user just sent (comma strings).
    Echoing the submitted payload back is what lets a user clear the lock and
    see it stay cleared, instead of having the stored mapping restored under
    them. Value lists are comma-joined because the text selector holds a single
    string; passing the raw lists would prefill `['1']` verbatim.
    """

    def as_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return ",".join(str(item) for item in value)

    if not door:
        return {}
    return {
        "event_entity_id": door.get("event_entity_id"),
        "action_attribute": door.get("action_attribute"),
        "open_values": as_text(door.get("open_values")),
        "close_values": as_text(door.get("close_values")),
        "side_attribute": door.get("side_attribute"),
        "inside_values": as_text(door.get("inside_values")),
        "outside_values": as_text(door.get("outside_values")),
        "contact_entity_id": door.get("contact_entity_id"),
        "doorbell_event_entity_id": door.get("doorbell_event_entity_id"),
    }


async def async_validate_frigate(
    hass: HomeAssistant,
    base_url: str,
    auth_mode: str = AUTH_NONE,
    username: str | None = None,
    password: str | None = None,
) -> str:
    """Validate the public Frigate version endpoint."""
    if auth_mode == AUTH_NATIVE:
        async with ClientSession(cookie_jar=CookieJar(unsafe=True)) as session:
            async with session.post(
                f"{base_url.rstrip('/')}/api/login",
                json={"user": username, "password": password},
                timeout=ClientTimeout(total=10),
            ) as response:
                if response.status != 200:
                    raise ConnectionError
            return await _async_frigate_version(session, base_url)
    return await _async_frigate_version(async_get_clientsession(hass), base_url)


async def _async_frigate_version(session: ClientSession, base_url: str) -> str:
    async with session.get(
        f"{base_url.rstrip('/')}/api/version", timeout=ClientTimeout(total=10)
    ) as response:
        if response.status != 200:
            raise ConnectionError
        version = (await response.text()).strip()
        if not version:
            raise ConnectionError
        return version


class FrigateEntryIntelligenceConfigFlow(config_entries.ConfigFlow, domain=DOMAIN):
    """Configure a single entrance."""

    VERSION = 1
    MINOR_VERSION = 1

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}
        self._reconfigure_data: dict[str, Any] = {}
        self._reconfigure_unique_id: str | None = None

    @staticmethod
    def async_get_options_flow(
        config_entry: config_entries.ConfigEntry,
    ) -> config_entries.OptionsFlow:
        return FrigateEntryIntelligenceOptionsFlow()

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            try:
                parsed_url = _parse_base_url(user_input[CONF_BASE_URL])
                auth_mode = user_input[CONF_AUTH_MODE]
                username = user_input.get("username")
                password = user_input.get("password")
                if auth_mode == AUTH_NATIVE and (not username or not password):
                    raise vol.Invalid("invalid_auth")
                if auth_mode == AUTH_NONE:
                    username = password = None
                await async_validate_frigate(
                    self.hass,
                    user_input[CONF_BASE_URL],
                    auth_mode,
                    username,
                    password,
                )
                zones = {
                    "near": _split_values(
                        user_input[CONF_NEAR_ZONES], allow_empty=True
                    ),
                    "transition": _split_values(
                        user_input[CONF_TRANSITION_ZONES], allow_empty=True
                    ),
                    "far": _split_values(user_input[CONF_FAR_ZONES], allow_empty=True),
                }
                if (
                    set(zones["near"]) & set(zones["transition"])
                    or set(zones["near"]) & set(zones["far"])
                    or set(zones["transition"]) & set(zones["far"])
                ):
                    raise vol.Invalid("invalid_zones")
            except InvalidURL:
                errors["base"] = "invalid_url"
            except ClientError, ConnectionError, TimeoutError:
                errors["base"] = "cannot_connect"
            except vol.Invalid as exc:
                errors["base"] = str(exc)
            else:
                unique_id = (
                    f"{str(parsed_url.origin()).lower()}|"
                    f"{user_input[CONF_CAMERA].lower()}"
                )
                await self.async_set_unique_id(unique_id)
                self._abort_if_unique_id_configured()
                self._data = {
                    CONF_NAME: user_input[CONF_NAME],
                    "frigate": {
                        CONF_BASE_URL: user_input[CONF_BASE_URL].rstrip("/"),
                        CONF_AUTH_MODE: user_input[CONF_AUTH_MODE],
                        CONF_MQTT_TOPIC_PREFIX: user_input[CONF_MQTT_TOPIC_PREFIX],
                        CONF_CAMERA: user_input[CONF_CAMERA],
                        "username": username,
                        "password": password,
                    },
                    "zones": zones,
                }
                return await self.async_step_door()

        schema = vol.Schema(
            {
                vol.Required(CONF_NAME): _text(),
                vol.Required(CONF_BASE_URL): _text(),
                vol.Required(
                    CONF_AUTH_MODE, default=AUTH_NONE
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[AUTH_NONE, AUTH_NATIVE])
                ),
                vol.Required(CONF_MQTT_TOPIC_PREFIX, default="frigate"): _text(),
                vol.Required(CONF_CAMERA): _text(),
                vol.Optional("username"): _text(),
                vol.Optional("password"): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
                vol.Required(CONF_NEAR_ZONES): _text(),
                vol.Required(CONF_TRANSITION_ZONES): _text(),
                vol.Required(CONF_FAR_ZONES): _text(),
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async def async_step_door(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            door, error = _parse_door_mapping(user_input)
            if error is not None:
                errors["base"] = error
            else:
                self._data["door"] = door
                return await self.async_step_llmvision()
        return self.async_show_form(
            step_id="door",
            data_schema=self.add_suggested_values_to_schema(
                _door_schema(), _door_suggestions(user_input or {})
            ),
            errors=errors,
        )

    async def async_step_llmvision(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        if user_input is not None:
            base_url = str(user_input.get(CONF_LLM_BASE_URL, "")).strip()
            model = str(user_input.get(CONF_LLM_MODEL, "")).strip()
            api_key = str(user_input.get(CONF_LLM_API_KEY, "")).strip()
            if not base_url or not model:
                errors["base"] = "llm_incomplete"
            elif not api_key:
                errors["base"] = "llm_missing_key"
            else:
                self._data["llm"] = {
                    CONF_LLM_BASE_URL: base_url,
                    CONF_LLM_API_KEY: api_key,
                    CONF_LLM_MODEL: model,
                    CONF_LLM_THINKING: str(
                        user_input.get(CONF_LLM_THINKING, "default")
                    ),
                    CONF_LLM_REASONING_EFFORT: str(
                        user_input.get(CONF_LLM_REASONING_EFFORT, "default")
                    ),
                }
                return await self.async_step_options()
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_LLM_BASE_URL, default=CONF_LLM_BASE_URL_DEFAULT
                ): _text(),
                vol.Required(CONF_LLM_API_KEY): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
                vol.Required(CONF_LLM_MODEL): _text(),
                vol.Required(
                    CONF_LLM_REASONING_EFFORT, default="default"
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=list(REASONING_EFFORTS))
                ),
                vol.Required(
                    CONF_LLM_THINKING, default="default"
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=list(THINKING_MODES))
                ),
            }
        )
        return self.async_show_form(
            step_id="llmvision", data_schema=schema, errors=errors
        )

    async def async_step_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(
                title=self._data[CONF_NAME], data=self._data, options=user_input
            )
        return self.async_show_form(step_id="options", data_schema=_options_schema())

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        current = entry.data["frigate"]
        zones_current = entry.data["zones"]
        if user_input is not None:
            try:
                parsed_url = _parse_base_url(user_input[CONF_BASE_URL])
                username = user_input.get("username")
                password = user_input.get("password")
                if user_input[CONF_AUTH_MODE] == AUTH_NATIVE and (
                    not username or not password
                ):
                    raise vol.Invalid("invalid_auth")
                if user_input[CONF_AUTH_MODE] == AUTH_NONE:
                    username = password = None
                zones = {
                    "near": _split_values(
                        user_input[CONF_NEAR_ZONES], allow_empty=True
                    ),
                    "transition": _split_values(
                        user_input[CONF_TRANSITION_ZONES], allow_empty=True
                    ),
                    "far": _split_values(user_input[CONF_FAR_ZONES], allow_empty=True),
                }
                if (
                    set(zones["near"]) & set(zones["transition"])
                    or set(zones["near"]) & set(zones["far"])
                    or set(zones["transition"]) & set(zones["far"])
                ):
                    raise vol.Invalid("invalid_zones")
                await async_validate_frigate(
                    self.hass,
                    user_input[CONF_BASE_URL],
                    user_input[CONF_AUTH_MODE],
                    username,
                    password,
                )
            except InvalidURL:
                errors["base"] = "invalid_url"
            except ClientError, ConnectionError, TimeoutError:
                errors["base"] = "cannot_connect"
            except vol.Invalid as exc:
                errors["base"] = str(exc)
            else:
                unique_id = (
                    f"{str(parsed_url.origin()).lower()}|"
                    f"{user_input[CONF_CAMERA].lower()}"
                )
                if any(
                    other.entry_id != entry.entry_id and other.unique_id == unique_id
                    for other in self._async_current_entries()
                ):
                    return self.async_abort(reason="already_configured")
                data = dict(entry.data)
                data["frigate"] = {
                    CONF_BASE_URL: user_input[CONF_BASE_URL].rstrip("/"),
                    CONF_AUTH_MODE: user_input[CONF_AUTH_MODE],
                    CONF_MQTT_TOPIC_PREFIX: user_input[CONF_MQTT_TOPIC_PREFIX],
                    CONF_CAMERA: user_input[CONF_CAMERA],
                    "username": username,
                    "password": password,
                }
                data["zones"] = zones
                self._reconfigure_data = data
                self._reconfigure_unique_id = unique_id
                return await self.async_step_reconfigure_door()
        schema = vol.Schema(
            {
                vol.Required(CONF_BASE_URL, default=current[CONF_BASE_URL]): _text(),
                vol.Required(
                    CONF_AUTH_MODE, default=current[CONF_AUTH_MODE]
                ): selector.SelectSelector(
                    selector.SelectSelectorConfig(options=[AUTH_NONE, AUTH_NATIVE])
                ),
                vol.Required(
                    CONF_MQTT_TOPIC_PREFIX, default=current[CONF_MQTT_TOPIC_PREFIX]
                ): _text(),
                vol.Required(CONF_CAMERA, default=current[CONF_CAMERA]): _text(),
                vol.Optional(
                    "username", default=current.get("username") or ""
                ): _text(),
                vol.Optional(
                    "password", default=current.get("password") or ""
                ): selector.TextSelector(
                    selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
                ),
                vol.Required(
                    CONF_NEAR_ZONES, default=",".join(zones_current["near"])
                ): _text(),
                vol.Required(
                    CONF_TRANSITION_ZONES,
                    default=",".join(zones_current["transition"]),
                ): _text(),
                vol.Required(
                    CONF_FAR_ZONES, default=",".join(zones_current["far"])
                ): _text(),
            }
        )
        return self.async_show_form(
            step_id="reconfigure", data_schema=schema, errors=errors
        )

    async def async_step_reconfigure_door(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Edit the optional lock mapping, or clear it to run review-only.

        Reconfigure exists so an existing entry can move between door-cycle and
        review-only operation without deleting the entry and losing its stored
        activity history. Leaving the lock entity empty is the review-only case.
        """
        entry = self._get_reconfigure_entry()
        errors: dict[str, str] = {}
        if user_input is not None:
            door, error = _parse_door_mapping(user_input)
            if error is not None:
                errors["base"] = error
            else:
                data = dict(self._reconfigure_data)
                data["door"] = door
                return self.async_update_reload_and_abort(
                    entry,
                    unique_id=self._reconfigure_unique_id,
                    data=data,
                )
        schema = self.add_suggested_values_to_schema(
            _door_schema(),
            _door_suggestions(
                user_input if user_input is not None else (entry.data.get("door") or {})
            ),
        )
        return self.async_show_form(
            step_id="reconfigure_door", data_schema=schema, errors=errors
        )


class FrigateEntryIntelligenceOptionsFlow(config_entries.OptionsFlowWithReload):
    """Update behavior options and reload the entry."""

    def __init__(self) -> None:
        self._pending: dict[str, Any] = {}

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        # Labels are given as a dict rather than a list. With the list form the
        # rows rendered without text in this deployment even though the step
        # title resolved, leaving two unlabelled entries; the dict form carries
        # its own labels and needs no translation lookup.
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "settings": "配置参数 / Configure settings",
                "test_connection": "测试视觉模型连通性 / Test provider connection",
            },
        )

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            return self.async_create_entry(data=user_input)
        schema = self.add_suggested_values_to_schema(
            _options_schema(), self.config_entry.options
        )
        return self.async_show_form(step_id="settings", data_schema=schema)

    async def async_step_test_connection(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Probe the provider with a text-only call and report the result.

        Runs against the saved options rather than unsaved form input, so the
        result always describes the configuration the entry will actually use.
        """
        errors: dict[str, str] = {}
        placeholders: dict[str, str] = {}
        options = self.config_entry.options
        config = vision_config_from(self.config_entry.data, options)
        try:
            session = async_get_clientsession(self.hass)
            report = await async_test_connection(session, config)
        except VisionError as exc:
            errors["base"] = str(exc)
        except Exception:  # noqa: BLE001
            errors["base"] = "provider_unavailable"
        else:
            placeholders = {
                "model": report.model,
                "seconds": f"{report.seconds:.1f}",
                "tokens": str(report.total_tokens if report.total_tokens else "-"),
            }
        return self.async_show_form(
            step_id="test_connection",
            data_schema=vol.Schema({}),
            errors=errors,
            description_placeholders=placeholders,
        )
