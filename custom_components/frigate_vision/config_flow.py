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
from homeassistant.helpers.selector import SelectOptionDict
from yarl import URL

from .const import (
    AUTH_NATIVE,
    AUTH_NONE,
    CONF_AUTH_MODE,
    CONF_BASE_URL,
    CONF_CAMERA,
    CONF_FACE_SERVICE_URL,
    CONF_FACE_SERVICE_URL_DEFAULT,
    CONF_FALLBACK_LLM_API_KEY,
    CONF_FALLBACK_LLM_BASE_URL,
    CONF_FALLBACK_LLM_MODEL,
    CONF_FALLBACK_LLM_REASONING_EFFORT,
    CONF_FALLBACK_LLM_THINKING,
    CONF_FAR_ZONES,
    CONF_LLM_API_KEY,
    CONF_LLM_BASE_URL,
    CONF_LLM_BASE_URL_DEFAULT,
    CONF_LLM_MODEL,
    CONF_LLM_PROVIDER,
    CONF_LLM_PROVIDER_DEFAULT,
    CONF_LLM_REASONING_EFFORT,
    CONF_LLM_THINKING,
    CONF_MQTT_TOPIC_PREFIX,
    CONF_NAME,
    CONF_NEAR_ZONES,
    CONF_PERSON_HIGHLIGHT,
    CONF_PERSON_HIGHLIGHT_DEFAULT,
    CONF_PROMPT_OVERRIDE,
    CONF_SCENE_DESCRIPTION,
    CONF_SCENE_LABELS,
    CONF_TRANSITION_ZONES,
    DEFAULT_PROMPT_OVERRIDE,
    DEFAULT_SCENE_LABELS,
    DOMAIN,
    PROVIDER_PRESETS,
    provider_for_url,
)
from .scenes import parse_scene_labels
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
            # Provider settings live in options rather than data so they can be
            # changed from the UI at any time; changing the credential should
            # not require removing and re-adding the integration.
            #
            # `llm_provider` is deliberately not a field here. It used to be, and
            # it was the source of the "I switched provider and nothing happened"
            # report: the form showed a value derived from the URL while the
            # submit-time guard compared against the stored one, so the two could
            # disagree and the switch silently did nothing. Choosing a provider is
            # now a menu tap that prefills this form, so there is nothing left for
            # a dropdown here to do.
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
            # The second provider: tried when the first one cannot answer, so the
            # activity is not lost. Plain fields rather than a provider menu --
            # the primary's menu exists because its URL is the part that cannot be
            # guessed, but the fallback is usually "another known endpoint", and a
            # menu of its own would double that machinery for little gain.
            #
            # All three of URL/key/model are required *together*, and the form
            # refuses a partial one (`_fallback_errors`) instead of storing it: a
            # half-configured backup is tried on every failure and fails for a
            # configuration reason, which hides the real failure behind a second
            # one. The URL's default is empty rather than the primary's, because
            # that empty value is what "the fallback is off" looks like -- it is
            # the default state.
            vol.Optional(CONF_FALLBACK_LLM_BASE_URL, default=""): _text(),
            vol.Optional(CONF_FALLBACK_LLM_API_KEY): selector.TextSelector(
                selector.TextSelectorConfig(type=selector.TextSelectorType.PASSWORD)
            ),
            vol.Optional(CONF_FALLBACK_LLM_MODEL): _text(),
            # Both dropdowns default to `default`, which is also what an
            # untouched form submits. They carry no evidence that the user meant
            # to configure a second provider, so the validator ignores them when
            # deciding whether the three above were filled in as a group.
            vol.Optional(
                CONF_FALLBACK_LLM_REASONING_EFFORT, default="default"
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(options=list(REASONING_EFFORTS))
            ),
            vol.Optional(
                CONF_FALLBACK_LLM_THINKING, default="default"
            ): selector.SelectSelector(
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
            # 该摄像头自己的标签集，每行一条「标签: 定义」。预填出厂默认的电梯厅
            # 那套，用户在此基础上改；留空表示用内置场景的标签。
            vol.Optional(
                CONF_SCENE_LABELS, default=DEFAULT_SCENE_LABELS
            ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
            # 该摄像头自己的判断规则。预填内置规则原文，用户在此基础上改；留空表示
            # 用内置规则。契约与现场布局仍会自动附加，因为解析器和规则都依赖它们。
            vol.Optional(
                CONF_PROMPT_OVERRIDE, default=DEFAULT_PROMPT_OVERRIDE
            ): selector.TextSelector(selector.TextSelectorConfig(multiline=True)),
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


def _setup_schema() -> vol.Schema:
    """Everything the initial flow's last step asks for.

    The initial flow is a single form built once, so it carries the close-up fields
    too; only the options flow splits them onto their own submenu. Built from the
    two halves rather than as one big literal, which is what stops them from
    drifting -- each half is defined once and both callers agree by construction.

    `vol.Schema.extend` cannot be used here: it asserts both operands are plain
    dicts, and both halves are Schemas. Merging the marker dicts is equivalent and
    is what the schema object is built from anyway.
    """
    combined: dict[Any, Any] = {}
    for schema in (_options_schema(), _close_up_schema()):
        for marker in schema.schema:
            combined[marker] = schema.schema[marker]
    return vol.Schema(combined)


def _close_up_schema() -> vol.Schema:
    """The person close-up's own settings.

    Split out of `_options_schema()` and given its own submenu row, because as the
    final two rows of that 20-field form they sat below the fold. The owner went
    looking for the face-service field, did not find it, and reported it as missing
    rather than as off-screen -- which is the cost of a field nobody scrolls to.

    A schema of its own rather than a slice of the big one, so the submenu form is
    two rows and the settings form is eighteen. Both stay in step because each is
    defined once.
    """
    return vol.Schema(
        {
            # 证据图右侧是否再附一栏人物放大特写。默认关：打开会改变发给模型的
            # 图并改变缓存键，属于用户要显式选择的行为变更。
            vol.Required(
                CONF_PERSON_HIGHLIGHT, default=CONF_PERSON_HIGHLIGHT_DEFAULT
            ): bool,
            # 可选的人脸检测服务，用来挑【哪一帧】做特写。留空即关闭，行为与今天
            # 完全一致：只按「检测框面积最大」选。
            #
            # 之所以是 URL 而不是开关：服务是用户自己另外跑的容器，「开没开」和
            # 「在哪」是同一个问题，一个空串就回答了，不必再加一个要同步的开关。
            #
            # 它是【提升项】不是必须项。服务没配、连不上、超时、或答案不对时，
            # 集成一律退回「面积最大」那条规则。
            vol.Optional(
                CONF_FACE_SERVICE_URL, default=CONF_FACE_SERVICE_URL_DEFAULT
            ): selector.TextSelector(),
        }
    )


# Text carried by each dropdown row. Keyed separately from the preset so a row's
# wording can change without touching the URL it selects.
_PROVIDER_LABELS = {
    "deepseek": "DeepSeek（api.deepseek.com）",
    "gemini": "Google Gemini（OpenAI 兼容接口）",
    "glm": "智谱 GLM（open.bigmodel.cn）",
    "openai": "OpenAI（api.openai.com）",
}


def _provider_options() -> list[SelectOptionDict]:
    """Build the provider dropdown, labelled rather than keyed.

    Labels carry text because an option labelled with the raw key renders as a
    blank row in this deployment (the same reason the options menu passes a dict).
    """
    options: list[SelectOptionDict] = [
        {"value": name, "label": _PROVIDER_LABELS.get(name, name)}
        for name in PROVIDER_PRESETS
    ]
    options.append({"value": "custom", "label": "其他 / Other (type a URL)"})
    return options


def _preset_base_url(provider: str) -> str | None:
    """The URL a chosen provider implies, or None when it implies nothing.

    `custom` is deliberately absent from the presets: the URL field is free text,
    and an unknown choice must leave whatever the user typed alone rather than
    blanking a working endpoint.
    """
    preset = PROVIDER_PRESETS.get(provider)
    return preset["base_url"] if preset else None


def _provider_for_url(base_url: str) -> str:
    """The provider a stored URL belongs to, or `custom` when none matches.

    Delegates to `const.provider_for_url`, which is also what the request builder
    uses to decide whether an endpoint accepts the `thinking` field. Two copies of
    this comparison would be two chances to disagree, and the disagreement would
    show up as requests that silently drop a cost control -- or that 400 on a
    provider that rejects the key.

    The behaviour is unchanged from the local copy this replaced: an entry stored
    before the dropdown existed has no provider value, so the URL is the only
    honest signal. A dialog that names the wrong provider is worse than no
    dropdown, because the next save adopts that claim and rewrites the endpoint.
    """
    return provider_for_url(base_url)


def _llm_schema() -> vol.Schema:
    """The provider step: a preset to pick, and a URL that stays editable.

    The URL is the field most easily got wrong -- it must be the OpenAI-compatible
    root, and providers disagree about the shape, so it is worth choosing from a
    list. It remains free text because a local router or reverse proxy is a normal
    deployment (this one runs that way), and a preset only pre-fills it.
    """
    return vol.Schema(
        {
            vol.Required(
                CONF_LLM_PROVIDER, default=CONF_LLM_PROVIDER_DEFAULT
            ): selector.SelectSelector(
                selector.SelectSelectorConfig(options=_provider_options())
            ),
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


def _label_errors(user_input: Mapping[str, Any]) -> dict[str, str]:
    """Return form errors for the labels field, or an empty dict when valid.

    Both flows must validate: the labels are stored verbatim and a malformed line
    fails every later analysis, where the user only sees "no notification" -- an
    error far from its cause.
    """
    try:
        parse_scene_labels(str(user_input.get(CONF_SCENE_LABELS, "")))
    except ValueError as exc:
        return {CONF_SCENE_LABELS: str(exc)}
    return {}


def _strip_secrets(suggestions: Mapping[str, Any]) -> dict[str, Any]:
    """Return the suggestions with every credential removed.

    `add_suggested_values_to_schema` writes each suggestion into
    `description.suggested_value`, which is serialised into the flow result the
    browser receives. A password selector only masks the *rendering*; the payload
    still carries the plaintext, so prefilling a stored key ships it to the DOM,
    devtools, screenshots and session restore. Measured: both keys appeared
    verbatim in the serialised form.

    Home Assistant's own `ollama` integration draws the same line -- it suggests
    the model but never the api_key. The rule here is the whole set of credential
    fields, so a future provider cannot reintroduce the leak by forgetting one.

    Everything else is still suggested: the user has to be able to see which
    address, model and retention they currently have.
    """
    return {
        key: value
        for key, value in suggestions.items()
        if key not in _SECRET_OPTION_KEYS
    }


# Every option that holds a credential. Listed once so the two flows and the
# provider menu cannot drift apart on which fields are safe to prefill.
_SECRET_OPTION_KEYS = frozenset(
    {
        CONF_LLM_API_KEY,
        CONF_FALLBACK_LLM_API_KEY,
    }
)


def _merge_options(
    submitted: Mapping[str, Any], stored: Mapping[str, Any]
) -> dict[str, Any]:
    """Fold a form submission into the stored options, keeping blank secrets.

    Two rules, both needed together:

    * **Merge, never replace.** `async_create_entry` replaces the whole options
      dict, and no single form owns all of it -- the close-up fields live in their
      own submenu schema. Writing a form back verbatim therefore deleted every
      field that form does not show, silently.

    * **A blank credential means "leave it alone".** Keys are no longer prefilled
      (see `_strip_secrets`), so the browser submits the password box empty
      whenever the user does not retype it, and an empty string written back would
      delete the credential. `if api_key:` in Home Assistant's `ollama` flow is the
      same rule.

    The result is what validation runs against, so an untouched fallback is judged
    on the credentials that will actually be stored rather than on the empty box
    the browser sent.
    """
    merged = dict(stored)
    merged.update(submitted)
    for key in _SECRET_OPTION_KEYS:
        if not str(submitted.get(key) or "").strip():
            if key in stored:
                merged[key] = stored[key]
            else:
                merged.pop(key, None)
    return merged


def _primary_url_errors(user_input: Mapping[str, Any]) -> dict[str, str]:
    """Return form errors for the primary provider's address, or {} when valid.

    The fallback's address has always been checked here; the primary's was not, on
    the reasoning that adding the check "would start rejecting values that already
    work". That reasoning does not hold for the cases this catches.

    A URL carrying credentials -- the realistic mistake being a key pasted into the
    address -- makes aiohttp raise

        ValueError: Cannot combine AUTHORIZATION header with AUTH argument or
                    credentials encoded in URL

    when the request is built. Measured: that `ValueError` is neither a `ClientError`
    nor a `TimeoutError`, so `_CONNECTION_ERRORS` does not catch it, and it is not a
    `VisionError`, so the failover loop's `except VisionError` does not either. It
    reaches the catch-all as `analysis_outcome_unknown`: not failover-eligible, no
    repair card, and refused by `retry_failed`. One paste therefore disables analysis
    silently and permanently, with the credential sitting in cleartext in the entry's
    options.

    Only the address is checked, and only for the reasons `_parse_base_url` already
    enforces (scheme, host, no userinfo). A working endpoint keeps working: the test
    beside this one pins that.
    """
    url = str(user_input.get(CONF_LLM_BASE_URL) or "").strip()
    if not url:
        # An empty address is the initial flow's normal state, and the schema's own
        # default fills it; "missing" is reported elsewhere (`llm_incomplete`).
        return {}
    try:
        _parse_base_url(url)
    except InvalidURL:
        return {"base": "invalid_url"}
    return {}


def _fallback_errors(user_input: Mapping[str, Any]) -> dict[str, str]:
    """Return form errors for the second provider, or {} when valid.

    Shared by both flows (initial and options) for the same reason
    `_label_errors` is: two copies drift, and the one that drifts stores a
    configuration that can never work. The rule is all-or-nothing -- a URL
    without a key is not a provider, and storing it would make every failover
    attempt fail for a configuration reason, hiding the real failure behind a
    second one.

    Only the three fields that are required together count as evidence. The two
    dropdowns are deliberately excluded: they have defaults (`default`), so they
    are present on every submit including an untouched one -- a user who only
    opens the form would otherwise be forced to fill in a provider they do not
    want. `vision.fallback_config_from` applies the same three-field rule, which
    is what makes this form's rejection agree with what the runtime will accept.

    Two checks, both about the *second* provider only: the three fields must be
    present as a group, and their address must parse. The address check exists
    because a malformed one is not a harmless extra -- the fallback is reached
    exactly when the primary has already failed, so an unparseable URL stacks a
    second failure on the first and the diagnosis of the first is what gets
    lost. This layer reports the typo while both providers are still healthy.
    """
    present = [
        key
        for key in (
            CONF_FALLBACK_LLM_BASE_URL,
            CONF_FALLBACK_LLM_API_KEY,
            CONF_FALLBACK_LLM_MODEL,
        )
        if str(user_input.get(key) or "").strip()
    ]
    if present and len(present) < 3:
        return {"base": "fallback_incomplete"}
    if present:
        # All three are present, so this is a provider the user means to use.
        # Its address must survive the same parse the primary's does, because a
        # malformed one is not merely useless -- it fails *at the moment the
        # primary has already failed*, so the second fault buries the first.
        # Refusing it here means the typo is reported while both providers are
        # still healthy and the user is looking at the form.
        #
        # Only the fallback is checked. The primary's URL is not run through
        # `_parse_base_url` on this path and never has been; adding that here
        # would start rejecting values that already work, which is a behaviour
        # change no typo fix is allowed to make.
        try:
            _parse_base_url(str(user_input[CONF_FALLBACK_LLM_BASE_URL]).strip())
        except InvalidURL:
            # `_parse_base_url` raises HA's `yarl.InvalidURL`, not an aiohttp
            # error -- a different family from the one `async_request` catches,
            # which is exactly why the two layers are checked independently.
            return {"base": "fallback_invalid_url"}
    return {}


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
                return await self.async_step_llmvision()

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

    async def async_step_llmvision(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        suggestions: dict[str, Any] = {}
        if user_input is not None:
            # A provider chosen on its own pre-fills the URL and returns the form,
            # so the URL is filled in rather than typed from memory. Submitting the
            # provider together with a URL means it is the final answer.
            provider = str(user_input.get(CONF_LLM_PROVIDER, "")).strip()
            preset_url = _preset_base_url(provider)
            typed_url = str(user_input.get(CONF_LLM_BASE_URL, "")).strip()
            wants_only_the_preset = (
                preset_url is not None
                and not typed_url
                and not str(user_input.get(CONF_LLM_API_KEY, "")).strip()
            )
            if wants_only_the_preset:
                return self.async_show_form(
                    step_id="llmvision",
                    data_schema=self.add_suggested_values_to_schema(
                        _llm_schema(),
                        _strip_secrets(
                            {
                                CONF_LLM_PROVIDER: provider,
                                CONF_LLM_BASE_URL: preset_url,
                            }
                        ),
                    ),
                    errors=errors,
                )
            # A provider picked from the list is authoritative for the URL: the
            # field is pre-filled, so leaving the old value in place would silently
            # send requests to the previous provider.
            base_url = typed_url or (preset_url or "")
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
            suggestions = {
                CONF_LLM_PROVIDER: provider,
                CONF_LLM_BASE_URL: base_url,
                CONF_LLM_MODEL: model,
            }
        return self.async_show_form(
            step_id="llmvision",
            # Stripped even though this path's `suggestions` are built from explicit
            # non-secret fields: the guarantee belongs with the schema that carries a
            # key, not with each caller's care. A future edit that adds the typed key
            # here would otherwise ship it back to the browser.
            data_schema=self.add_suggested_values_to_schema(
                _llm_schema(), _strip_secrets(suggestions)
            ),
            errors=errors,
        )

    async def async_step_options(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        if user_input is not None:
            # 初始流也要校验，与选项流共用同一个 helper：标签会被原样存下，填错
            # 了之后每次分析都失败，而用户只看到「没有通知」。两个流的校验一旦各
            # 写一份，就会有一个先漂移。
            #
            # 三份错误合并而不是二选一：`_label_errors` 的键是字段名、另两份是
            # `base`，两者互不相同，合并后能同时显示——否则修好一个才会看见另一个，
            # 用户要提交两次才知道全部问题。主组地址与第二组地址都校验，理由见
            # `_primary_url_errors`。
            errors = {
                **_primary_url_errors(user_input),
                **_fallback_errors(user_input),
                **_label_errors(user_input),
            }
            if not errors:
                return self.async_create_entry(
                    title=self._data[CONF_NAME], data=self._data, options=user_input
                )
            # 校验失败时回填用户这次提交的内容，否则表单会清空他填的一切。
            return self.async_show_form(
                step_id="options",
                data_schema=self.add_suggested_values_to_schema(
                    _setup_schema(), user_input
                ),
                errors=errors,
            )
        return self.async_show_form(
            step_id="options",
            data_schema=self.add_suggested_values_to_schema(
                _setup_schema(),
                self._data.get("llm") or {},
            ),
        )
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
                return self.async_update_reload_and_abort(
                    entry,
                    unique_id=unique_id,
                    data=data,
                )
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
        #
        # The current provider is folded into the menu label because switching it
        # is the whole point of this dialog, and a user who cannot see which one
        # is active has to open the form to find out. `llm_provider` is not
        # stored -- the URL is the truth -- so it is derived from the stored URL.
        # `_PROVIDER_LABELS` has no `custom` key and this deployment's router URL
        # resolves to exactly that, so the fallback has to be readable text:
        # showing the raw key would defeat the point of showing it at all.
        options = self.config_entry.options
        current = _provider_for_url(str(options.get(CONF_LLM_BASE_URL, "")))
        label = _PROVIDER_LABELS.get(current, "其他 / Other")
        return self.async_show_menu(
            step_id="init",
            menu_options={
                "settings": "配置参数 / Configure settings",
                "provider": f"切换服务商（当前：{label}）",
            },
        )

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Group the settings form with the connection test that exercises it.

        The two were siblings in the top-level menu, which read as though the test
        were unrelated to the settings it tests. Home Assistant's form offers only
        one action -- the submit button -- so a button inside the form is not
        possible; a submenu expresses the same grouping with the mechanism the flow
        already uses.

        Labels are literal text because a dict `menu_options` value is rendered
        verbatim, with no translation lookup.
        """
        return self.async_show_menu(
            step_id="settings",
            menu_options={
                "settings_form": "修改参数 / Edit settings",
                "close_up_form": "人物特写 / Person close-up",
                "test_connection": "测试视觉模型连通性 / Test provider connection",
            },
        )

    async def async_step_close_up_form(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """The close-up's own settings, on their own short form.

        These two fields were the last rows of the 20-field settings form, which
        put them below the fold: the owner could not find the face-service field at
        all, and reported it as missing rather than as off-screen. A submenu item is
        one visible row instead of a field nobody scrolls to.

        The save MERGES into the existing options rather than replacing them.
        `async_create_entry(data=...)` replaces the whole options dict, so returning
        just these two fields would silently wipe the other eighteen -- the LLM
        settings, the zones, the retention windows. Merging is what makes a partial
        form safe here.
        """
        if user_input is not None:
            merged = dict(self.config_entry.options)
            merged.update(user_input)
            return self.async_create_entry(data=merged)
        return self.async_show_form(
            step_id="close_up_form",
            data_schema=self.add_suggested_values_to_schema(
                _close_up_schema(), self.config_entry.options
            ),
        )

    async def async_step_provider(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Offer the providers as a menu.

        A menu rather than a dropdown on the settings form, because a menu tap is a
        real round trip: the settings form is then rebuilt from the server's answer,
        so it arrives with the URL and model already filled in. Selecting a dropdown
        entry sends nothing -- server code only runs on submit -- which is why the
        earlier dropdown left the URL untouched and read as "nothing happened".

        The labels are literal text rather than translation keys because a dict
        `menu_options` value is rendered verbatim, with no translation lookup.
        """
        return self.async_show_menu(
            step_id="provider",
            menu_options={
                **{
                    f"provider_{name}": _PROVIDER_LABELS.get(name, name)
                    for name in PROVIDER_PRESETS
                },
                "provider_custom": "其他 / Other (type a URL)",
            },
        )

    async def _async_apply_provider(self, provider: str) -> ConfigFlowResult:
        """Show the settings form with this provider's URL and model filled in.

        The URL and the first preset model are passed as suggested values, which the
        frontend prefers over the schema defaults when it rebuilds the form. Both
        stay editable, so this is a starting point rather than a commitment --
        switching provider is a shortcut for filling the fields in, not a lock.

        `custom` fills in nothing: the user picked it precisely because they intend
        to type their own endpoint, and overwriting a working one is worse than
        leaving it alone.
        """
        suggestions: dict[str, Any] = dict(self.config_entry.options)
        preset_url = _preset_base_url(provider)
        preset = PROVIDER_PRESETS.get(provider)
        if preset_url is not None:
            suggestions[CONF_LLM_BASE_URL] = preset_url
        if preset is not None and preset["models"]:
            suggestions[CONF_LLM_MODEL] = preset["models"][0]
        return self.async_show_form(
            step_id="settings_form",
            data_schema=self.add_suggested_values_to_schema(
                _options_schema(), _strip_secrets(suggestions)
            ),
        )

    async def async_step_provider_deepseek(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._async_apply_provider("deepseek")

    async def async_step_provider_gemini(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._async_apply_provider("gemini")

    async def async_step_provider_glm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._async_apply_provider("glm")

    async def async_step_provider_openai(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._async_apply_provider("openai")

    async def async_step_provider_custom(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        return await self._async_apply_provider("custom")

    async def async_step_settings_form(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        errors: dict[str, str] = {}
        suggestions: Mapping[str, Any] = self.config_entry.options
        if user_input is not None:
            # 地址替换只发生在用户明确点了某个服务商时（见 `_async_apply_provider`），
            # 这里不再猜「地址有没有被改过」。
            #
            # 那套猜测是上一版的实现，也是「切换服务商看不出是否成功」的根源：它要
            # 比较表单显示值与存储值来判断用户意图，而表单显示值是从 URL 推导的、
            # 存储值却可能是另一个，两者不一致时判断就静默失效。现在选服务商是一次
            # 真正的服务端往返，地址在那一步就填进表单，用户看得见、也能改。
            #
            # 保存时就校验标签格式：这个选项流是整体替换，填错了会被直接存进去，
            # 之后每次分析都失败，而用户只看到「没有通知」——错误发生在离原因
            # 很远的地方。校验与初始流共用 `_label_errors`，两个流不会各漂各的。
            #
            # 第二组服务商同样在保存时校验，理由一样，而且更硬：半配置存进去之后
            # 每次失败转移都因配置原因失败，把真正的故障藏在第二个故障后面。
            # 两份错误合并（字段名 + `base`，互不覆盖），一次提交报出全部问题。
            #
            # Validate the options that will actually be STORED, not the raw
            # submission: a blank password box is not a missing credential once the
            # stored one is carried forward, and judging the raw payload would report
            # an untouched fallback as half-configured (`fallback_incomplete`) and
            # refuse a save the user made for an unrelated reason.
            merged_input = _merge_options(user_input, self.config_entry.options)
            errors = {
                **_primary_url_errors(merged_input),
                **_fallback_errors(merged_input),
                **_label_errors(merged_input),
            }
            if not errors:
                # Merge, never replace. `async_create_entry` replaces the whole
                # options dict, and this form is not the only owner of it: the
                # close-up fields live in `_close_up_schema()` and are reached
                # through their own submenu. Submitting this form alone therefore
                # used to delete `person_highlight` and `face_service_url` with no
                # error -- the user enables the close-up, later edits an unrelated
                # setting, and the feature silently turns off. `person_highlight`
                # also feeds `effective_prompt_version`, so the cache key reverted
                # with it. The sibling `close_up_form` already merged; this is the
                # same rule applied to the larger form.
                return self.async_create_entry(data=merged_input)
            # 校验失败时回填用户这次提交的内容，否则表单会清空他填的一切。
            suggestions = user_input
        return self.async_show_form(
            step_id="settings_form",
            data_schema=self.add_suggested_values_to_schema(
                _options_schema(), _strip_secrets(suggestions)
            ),
            errors=errors,
        )

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
