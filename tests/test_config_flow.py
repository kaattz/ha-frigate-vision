from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from aiohttp import InvalidURL, web
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision import config_flow
from custom_components.frigate_vision.const import (
    CONF_FALLBACK_LLM_API_KEY,
    CONF_FALLBACK_LLM_BASE_URL,
    CONF_FALLBACK_LLM_MODEL,
    CONF_FALLBACK_LLM_REASONING_EFFORT,
    CONF_FALLBACK_LLM_THINKING,
    CONF_LLM_BASE_URL,
    CONF_LLM_BASE_URL_DEFAULT,
    CONF_LLM_PROVIDER,
    PROVIDER_PRESETS,
)

DOMAIN = "frigate_vision"


def test_provider_presets_cover_the_supported_endpoints() -> None:
    """Each preset must give a base URL the client can actually reach.

    Typing a base URL by hand is the step most likely to be got wrong: the value
    must be the OpenAI-compatible root, and providers disagree on whether that
    includes a version segment or a trailing `/v1`. Measured on this deployment, a
    wrong form fails as `provider_http_404`, which reads like an outage rather
    than a typo.
    """
    from custom_components.frigate_vision.const import PROVIDER_PRESETS

    assert PROVIDER_PRESETS, "no presets to offer"
    for name, preset in PROVIDER_PRESETS.items():
        assert preset["base_url"].startswith("https://"), name
        assert not preset["base_url"].endswith("/"), (
            f"{name}: a trailing slash would produce a double slash once the "
            "client appends /chat/completions"
        )
        assert "chat/completions" not in preset["base_url"], (
            f"{name}: the preset is a base URL; the client appends the path"
        )
        assert preset["models"], f"{name}: needs at least one model to suggest"


def test_the_gemini_preset_points_at_googles_openai_endpoint() -> None:
    """The URL is the part that cannot be guessed, so it is pinned here.

    Google serves an OpenAI-compatible surface at a path that is not derivable
    from the provider's usual host: it carries a version segment *and* an `openai`
    segment, so `https://generativelanguage.googleapis.com/v1` -- the form every
    other provider here uses -- is wrong.
    """
    from custom_components.frigate_vision.const import PROVIDER_PRESETS

    gemini = PROVIDER_PRESETS["gemini"]
    assert (
        gemini["base_url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert "gemini-3.8-flash" in gemini["models"]


def test_a_provider_preset_supplies_the_base_url_only() -> None:
    """A preset must not carry a key or a model choice the user did not make.

    It fills in the part that is a fact about the provider; the credential is the
    user's and the model is a cost decision, so both stay theirs. Leaving them
    empty also keeps the "provider is incomplete" validation meaningful.
    """
    from custom_components.frigate_vision.const import PROVIDER_PRESETS

    for name, preset in PROVIDER_PRESETS.items():
        assert "api_key" not in preset, name
        assert "model" not in preset, name


def _field(schema, name: str):
    """Return (marker, selector) for a field by name.

    Iterating a voluptuous schema yields the *markers* (the keys), not the
    selectors, and a suggestion is attached to the marker's `description`. So both
    halves are needed and this returns them together.
    """
    for marker, selector in schema.schema.items():
        if getattr(marker, "schema", None) == name:
            return marker, selector
    raise AssertionError(f"{name} is missing from the schema")


def _suggested_base_url(schema) -> object:
    """Read the base URL's suggested value off its marker."""
    description = _field(schema, CONF_LLM_BASE_URL)[0].description
    if isinstance(description, dict):
        return description.get("suggested_value")
    return description


def test_the_provider_step_offers_a_dropdown_and_an_editable_url() -> None:
    """The provider step must let the URL be chosen *or* typed.

    Driven through the real schema rather than a helper, because the thing being
    tested is what the form offers the user. Two properties matter: a provider
    dropdown exists (so the URL is not typed from memory), and the base URL
    remains a free-text field (so a local router or proxy still works, which is
    how this deployment itself runs).
    """
    provider_names = set(PROVIDER_PRESETS)

    schema = config_flow._llm_schema()  # noqa: SLF001
    fields = {getattr(marker, "schema", None) for marker in schema.schema}
    assert CONF_LLM_PROVIDER in fields, "no provider choice is offered"
    assert CONF_LLM_BASE_URL in fields, "the URL must stay available"

    # The URL field must stay free text: a preset pre-fills it, but a local router
    # or reverse proxy has to remain typeable.
    _, url_selector = _field(schema, CONF_LLM_BASE_URL)
    assert type(url_selector).__name__ == "TextSelector", (
        "the base URL must remain editable text, not a fixed choice"
    )

    _, provider_selector = _field(schema, CONF_LLM_PROVIDER)
    options = provider_selector.config["options"]
    offered = {option["value"] for option in options}
    assert provider_names <= offered, (
        "every preset must be selectable, or its URL cannot be reached from the UI"
    )
    # The labels are shown to a user, so they must carry text rather than the raw
    # key -- an unlabelled row renders blank in this deployment.
    for option in options:
        assert option["label"], f"{option['value']} would render as a blank row"


async def test_choosing_a_provider_prefills_the_url_in_the_flow(
    hass: HomeAssistant,
) -> None:
    """Picking a provider must fill the URL in, not merely accept the choice.

    The suggestion is what removes the typing, so it is checked where the user
    sees it: the form returned after the choice carries the preset as the base
    URL's suggested value.
    """
    from custom_components.frigate_vision.const import PROVIDER_PRESETS

    flow = config_flow.FrigateEntryIntelligenceConfigFlow()
    flow.hass = hass
    flow._data = {"name": "Front Door"}  # noqa: SLF001

    result = await flow.async_step_llmvision({"llm_provider": "gemini"})
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "llmvision"
    assert _suggested_base_url(result["data_schema"]) == PROVIDER_PRESETS["gemini"][
        "base_url"
    ]


async def test_a_provider_outside_the_presets_keeps_the_typed_url(
    hass: HomeAssistant,
) -> None:
    """This deployment points at a local router, so a free-text URL must survive.

    Choosing a provider that is not in the list must not blank a URL the user
    already entered -- pick the wrong row and their working endpoint would be lost.
    """
    flow = config_flow.FrigateEntryIntelligenceConfigFlow()
    flow.hass = hass
    flow._data = {"name": "Front Door"}  # noqa: SLF001
    local = "http://192.168.166.50:7864/v1"

    result = await flow.async_step_llmvision(
        {"llm_provider": "custom", "llm_base_url": local}
    )
    assert result["type"] is FlowResultType.FORM
    assert _suggested_base_url(result["data_schema"]) == local


async def test_every_field_on_the_provider_step_has_a_translated_label() -> None:
    """A field with no label renders as its raw key in the UI.

    Measured on this deployment: the options menu rendered two unlabelled rows
    when its entries were given as a list of keys. The same failure applies to any
    new field -- it appears as `llm_provider` rather than "Provider", which reads
    as a bug in the integration rather than a missing string.

    Both properties matter: a field offered by the form must be named in the
    strings file, and a name nothing uses is a leftover that will confuse the next
    reader.
    """
    import json
    from pathlib import Path

    root = Path(config_flow.__file__).parent
    schema_fields = {
        getattr(marker, "schema", None)
        for marker in config_flow._llm_schema().schema  # noqa: SLF001
    }
    schema_fields.discard(None)

    for name in ("strings.json", "translations/en.json", "translations/zh-Hans.json"):
        payload = json.loads((root / name).read_text("utf-8"))
        labelled = set(
            payload.get("config", {})
            .get("step", {})
            .get("llmvision", {})
            .get("data", {})
        )
        missing = schema_fields - labelled
        assert not missing, f"{name} has no label for {sorted(missing)}"

async def test_native_validation_carries_login_cookie(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def login(request: web.Request) -> web.Response:
        response = web.json_response({"ok": True})
        response.set_cookie("frigate_token", "token-1")
        return response

    async def version(request: web.Request) -> web.Response:
        assert request.cookies["frigate_token"] == "token-1"
        return web.Response(text="0.17.2")

    app.router.add_post("/api/login", login)
    app.router.add_get("/api/version", version)
    server = await aiohttp_server(app)
    result = await config_flow.async_validate_frigate(
        hass, str(server.make_url("/")), "native", "admin", "secret"
    )
    assert result == "0.17.2"


async def test_user_flow_creates_entry_with_data_and_options(
    hass: HomeAssistant,
) -> None:
    provider = MockConfigEntry(
        domain="llmvision",
        title="Vision Provider",
        entry_id="provider-1",
    )
    provider.add_to_hass(hass)
    provider.mock_state(hass, config_entries.ConfigEntryState.LOADED)
    hass.services.async_register("llmvision", "image_analyzer", lambda call: None)

    with patch.object(
        config_flow,
        "async_validate_frigate",
        AsyncMock(return_value="0.17.2"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        assert result["type"] is FlowResultType.FORM
        assert result["step_id"] == "user"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Front Door",
                "base_url": "http://frigate.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front_door",
                "near_zones": "home_door",
                "transition_zones": "bench",
                "far_zones": "elevator,elevator_2",
            },
        )
        # The door step is gone: the flow goes straight from Frigate
        # connection to the vision provider.
        assert result["step_id"] == "llmvision"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "llm_base_url": "https://api.deepseek.com/v1",
                "llm_api_key": "test-key",
                "llm_model": "vision-model",
                "llm_thinking": "disabled",
            },
        )
        assert result["step_id"] == "options"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "target_width": 768,
                "max_tokens": 300,
                "output_language": "zh-CN",
                "history_retention_days": 30,
                "media_retention_days": 7,
                "queue_size": 10,
            },
        )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Front Door"
    assert result["data"]["frigate"]["camera"] == "front_door"
    assert result["data"]["zones"]["far"] == ["elevator", "elevator_2"]
    assert result["data"]["llm"]["llm_model"] == "vision-model"
    assert result["data"]["llm"]["llm_base_url"] == "https://api.deepseek.com/v1"
    assert result["data"]["llm"]["llm_thinking"] == "disabled"
    # No processing mode is stored at all: the pipeline is the only behaviour.
    assert "processing_mode" not in result["options"]


async def test_frigate_connection_error_stays_on_user_step(
    hass: HomeAssistant,
) -> None:
    with patch.object(
        config_flow,
        "async_validate_frigate",
        AsyncMock(side_effect=ConnectionError),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Front Door",
                "base_url": "http://frigate.invalid",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front_door",
                "near_zones": "home_door",
                "transition_zones": "bench",
                "far_zones": "elevator",
            },
        )

    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"
    assert result["errors"]["base"] == "cannot_connect"


async def test_invalid_url_and_zone_overlap_are_rejected(
    hass: HomeAssistant,
) -> None:
    with patch.object(
        config_flow,
        "async_validate_frigate",
        AsyncMock(side_effect=InvalidURL("bad")),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Front Door",
                "base_url": "http://user:secret@frigate.local",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front_door",
                "near_zones": "same",
                "transition_zones": "bench",
                "far_zones": "same",
            },
        )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"]["base"] in {"invalid_url", "invalid_zones"}


async def test_bad_port_is_reported_as_invalid_url(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Front Door",
            "base_url": "http://frigate.local:bad",
            "auth_mode": "none",
            "mqtt_topic_prefix": "frigate",
            "camera": "front",
            "near_zones": "near",
            "transition_zones": "mid",
            "far_zones": "far",
        },
    )
    assert result["step_id"] == "user"
    assert result["errors"]["base"] == "invalid_url"


async def test_duplicate_frigate_camera_is_rejected(hass: HomeAssistant) -> None:
    existing = MockConfigEntry(
        domain=DOMAIN,
        title="Existing",
        unique_id="http://frigate.local:5000|front_door",
    )
    existing.add_to_hass(hass)
    with patch.object(
        config_flow,
        "async_validate_frigate",
        AsyncMock(return_value="0.17.2"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Duplicate",
                "base_url": "http://frigate.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front_door",
                "near_zones": "home_door",
                "transition_zones": "bench",
                "far_zones": "elevator",
            },
        )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_options_flow_updates_behavior_settings(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "init"
    # The menu routes to settings or the provider switch. The connectivity check
    # lives inside「配置参数」now: it tests the settings, so listing it as a
    # sibling made it read as unrelated to them.
    assert set(result["menu_options"]) == {"settings", "provider"}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    # 「配置参数」是子菜单：表单在它里面，要多走一层。
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )
    assert result["step_id"] == "settings_form"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "target_width": 768,
            # The cap must cover the reasoning budget the provider needs:
            # measured, max_tokens at 800 returned empty content every time
            # while the model spent the whole budget reasoning. 20000 is
            # accepted by the endpoint.
            "max_tokens": 20000,
            "output_language": "zh-CN",
            "history_retention_days": 30,
            "media_retention_days": 7,
            "queue_size": 10,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert "processing_mode" not in result["data"]
    assert result["data"]["max_tokens"] == 20000


async def test_the_options_flow_still_saves_normally(hass: HomeAssistant) -> None:
    """加了 provider 交互之后，正常的保存路径必须仍然工作。

    `llm_provider` 不再提交：它已不是表单字段（选服务商改走菜单），而 HA 会用
    schema 校验提交内容，多出的键会让整个保存 400。
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "target_width": 768,
            "max_tokens": 20000,
            "output_language": "zh-CN",
            "history_retention_days": 30,
            "media_retention_days": 7,
            "queue_size": 10,
            "llm_base_url": "https://api.deepseek.com/v1",
            "llm_api_key": "k",
            "llm_model": "m",
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert "processing_mode" not in result["data"]


async def test_connection_test_reports_a_missing_key(hass: HomeAssistant) -> None:
    """The check must name what is missing rather than failing vaguely."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"llm_base_url": "https://x/v1"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    # 连通性测试现在在「配置参数」子菜单里，顶层已无此项。
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "test_connection"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "test_connection"
    assert result["errors"] == {"base": "llm_missing_key"}


async def test_connection_test_reports_success_with_details(
    hass: HomeAssistant,
) -> None:
    """A working setup must show the model and cost it measured."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={
            "llm_base_url": "https://api.example.com/v1",
            "llm_api_key": "key",
            "llm_model": "vision-model",
            "llm_thinking": "disabled",
        },
    )
    entry.add_to_hass(hass)

    async def fake_test(_session, _config):
        from custom_components.frigate_vision.vision import (
            ConnectionReport,
        )

        return ConnectionReport(model="vision-model", seconds=1.25, total_tokens=12)

    with patch.object(config_flow, "async_test_connection", fake_test):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "settings"}
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "test_connection"}
        )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] in (None, {})
    assert result["description_placeholders"]["model"] == "vision-model"
    assert result["description_placeholders"]["tokens"] == "12"


async def test_native_auth_requires_credentials(hass: HomeAssistant) -> None:
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {
            "name": "Front Door",
            "base_url": "https://frigate.local:8971",
            "auth_mode": "native",
            "mqtt_topic_prefix": "frigate",
            "camera": "front_door",
            "near_zones": "home_door",
            "transition_zones": "bench",
            "far_zones": "elevator",
        },
    )
    assert result["step_id"] == "user"
    assert result["errors"]["base"] == "invalid_auth"


async def test_reconfigure_updates_frigate_connection(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        unique_id="http://old.local:5000|front",
        data={
            "name": "Front Door",
            "frigate": {
                "base_url": "http://old.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "username": None,
                "password": None,
            },
            "zones": {"near": ["near"], "transition": ["mid"], "far": ["far"]},
            "door": {},
            "llmvision": {},
        },
    )
    entry.add_to_hass(hass)
    with patch.object(
        config_flow,
        "async_validate_frigate",
        AsyncMock(return_value="0.17.2"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": config_entries.SOURCE_RECONFIGURE,
                "entry_id": entry.entry_id,
            },
        )
        assert result["step_id"] == "reconfigure"
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "base_url": "http://new.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "near_zones": "near",
                "transition_zones": "mid",
                "far_zones": "far",
            },
        )
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["frigate"]["base_url"] == "http://new.local:5000"


async def test_reconfigure_none_clears_native_credentials(hass: HomeAssistant) -> None:
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        unique_id="https://old.local:8971|front",
        data={
            "name": "Front Door",
            "frigate": {
                "base_url": "https://old.local:8971",
                "auth_mode": "native",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "username": "admin",
                "password": "secret",
            },
            "zones": {"near": ["near"], "transition": ["mid"], "far": ["far"]},
            "door": {},
            "llmvision": {},
        },
    )
    entry.add_to_hass(hass)
    with patch.object(
        config_flow, "async_validate_frigate", AsyncMock(return_value="0.17.2")
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN,
            context={
                "source": config_entries.SOURCE_RECONFIGURE,
                "entry_id": entry.entry_id,
            },
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "base_url": "http://new.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "username": "admin",
                "password": "secret",
                "near_zones": "near",
                "transition_zones": "mid",
                "far_zones": "far",
            },
        )
    assert result["type"] is FlowResultType.ABORT
    assert entry.data["frigate"]["username"] is None
    assert entry.data["frigate"]["password"] is None


async def test_zone_lists_may_be_empty_for_review_only(hass: HomeAssistant) -> None:
    provider = MockConfigEntry(
        domain="llmvision",
        title="Vision Provider",
        entry_id="provider-1",
    )
    provider.add_to_hass(hass)
    provider.mock_state(hass, config_entries.ConfigEntryState.LOADED)
    hass.services.async_register("llmvision", "image_analyzer", lambda call: None)

    with patch.object(
        config_flow,
        "async_validate_frigate",
        AsyncMock(return_value="0.17.2"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Front Door",
                "base_url": "http://frigate.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front_door",
                "near_zones": "",
                "transition_zones": "",
                "far_zones": "",
            },
        )
        assert result["step_id"] == "llmvision"
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "llm_base_url": "https://api.deepseek.com/v1",
                "llm_api_key": "test-key",
                "llm_model": "vision-model",
                "llm_thinking": "disabled",
            },
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "target_width": 768,
                "max_tokens": 300,
                "output_language": "zh-CN",
                "history_retention_days": 30,
                "media_retention_days": 7,
                "queue_size": 10,
            },
        )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["zones"] == {"near": [], "transition": [], "far": []}


async def test_options_flow_round_trips_the_scene_description(
    hass: HomeAssistant,
) -> None:
    """The layout description must survive a save/read cycle.

    It is optional, so the flow has to accept both an omitted and a supplied
    value; an option the schema rejects would surface only when the user tried
    to save their camera's layout, which is the one moment it must work.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )
    description = "入户门在画面左侧画外；画面中央是电梯门，走廊远端通往另一部画外电梯。"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "target_width": 768,
            "max_tokens": 20000,
            "output_language": "zh-CN",
            "history_retention_days": 30,
            "media_retention_days": 7,
            "queue_size": 10,
            "scene_description": description,
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["scene_description"] == description


async def test_scene_description_reaches_the_vision_config(
    hass: HomeAssistant,
) -> None:
    """The stored option must actually reach the prompt, not just the entry.

    A saved option that never reaches `VisionConfig` would look configured in
    the UI while every analysis still ran the old prompt -- the silent-failure
    shape this project has hit before.
    """
    from custom_components.frigate_vision.vision import vision_config_from

    text = "入户门在画面左侧画外"
    config = vision_config_from({}, {"scene_description": text})
    assert config.scene_description == text

    # Unset must stay empty rather than becoming a placeholder.
    assert vision_config_from({}, {}).scene_description == ""


async def test_a_long_scene_description_is_truncated_not_rejected(
    hass: HomeAssistant,
) -> None:
    """A pasted essay must not break analysis.

    The cap protects the prompt budget. Truncating keeps a working
    configuration; rejecting at save time would lose the whole edit.
    """
    from custom_components.frigate_vision.const import MAX_SCENE_DESCRIPTION_LENGTH
    from custom_components.frigate_vision.vision import vision_config_from

    config = vision_config_from({}, {"scene_description": "长" * 2000})
    assert len(config.scene_description) == MAX_SCENE_DESCRIPTION_LENGTH


def test_the_options_form_offers_labels_and_a_prompt_override() -> None:
    """两个新字段必须在表单上，否则用户改不了。"""
    from custom_components.frigate_vision import config_flow
    from custom_components.frigate_vision.const import (
        CONF_PROMPT_OVERRIDE,
        CONF_SCENE_LABELS,
    )

    schema = config_flow._options_schema()
    fields = {getattr(m, "schema", None) for m in schema.schema}
    assert CONF_SCENE_LABELS in fields
    assert CONF_PROMPT_OVERRIDE in fields


def test_the_close_up_fields_are_reachable_and_on_their_own_form() -> None:
    """特写与人脸服务必须在【能到达的表单】上，且不再埋在 20 字段底部。

    这两项原本是 `_options_schema()` 的最后两行 —— 表单第 19/20 位。用户去找人脸
    服务地址时没找到，报了「没有这个字段」：一个要滚到底才看得见的字段，等于没有。

    同时钉住初始流仍然问这两项：初始流是一次性表单，拆分的只是选项流。
    """
    from custom_components.frigate_vision.const import (
        CONF_FACE_SERVICE_URL,
        CONF_PERSON_HIGHLIGHT,
    )
    from custom_components.frigate_vision.vision import (
        vision_config_from,
    )

    # 选项流：两项在各自的短表单里。
    close_up = {
        getattr(m, "schema", None) for m in config_flow._close_up_schema().schema
    }
    assert CONF_PERSON_HIGHLIGHT in close_up, "特写表单上没有人物特写开关"
    assert CONF_FACE_SERVICE_URL in close_up, "特写表单上没有人脸服务地址"

    # 大表单不再重复它们，否则同一项有两个入口、两处默认值。
    settings = {
        getattr(m, "schema", None) for m in config_flow._options_schema().schema
    }
    assert CONF_PERSON_HIGHLIGHT not in settings
    assert CONF_FACE_SERVICE_URL not in settings

    # 初始流是一次性表单，仍要问到这两项。
    setup = {getattr(m, "schema", None) for m in config_flow._setup_schema().schema}
    assert CONF_PERSON_HIGHLIGHT in setup
    assert CONF_FACE_SERVICE_URL in setup

    # 默认值是关的：既有部署保存一次表单不会意外打开一个会改变缓存键的开关。
    marker = next(
        m
        for m in config_flow._close_up_schema().schema
        if getattr(m, "schema", None) == CONF_PERSON_HIGHLIGHT
    )
    default = marker.default
    assert (default() if callable(default) else default) is False

    # 表单存下的值要能被读回配置对象，否则开关存了也不生效。
    assert vision_config_from({}, {CONF_PERSON_HIGHLIGHT: True}).person_highlight
    assert (
        vision_config_from({}, {CONF_FACE_SERVICE_URL: "http://h:8788"}).face_service_url
        == "http://h:8788"
    )


async def test_the_close_up_form_merges_instead_of_wiping_the_other_options(
    hass: HomeAssistant,
) -> None:
    """子菜单提交必须【合并】——否则它会静默清空另外 18 项设置。

    `async_create_entry(data=...)` 是整体替换。这个子菜单只提交 2 个字段，直接写回
    就会把 LLM 地址、密钥、分区、保留天数全部抹掉，而用户只看到「保存成功」。
    """
    from custom_components.frigate_vision.const import (
        CONF_FACE_SERVICE_URL,
        CONF_PERSON_HIGHLIGHT,
    )

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={
            "llm_base_url": "http://keep.me/v1",
            "llm_api_key": "secret",
            "target_width": 767,
            CONF_PERSON_HIGHLIGHT: False,
        },
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    # The new submenu row is what makes these fields reachable at all.
    assert "close_up_form" in result["menu_options"], "配置参数下没有人物特写入口"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "close_up_form"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            CONF_PERSON_HIGHLIGHT: True,
            CONF_FACE_SERVICE_URL: "http://192.168.166.50:8788",
        },
    )
    saved = entry.options
    assert saved["llm_base_url"] == "http://keep.me/v1", "合并丢了 LLM 地址"
    assert saved["llm_api_key"] == "secret", "合并丢了密钥"
    assert saved["target_width"] == 767, "合并丢了宽度"
    assert saved[CONF_PERSON_HIGHLIGHT] is True
    assert saved[CONF_FACE_SERVICE_URL] == "http://192.168.166.50:8788"


def test_every_options_field_has_a_translated_label() -> None:
    """选项表单上每个字段都要有标签，否则界面显示原始键名。

    scene_description 此前在三份文件里都缺标签，llm_reasoning_effort 与
    min_review_seconds 在 en.json 里也缺——用户看到的是 `scene_description`
    这样的原始键，读起来像集成出了 bug。
    """
    import json
    from pathlib import Path

    from custom_components.frigate_vision import config_flow

    root = Path(config_flow.__file__).parent
    fields = {
        getattr(m, "schema", None) for m in config_flow._options_schema().schema
    }
    fields.discard(None)
    for name in ("strings.json", "translations/en.json", "translations/zh-Hans.json"):
        payload = json.loads((root / name).read_text("utf-8"))
        labelled = set(payload["options"]["step"]["settings_form"].get("data", {}))
        missing = sorted(fields - labelled)
        assert not missing, f"{name} 缺少标签：{missing}"


def test_the_label_field_explains_the_halfwidth_colon() -> None:
    """字段说明必须写明用半角冒号——中文输入法默认打全角，会报格式错误。

    实测 `parse_scene_labels("宠物：只有宠物")`（全角冒号 U+FF1A）抛
    label_malformed。中文用户按默认输入法打字就会踩到，所以提示是必需的，
    不是客套话。
    """
    import json
    from pathlib import Path

    from custom_components.frigate_vision import config_flow
    from custom_components.frigate_vision.const import CONF_SCENE_LABELS

    root = Path(config_flow.__file__).parent
    for name in ("translations/en.json", "translations/zh-Hans.json"):
        payload = json.loads((root / name).read_text("utf-8"))
        descriptions = payload["options"]["step"]["settings_form"].get(
            "data_description", {}
        )
        text = descriptions.get(CONF_SCENE_LABELS, "")
        assert text, f"{name} 没有为 {CONF_SCENE_LABELS} 写字段说明"
        assert ":" in text, (
            f"{name} 的说明必须展示半角冒号的格式示例，"
            "否则中文输入法用户不知道要用半角"
        )


def test_an_invalid_label_line_is_rejected_when_saving() -> None:
    """保存时就报错，而不是等到分析失败——那时用户只看到「没有通知」。"""
    from custom_components.frigate_vision.scenes import parse_scene_labels

    with pytest.raises(ValueError, match="label_malformed"):
        parse_scene_labels("宠物 没有冒号")


async def test_saving_bad_labels_shows_an_error_instead_of_storing_them(
    hass: HomeAssistant,
) -> None:
    """畸形标签必须挡在保存之前，且表单要保留用户已填的其他内容。"""
    from custom_components.frigate_vision.const import CONF_SCENE_LABELS

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"scene_labels": ""},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )
    assert result["step_id"] == "settings_form"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "target_width": 768,
            "max_tokens": 20000,
            "output_language": "zh-CN",
            "history_retention_days": 30,
            "media_retention_days": 7,
            "queue_size": 10,
            CONF_SCENE_LABELS: "宠物 没有冒号",
        },
    )
    # 表单重新显示并带上错误，而不是创建条目。
    assert result["type"] is FlowResultType.FORM
    assert CONF_SCENE_LABELS in result.get("errors", {}), (
        "畸形标签必须在保存时报错"
    )


async def _drive_the_initial_flow_to_the_options_step(hass: HomeAssistant) -> str:
    """把初始配置流开到 options 这一步，返回 flow_id。

    初始流是 user → llmvision → options 三步（door 步已随门周期删除），只有
    最后一步碰标签。把驱动过程抽出来是为了让每个断言只讲它要讲的事，而不是把
    30 行表单填写抄一遍。
    """
    with patch.object(
        config_flow,
        "async_validate_frigate",
        AsyncMock(return_value="0.17.2"),
    ):
        result = await hass.config_entries.flow.async_init(
            DOMAIN, context={"source": config_entries.SOURCE_USER}
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "name": "Front Door",
                "base_url": "http://frigate.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front_door",
                "near_zones": "home_door",
                "transition_zones": "bench",
                "far_zones": "elevator",
            },
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "llm_base_url": "https://api.deepseek.com/v1",
                "llm_api_key": "test-key",
                "llm_model": "vision-model",
                "llm_thinking": "disabled",
            },
        )
    assert result["step_id"] == "options"
    return result["flow_id"]


_OPTIONS_PAYLOAD = {
    "target_width": 768,
    "max_tokens": 300,
    "output_language": "zh-CN",
    "history_retention_days": 30,
    "media_retention_days": 7,
    "queue_size": 10,
}


async def test_the_initial_flow_rejects_malformed_labels_too(
    hass: HomeAssistant,
) -> None:
    """初始流也必须校验标签，否则畸形配置会被静默存下。

    选项流有校验，初始流此前直接 `async_create_entry`。实测：初始流填入畸形标签
    → 静默存下 → 之后每次分析都失败，而用户只看到「没有通知」——错误发生在离
    原因很远的地方，而且已经存进 entry，不会因为重开表单就消失。
    """
    from custom_components.frigate_vision.const import CONF_SCENE_LABELS

    flow_id = await _drive_the_initial_flow_to_the_options_step(hass)

    result = await hass.config_entries.flow.async_configure(
        flow_id, {**_OPTIONS_PAYLOAD, CONF_SCENE_LABELS: "宠物 没有冒号"}
    )

    assert result["type"] is FlowResultType.FORM, (
        "畸形标签必须重新显示表单，而不是创建 entry"
    )
    assert result["step_id"] == "options"
    assert CONF_SCENE_LABELS in result.get("errors", {}), "畸形标签必须报错"
    # 关键：entry 不能被创建。只看返回值不够——静默存下才是这个缺陷的形态。
    assert hass.config_entries.async_entries(DOMAIN) == [], (
        "表单报错时 entry 不该被创建"
    )


async def test_the_initial_flow_still_creates_an_entry_with_valid_labels(
    hass: HomeAssistant,
) -> None:
    """校验不能误伤正常路径：合法标签必须照常建 entry。"""
    from custom_components.frigate_vision.const import DEFAULT_SCENE_LABELS

    flow_id = await _drive_the_initial_flow_to_the_options_step(hass)

    result = await hass.config_entries.flow.async_configure(
        flow_id, {**_OPTIONS_PAYLOAD, "scene_labels": DEFAULT_SCENE_LABELS}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["options"]["scene_labels"] == DEFAULT_SCENE_LABELS


def test_the_label_prefill_covers_every_scene_not_just_the_lobby() -> None:
    """预填必须覆盖所有场景的标签并集，不能只列电梯厅的 11 个。

    scene_labels 非空时会**全局替换** allowed 集合，不区分场景。漏掉某个标签它
    就永远无法报出，而且没有任何报错——short_roundtrip 在门场景删除后就落进过
    这个坑，靠这个测试抓出来。
    """
    from custom_components.frigate_vision.const import DEFAULT_SCENE_LABELS
    from custom_components.frigate_vision.scenes import SCENES, parse_scene_labels

    offered = {name for name, _ in parse_scene_labels(DEFAULT_SCENE_LABELS)}
    union = {label for scene in SCENES.values() for label in scene.classifications}
    assert offered == union, (
        f"预填标签与全场景并集不一致：缺 {sorted(union - offered)}，"
        f"多 {sorted(offered - union)}"
    )
    # 每个场景自己独有的标签都必须在里面。
    for mode, scene in SCENES.items():
        missing = sorted(set(scene.classifications) - offered)
        assert not missing, f"{mode} 的标签 {missing} 不在预填里"
    for name, definition in parse_scene_labels(DEFAULT_SCENE_LABELS):
        assert definition.strip(), f"{name} 的定义是空的"


def test_the_prompt_override_is_prefilled_with_the_builtin_rules() -> None:
    """规则预填必须与内置 template 逐字节相同，否则预填就成了另一套规则。"""
    from custom_components.frigate_vision.const import DEFAULT_PROMPT_OVERRIDE
    from custom_components.frigate_vision.scenes import SCENES

    assert DEFAULT_PROMPT_OVERRIDE == SCENES["review_six"].template


def test_the_option_defaults_are_the_prefilled_lobby_text() -> None:
    """两个字段的表单默认值必须是预填文本，界面才会显示出来。"""
    from custom_components.frigate_vision import config_flow
    from custom_components.frigate_vision.const import (
        CONF_PROMPT_OVERRIDE,
        CONF_SCENE_LABELS,
        DEFAULT_PROMPT_OVERRIDE,
        DEFAULT_SCENE_LABELS,
    )

    schema = config_flow._options_schema()
    found = {}
    for marker, _selector in schema.schema.items():
        name = getattr(marker, "schema", None)
        if name not in (CONF_SCENE_LABELS, CONF_PROMPT_OVERRIDE):
            continue
        # voluptuous 把默认值包成无参工厂：`vol.Optional.__init__` 里写的是
        # `self.default = default_factory(default)`，所以 `marker.default` 永远
        # 是函数而不是字面值，取默认值必须调用它。
        default = marker.default
        found[name] = default() if callable(default) else default
    assert found[CONF_SCENE_LABELS] == DEFAULT_SCENE_LABELS
    assert found[CONF_PROMPT_OVERRIDE] == DEFAULT_PROMPT_OVERRIDE


def test_an_entry_that_never_saved_them_still_uses_the_builtin_scene() -> None:
    """存储里没有值时必须仍然走内置场景——预填只影响界面，不改变既有行为。

    这是零回归的关键：线上 entry 的两个字段都是空的，加预填之后它的提示词必须
    逐字节不变。
    """
    from custom_components.frigate_vision.vision import vision_config_from

    config = vision_config_from(
        {},
        {
            "llm_base_url": "https://api.example.com/v1",
            "llm_api_key": "k",
            "llm_model": "m",
        },
    )
    assert config.scene_labels == ""
    assert config.prompt_override == ""


def test_the_initial_flow_has_labels_for_every_option_field() -> None:
    """初始配置流渲染的是同一个 _options_schema()，也要有全部字段标签。

    config.step.options 此前只有 9/18，缺的字段在界面显示成原始键名
    （例如 scene_labels），读起来像集成出了 bug。
    """
    import json
    from pathlib import Path

    from custom_components.frigate_vision import config_flow

    root = Path(config_flow.__file__).parent
    fields = {
        getattr(m, "schema", None) for m in config_flow._options_schema().schema
    }
    fields.discard(None)
    for name in ("strings.json", "translations/en.json", "translations/zh-Hans.json"):
        payload = json.loads((root / name).read_text("utf-8"))
        labelled = set(
            payload["config"]["step"]["options"].get("data", {})
        )
        missing = sorted(fields - labelled)
        assert not missing, f"{name} 的初始流缺少标签：{missing}"


def test_every_label_error_code_has_a_translation() -> None:
    """标签解析抛出的每个错误码都要有译文，否则用户看到 label_malformed 原始键。

    错误码是 parse_scene_labels 的 ValueError 消息，会经
    errors[CONF_SCENE_LABELS] 直接显示在表单上。
    """
    import json
    import re
    from pathlib import Path

    from custom_components.frigate_vision import config_flow

    source = Path(config_flow.__file__).parent.joinpath("scenes.py").read_text(
        "utf-8"
    )
    codes = set(re.findall(r'raise ValueError\("([a-z_]+)"\)', source))
    assert codes, "没有从 scenes.py 里找到任何错误码——正则可能过期了"
    # 闭环：光有正则不算数，抓到的每个码都必须是解析器**真的**会抛出来的。
    # 缺了这步，正则写错（例如只抓到一半）也会静默通过。
    from custom_components.frigate_vision.scenes import parse_scene_labels

    probes = {
        # 冒号缺失
        "label_malformed": ["没有冒号"],
        "label_name_too_long": ["a" * 193 + ": 定义"],
        # 制表符（\x09）落在 SAFE_CLASSIFICATION 排除的控制字符区间里；用内部
        # 制表符而不是换行——换行会被 splitlines 先切开，反而报 label_malformed。
        "label_name_invalid": ["坏\t名: 定义"],
        "label_definition_missing": ["宠物:"],
        "label_definition_too_long": ["宠物: " + "定" * 201],
        "label_duplicate": ["宠物: 甲\n宠物: 乙"],
        "too_many_labels": [f"标签{i}: 定义" for i in range(31)],
    }
    assert set(probes) == codes, (
        f"实测错误码与正则抓到的不一致：probes={sorted(probes)} codes={sorted(codes)}"
    )
    for code, lines in probes.items():
        with pytest.raises(ValueError) as caught:
            parse_scene_labels("\n".join(lines))
        assert str(caught.value) == code
    for name in ("strings.json", "translations/en.json", "translations/zh-Hans.json"):
        payload = json.loads(
            Path(config_flow.__file__).parent.joinpath(name).read_text("utf-8")
        )
        translated = set(payload["options"].get("error", {}))
        missing = sorted(codes - translated)
        assert not missing, f"{name} 缺少错误码译文：{missing}"


def test_the_label_errors_are_translated_where_each_flow_reads_them() -> None:
    """两个流读的是**不同**的翻译路径，两端都要有译文。

    前端把「字段级」错误按 flowType 解析到不同的键（实测 home-assistant/frontend
    的 show-dialog-config-flow.ts 与 show-dialog-options-flow.ts）：

      * 初始配置流 -> `component.<domain>.config.error.<code>`
      * 选项流     -> `component.<domain>.options.error.<code>`

    两条路径都**不含 step_id**——所以把错误码只挂在某个 step 下是查不到的。
    这条断言覆盖初始流的那个渲染点：`errors[CONF_SCENE_LABELS]` 正是在
    `step_id="options"` 的初始流表单上抛出的。
    """
    import json
    import re
    from pathlib import Path

    from custom_components.frigate_vision import config_flow

    source = Path(config_flow.__file__).parent.joinpath("scenes.py").read_text(
        "utf-8"
    )
    codes = set(re.findall(r'raise ValueError\("([a-z_]+)"\)', source))
    for name in ("strings.json", "translations/en.json", "translations/zh-Hans.json"):
        payload = json.loads(
            Path(config_flow.__file__).parent.joinpath(name).read_text("utf-8")
        )
        translated = set(payload["config"].get("error", {}))
        missing = sorted(codes - translated)
        assert not missing, f"{name} 的初始流缺少错误码译文：{missing}"


# ---------------------------------------------------------------------------
# The second provider (fallback): five plain fields on the settings form.
#
# Deliberately no preset menu: the primary provider has one because its URL is
# the part that cannot be guessed, but the fallback is usually "another known
# endpoint" and a menu of its own would double that machinery for little gain.
# ---------------------------------------------------------------------------

_FALLBACK_FIELDS = (
    CONF_FALLBACK_LLM_BASE_URL,
    CONF_FALLBACK_LLM_API_KEY,
    CONF_FALLBACK_LLM_MODEL,
    CONF_FALLBACK_LLM_THINKING,
    CONF_FALLBACK_LLM_REASONING_EFFORT,
)

# The three fields that are required *together*. `thinking` and
# `reasoning_effort` are deliberately absent: they have defaults, so they are
# never evidence that the user meant to configure a provider.
_FALLBACK_REQUIRED = (
    CONF_FALLBACK_LLM_BASE_URL,
    CONF_FALLBACK_LLM_API_KEY,
    CONF_FALLBACK_LLM_MODEL,
)


async def _open_the_settings_form(hass: HomeAssistant, entry) -> str:
    """Drive the options flow to the settings form, the way the frontend does.

    Returns the flow_id. The fallback fields live on this form, so every test
    below starts here rather than calling the step function directly -- what is
    being tested is what the user can actually reach.
    """
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )
    assert result["step_id"] == "settings_form"
    return result["flow_id"]


async def test_fallback_fields_are_reachable_from_the_settings_menu(
    hass: HomeAssistant,
) -> None:
    """第二组的五个字段在「配置参数」页，与主组分开。"""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"llm_base_url": "https://api.deepseek.com/v1"},
    )
    entry.add_to_hass(hass)

    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert set(result["menu_options"]) == {"settings", "provider"}, (
        "第二组不该新增顶层菜单项：它只有输入框，没有预设菜单"
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    assert set(result["menu_options"]) == {
        "settings_form",
        "close_up_form",
        "test_connection",
    }, "预设菜单是主组专有的，第二组不加菜单"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )

    fields = {getattr(m, "schema", None) for m in result["data_schema"].schema}
    missing = sorted(set(_FALLBACK_FIELDS) - fields)
    assert not missing, f"配置参数表单上没有第二组的字段：{missing}"
    # 五个都在，且只有这五个：多出来的 fallback_llm_provider 之类正是被否掉的预设菜单。
    assert {f for f in fields if f and str(f).startswith("fallback_")} == set(
        _FALLBACK_FIELDS
    ), "第二组只有这五个字段，没有预设菜单"

    schema = result["data_schema"]
    # 地址默认值必须是空串，不能是主组的默认地址——否则第二组会「默认开启」，
    # 而它默认必须是关的（见 vision.fallback_config_from：三项齐全才算配置）。
    url_default = _field(schema, CONF_FALLBACK_LLM_BASE_URL)[0].default
    url_default = url_default() if callable(url_default) else url_default
    assert url_default == "", f"第二组地址默认值应为空串，实际：{url_default!r}"
    assert url_default != CONF_LLM_BASE_URL_DEFAULT

    for name, expected in (
        (CONF_FALLBACK_LLM_THINKING, "default"),
        (CONF_FALLBACK_LLM_REASONING_EFFORT, "default"),
    ):
        marker, dropdown = _field(schema, name)
        default = marker.default
        default = default() if callable(default) else default
        assert default == expected, f"{name} 默认值应为 {expected}，实际 {default!r}"
        assert type(dropdown).__name__ == "SelectSelector", f"{name} 应是下拉框"

    # 密钥必须是密码框：它会像主组密钥一样显示在表单上。
    _, key_selector = _field(schema, CONF_FALLBACK_LLM_API_KEY)
    assert key_selector.config["type"] == "password", "第二组密钥必须是密码框"

    texts = [
        _field(schema, CONF_FALLBACK_LLM_BASE_URL)[1],
        _field(schema, CONF_FALLBACK_LLM_MODEL)[1],
    ]
    for selector_ in texts:
        assert type(selector_).__name__ == "TextSelector", (
            "地址与模型名是自由文本：本地路由器或反向代理都要能填"
        )


@pytest.mark.parametrize(
    "provided",
    [
        (CONF_FALLBACK_LLM_BASE_URL,),
        (CONF_FALLBACK_LLM_API_KEY,),
        (CONF_FALLBACK_LLM_MODEL,),
        (CONF_FALLBACK_LLM_BASE_URL, CONF_FALLBACK_LLM_API_KEY),
        (CONF_FALLBACK_LLM_BASE_URL, CONF_FALLBACK_LLM_MODEL),
        (CONF_FALLBACK_LLM_API_KEY, CONF_FALLBACK_LLM_MODEL),
    ],
    ids=lambda keys: "+".join(k.removeprefix("fallback_llm_") for k in keys),
)
async def test_a_half_configured_fallback_is_rejected(
    hass: HomeAssistant, provided: tuple[str, ...]
) -> None:
    """只填 URL 不给 Key -> 表单报错，而不是存下一个永远失败的半配置。"""
    values = {
        CONF_FALLBACK_LLM_BASE_URL: "https://backup.example.com/v1",
        CONF_FALLBACK_LLM_API_KEY: "backup-key",
        CONF_FALLBACK_LLM_MODEL: "backup-model",
    }
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"llm_base_url": "https://api.deepseek.com/v1", "max_tokens": 1234},
    )
    entry.add_to_hass(hass)
    flow_id = await _open_the_settings_form(hass, entry)

    result = await hass.config_entries.options.async_configure(
        flow_id,
        {**_OPTIONS_PAYLOAD, **{key: values[key] for key in provided}},
    )

    assert result["type"] is FlowResultType.FORM, (
        "三项必填只给了一部分时必须重新显示表单，而不是创建 entry"
    )
    assert result["step_id"] == "settings_form"
    assert result.get("errors", {}).get("base") == "fallback_incomplete", (
        f"只填了 {provided} 就保存，会存下一个永远失败的半配置"
    )
    # 关键：半配置不能被写进 entry。前面「重新显示表单」还不够——静默存下
    # 才是这个缺陷的形态。
    assert entry.options.get("max_tokens") == 1234, "表单报错时不该改动已存的选项"
    for key in values:
        assert key not in entry.options, f"半配置 {key} 被存进了 entry"


@pytest.mark.parametrize(
    "touched",
    [
        {},
        # What the real frontend submits when the user leaves the fallback
        # alone: the dropdowns come back at their defaults, and the text fields
        # are empty strings. This is the default single-provider state.
        {
            CONF_FALLBACK_LLM_BASE_URL: "",
            CONF_FALLBACK_LLM_API_KEY: "",
            CONF_FALLBACK_LLM_MODEL: "",
            CONF_FALLBACK_LLM_THINKING: "default",
            CONF_FALLBACK_LLM_REASONING_EFFORT: "default",
        },
    ],
    ids=["no-fallback-keys", "the-fields-at-their-defaults"],
)
async def test_an_empty_fallback_is_accepted(
    hass: HomeAssistant, touched: dict[str, str]
) -> None:
    """一个都不填 = 不启用第二组，这是默认状态。"""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"llm_base_url": "https://api.deepseek.com/v1"},
    )
    entry.add_to_hass(hass)
    flow_id = await _open_the_settings_form(hass, entry)

    result = await hass.config_entries.options.async_configure(
        flow_id, {**_OPTIONS_PAYLOAD, **touched}
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY, (
        f"空配置必须能保存，实际错误：{result.get('errors')}"
    )


@pytest.mark.parametrize(
    ("thinking", "effort"),
    [("disabled", "high"), ("default", "default")],
)
async def test_touching_only_a_dropdown_does_not_require_a_provider(
    hass: HomeAssistant, thinking: str, effort: str
) -> None:
    """只动了第二组的两个下拉框，不等于「在配置第二组」。

    两个下拉框有默认值（default），所以它们永远「有值」。若把「任何第二组字段有值」
    当成半配置，那么用户随手把推理模式调成「关闭」就会被迫去填一个他并不想要的
    备用服务商——而那才是默认状态下的正常提交。
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"llm_base_url": "https://api.deepseek.com/v1"},
    )
    entry.add_to_hass(hass)
    flow_id = await _open_the_settings_form(hass, entry)

    result = await hass.config_entries.options.async_configure(
        flow_id,
        {
            **_OPTIONS_PAYLOAD,
            CONF_FALLBACK_LLM_THINKING: thinking,
            CONF_FALLBACK_LLM_REASONING_EFFORT: effort,
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY, (
        f"下拉框有默认值，不该被当成半配置，实际错误：{result.get('errors')}"
    )
    assert result["data"][CONF_FALLBACK_LLM_THINKING] == thinking
    assert result["data"][CONF_FALLBACK_LLM_REASONING_EFFORT] == effort


async def test_a_complete_fallback_is_saved(hass: HomeAssistant) -> None:
    """三个必填项齐全时正常保存。"""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"llm_base_url": "https://api.deepseek.com/v1"},
    )
    entry.add_to_hass(hass)
    flow_id = await _open_the_settings_form(hass, entry)

    submitted = {
        **_OPTIONS_PAYLOAD,
        CONF_FALLBACK_LLM_BASE_URL: "https://backup.example.com/v1",
        CONF_FALLBACK_LLM_API_KEY: "backup-key",
        CONF_FALLBACK_LLM_MODEL: "backup-model",
        CONF_FALLBACK_LLM_THINKING: "disabled",
        CONF_FALLBACK_LLM_REASONING_EFFORT: "low",
    }
    result = await hass.config_entries.options.async_configure(flow_id, submitted)

    assert result["type"] is FlowResultType.CREATE_ENTRY, (
        f"三项齐全必须能保存，实际错误：{result.get('errors')}"
    )
    for key in _FALLBACK_FIELDS:
        assert result["data"][key] == submitted[key], f"{key} 没有被存下"

    # 存下的东西必须真的能被解析成第二组服务商——否则表单收了一份永远用不上的
    # 配置，而「保存成功」会让用户以为已经生效。
    from custom_components.frigate_vision.vision import fallback_config_from

    config = fallback_config_from({}, dict(result["data"]))
    assert config is not None, "三项齐全却没有被解析成第二组服务商"
    assert config.base_url == "https://backup.example.com/v1"
    assert config.model == "backup-model"
    assert config.thinking == "disabled"
    assert config.reasoning_effort == "low"


async def test_the_initial_flow_rejects_a_half_configured_fallback_too(
    hass: HomeAssistant,
) -> None:
    """初始流渲染同一个 `_options_schema()`，也要挡下半配置。

    两个流各写一份校验就会有一个先漂移，而漂移的那一份会把半配置存进 entry——
    之后每次失败转移都因为配置原因失败，把真正的故障藏在第二个故障后面。
    """
    flow_id = await _drive_the_initial_flow_to_the_options_step(hass)

    result = await hass.config_entries.flow.async_configure(
        flow_id,
        {**_OPTIONS_PAYLOAD, CONF_FALLBACK_LLM_BASE_URL: "https://backup/v1"},
    )

    assert result["type"] is FlowResultType.FORM, (
        "初始流也必须重新显示表单，而不是创建 entry"
    )
    assert result["step_id"] == "options"
    assert result.get("errors", {}).get("base") == "fallback_incomplete"
    assert hass.config_entries.async_entries(DOMAIN) == [], (
        "表单报错时 entry 不该被创建"
    )


async def test_a_bad_label_and_a_half_fallback_are_both_reported(
    hass: HomeAssistant,
) -> None:
    """两份校验合并上报，不许一个盖住另一个。

    键本来就不同（标签错 -> 字段名，第二组错 -> `base`），所以合并是安全的；
    反过来若写成 `errors = _fallback_errors(...) or _label_errors(...)`，用户会
    先修好一个、再提交一次才看到另一个——修一个冒一个。
    """
    from custom_components.frigate_vision.const import CONF_SCENE_LABELS

    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"llm_base_url": "https://api.deepseek.com/v1"},
    )
    entry.add_to_hass(hass)
    flow_id = await _open_the_settings_form(hass, entry)

    result = await hass.config_entries.options.async_configure(
        flow_id,
        {
            **_OPTIONS_PAYLOAD,
            CONF_SCENE_LABELS: "宠物 没有冒号",
            CONF_FALLBACK_LLM_BASE_URL: "https://backup.example.com/v1",
        },
    )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"].get("base") == "fallback_incomplete", (
        f"第二组的错误被标签错误盖住了：{result['errors']}"
    )
    assert CONF_SCENE_LABELS in result["errors"], (
        f"标签的错误被第二组的错误盖住了：{result['errors']}"
    )


def test_the_fallback_error_is_translated_where_each_flow_reads_it() -> None:
    """错误码要在**两条**路径上都有译文，因为它由两个流各自抛出。

    简报只说加在 `config.error`，但那只够初始流：前端按 flowType 解析基础错误，
    初始流读 `component.<domain>.config.error.<code>`，选项流读
    `component.<domain>.options.error.<code>`（见上面那条标签测试里引的
    frontend 源码）。而设置表单在选项流里，`base` 错误就在那里抛出——只加一份的
    话，用户在选项流里看到的会是原始键名 `fallback_incomplete`。

    这里同时也检查 `data` 块：初始流渲染 `config.step.options`，选项流渲染
    `options.step.settings_form`，同一个 `_options_schema()` 两处都要五个标签。
    """
    import json
    from pathlib import Path

    from custom_components.frigate_vision import config_flow

    root = Path(config_flow.__file__).parent
    for name in ("strings.json", "translations/en.json", "translations/zh-Hans.json"):
        payload = json.loads((root / name).read_text("utf-8"))
        for path, blocks in (
            (
                "config.error（初始流）",
                [payload["config"].get("error", {})],
            ),
            (
                "options.error（选项流）",
                [payload["options"].get("error", {})],
            ),
        ):
            for block in blocks:
                assert block.get("fallback_incomplete"), (
                    f"{name} 的 {path} 缺少 fallback_incomplete 译文"
                )
        for path, block in (
            (
                "config.step.options.data",
                payload["config"]["step"]["options"].get("data", {}),
            ),
            (
                "options.step.settings_form.data",
                payload["options"]["step"]["settings_form"].get("data", {}),
            ),
        ):
            missing = sorted(set(_FALLBACK_FIELDS) - set(block))
            assert not missing, f"{name} 的 {path} 缺少标签：{missing}"

