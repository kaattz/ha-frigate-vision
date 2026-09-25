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
    CONF_LLM_BASE_URL,
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
        assert result["step_id"] == "door"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "event_entity_id": "event.front_door_lock",
                "action_attribute": "action",
                "open_values": "open",
                "close_values": "close",
                "side_attribute": "side",
                "inside_values": "inside",
                "outside_values": "outside",
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
        assert result["step_id"] == "options"

        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "processing_mode": "observe",
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
    assert result["options"]["processing_mode"] == "observe"


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
        options={"processing_mode": "observe"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    assert result["step_id"] == "init"
    # The menu routes to settings, the provider switch or a connectivity check,
    # so provider fields can be changed without re-adding the integration.
    assert set(result["menu_options"]) == {"settings", "provider", "test_connection"}
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    assert result["step_id"] == "settings"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "processing_mode": "shadow",
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
    assert result["data"]["processing_mode"] == "shadow"
    assert result["data"]["max_tokens"] == 20000


def test_the_options_form_also_offers_the_provider_dropdown() -> None:
    """服务商下拉必须在选项表单里也有，否则换供应商只能删掉集成重加。

    它原先只加在 _llm_schema()（初始配置流），而「行为选项」用的是
    _options_schema()，用户在那里看不到任何下拉项——而换供应商正是那个
    对话框最自然的用途。
    """
    from custom_components.frigate_vision import config_flow
    from custom_components.frigate_vision.const import PROVIDER_PRESETS

    schema = config_flow._options_schema()
    fields = {getattr(m, "schema", None) for m in schema.schema}
    assert "llm_provider" in fields, "选项表单缺少服务商下拉"
    assert "llm_base_url" in fields, "URL 仍须可编辑"

    for marker, selector in schema.schema.items():
        if getattr(marker, "schema", None) == "llm_provider":
            offered = {o["value"] for o in selector.config["options"]}
            assert set(PROVIDER_PRESETS) <= offered
            # 标签必须有文字，否则渲染成空行。
            for option in selector.config["options"]:
                assert option["label"]
            break


async def test_choosing_a_provider_in_the_options_flow_writes_the_url_at_once(
    hass: HomeAssistant,
) -> None:
    """在选项流里选服务商，地址必须在这一次提交就写进去并保存。

    不能靠「重新渲染表单让前端显示预填值」：HA 的原生表单只在**提交**时跑服务端
    代码，在下拉里选中一项不会触发任何请求，所以服务端无法在选中瞬间改地址。前一
    版实现就是这样，结果是用户选了 Gemini、点提交，什么都没保存、地址也没变——看
    起来完全没反应。
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"processing_mode": "observe", "llm_base_url": "http://old/v1"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    assert result["step_id"] == "settings"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {"llm_provider": "gemini", "processing_mode": "observe"},
    )
    # 一次提交就完成保存，地址被写成 gemini 的地址。
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert (
        result["data"]["llm_base_url"]
        == "https://generativelanguage.googleapis.com/v1beta/openai"
    )


async def test_the_options_flow_still_saves_normally(hass: HomeAssistant) -> None:
    """加了 provider 交互之后，正常的保存路径必须仍然工作。"""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"processing_mode": "observe"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "processing_mode": "shadow",
            "target_width": 768,
            "max_tokens": 20000,
            "output_language": "zh-CN",
            "history_retention_days": 30,
            "media_retention_days": 7,
            "queue_size": 10,
            "llm_base_url": "https://api.deepseek.com/v1",
            "llm_api_key": "k",
            "llm_model": "m",
            "llm_provider": "deepseek",
        },
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["processing_mode"] == "shadow"
    assert result["data"]["llm_provider"] == "deepseek"


async def test_connection_test_reports_a_missing_key(hass: HomeAssistant) -> None:
    """The check must name what is missing rather than failing vaguely."""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={"processing_mode": "observe", "llm_base_url": "https://x/v1"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
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
            "processing_mode": "observe",
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


async def test_door_mapping_values_must_be_disjoint(hass: HomeAssistant) -> None:
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
                "event_entity_id": "event.front_door_lock",
                "action_attribute": "action",
                "open_values": "same",
                "close_values": "same",
                "side_attribute": "side",
                "inside_values": "inside",
                "outside_values": "outside",
            },
        )
    assert result["step_id"] == "door"
    assert result["errors"]["base"] == "invalid_door_mapping"


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
        assert result["step_id"] == "reconfigure_door"
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["frigate"]["base_url"] == "http://new.local:5000"
    assert entry.data["door"] == {}


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
        assert result["step_id"] == "reconfigure_door"
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.ABORT
    assert entry.data["frigate"]["username"] is None
    assert entry.data["frigate"]["password"] is None


async def test_reconfigure_can_clear_the_lock_to_run_review_only(
    hass: HomeAssistant,
) -> None:
    """An existing door-cycle entry must be movable to review-only without
    deleting the entry, which would discard its stored activity history."""
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
            "zones": {
                "near": ["home_door"],
                "transition": ["bench"],
                "far": ["elevator"],
            },
            "door": {
                "event_entity_id": "event.lock_events",
                "action_attribute": "锁动作",
                "open_values": ["2"],
                "close_values": ["1"],
                "side_attribute": "操作位置",
                "inside_values": ["1"],
                "outside_values": ["2"],
                "contact_entity_id": None,
                "doorbell_event_entity_id": None,
            },
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
                "base_url": "http://old.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "near_zones": "home_door",
                "transition_zones": "bench",
                "far_zones": "elevator",
            },
        )
        assert result["step_id"] == "reconfigure_door"
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert entry.data["door"] == {}
    assert entry.data["zones"]["near"] == ["home_door"]


async def test_reconfigure_door_keeps_existing_mapping_when_resubmitted(
    hass: HomeAssistant,
) -> None:
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
            "zones": {"near": ["home_door"], "transition": [], "far": []},
            "door": {
                "event_entity_id": "event.lock_events",
                "action_attribute": "锁动作",
                "open_values": ["2"],
                "close_values": ["1"],
                "side_attribute": "操作位置",
                "inside_values": ["1"],
                "outside_values": ["2"],
                "contact_entity_id": None,
                "doorbell_event_entity_id": None,
            },
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
                "base_url": "http://old.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "near_zones": "home_door",
                "transition_zones": "",
                "far_zones": "",
            },
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "event_entity_id": "event.lock_events",
                "action_attribute": "锁动作",
                "open_values": "2",
                "close_values": "1",
                "side_attribute": "操作位置",
                "inside_values": "1",
                "outside_values": "2",
            },
        )
    assert result["type"] is FlowResultType.ABORT
    assert entry.data["door"]["open_values"] == ["2"]
    assert entry.data["door"]["close_values"] == ["1"]


async def test_reconfigure_door_rejects_contact_without_lock(
    hass: HomeAssistant,
) -> None:
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
            "zones": {"near": ["home_door"], "transition": [], "far": []},
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
                "base_url": "http://old.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "near_zones": "home_door",
                "transition_zones": "",
                "far_zones": "",
            },
        )
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {"contact_entity_id": "binary_sensor.front_contact"},
        )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reconfigure_door"
    assert result["errors"]["base"] == "door_lock_required"
    assert entry.data["door"] == {}


async def test_reconfigure_door_error_echoes_submitted_input(
    hass: HomeAssistant,
) -> None:
    """A rejected submission must re-render what the user typed.

    Rendering the stored mapping instead silently restores the lock the user
    just cleared, so they can clear the lock, get a door_lock_required error,
    clear the doorbell, and submit believing the lock is gone while the stored
    mapping is still intact.
    """
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
            "zones": {"near": ["home_door"], "transition": [], "far": []},
            "door": {
                "event_entity_id": "event.lock_events",
                "action_attribute": "锁动作",
                "open_values": ["2"],
                "close_values": ["1"],
                "side_attribute": "操作位置",
                "inside_values": ["1"],
                "outside_values": ["2"],
                "contact_entity_id": None,
                "doorbell_event_entity_id": "event.doorbell",
            },
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
                "base_url": "http://old.local:5000",
                "auth_mode": "none",
                "mqtt_topic_prefix": "frigate",
                "camera": "front",
                "near_zones": "home_door",
                "transition_zones": "",
                "far_zones": "",
            },
        )
        # User cleared the lock but left the doorbell: rejected.
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {
                "action_attribute": "锁动作",
                "open_values": "2",
                "close_values": "1",
                "side_attribute": "操作位置",
                "inside_values": "1",
                "outside_values": "2",
                "doorbell_event_entity_id": "event.doorbell",
            },
        )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"]["base"] == "door_lock_required"

    schema = result["data_schema"].schema
    suggested = {
        str(getattr(key, "schema", key)): (key.description or {}).get("suggested_value")
        for key in schema
    }
    # The lock the user cleared must stay cleared in the re-rendered form.
    assert suggested.get("event_entity_id") in (None, ""), suggested
    # The doorbell the user left in place must still be shown.
    assert suggested.get("doorbell_event_entity_id") == "event.doorbell", suggested


async def test_door_step_can_be_skipped_without_lock(hass: HomeAssistant) -> None:
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
                "near_zones": "home_door",
                "transition_zones": "bench",
                "far_zones": "elevator",
            },
        )
        assert result["step_id"] == "door"
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
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
                "processing_mode": "observe",
                "target_width": 768,
                "max_tokens": 300,
                "output_language": "zh-CN",
                "history_retention_days": 30,
                "media_retention_days": 7,
                "queue_size": 10,
            },
        )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"].get("door") == {}


async def test_door_contact_without_lock_is_rejected(hass: HomeAssistant) -> None:
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
            result["flow_id"], {"contact_entity_id": "binary_sensor.front_contact"}
        )
    assert result["step_id"] == "door"
    assert result["errors"]["base"] == "door_lock_required"


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
        assert result["step_id"] == "door"
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
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
                "processing_mode": "observe",
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
        options={"processing_mode": "observe"},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    description = "入户门在画面左侧画外；画面中央是电梯门，走廊远端通往另一部画外电梯。"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "processing_mode": "shadow",
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
        labelled = set(payload["options"]["step"]["settings"].get("data", {}))
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
        descriptions = payload["options"]["step"]["settings"].get(
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
        options={"processing_mode": "observe", "scene_labels": ""},
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    assert result["step_id"] == "settings"

    result = await hass.config_entries.options.async_configure(
        result["flow_id"],
        {
            "processing_mode": "observe",
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

    初始流是 user → door → llmvision → options 四步，只有最后一步碰标签。把驱动
    过程抽出来是为了让每个断言只讲它要讲的事，而不是把 30 行表单填写抄一遍。
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
        result = await hass.config_entries.flow.async_configure(result["flow_id"], {})
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
    "processing_mode": "observe",
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

    scene_labels 非空时会**全局替换** allowed 集合，不区分场景。door_roundtrip
    的 short_roundtrip 只属于它自己，若预填漏掉，门锁场景的「短暂外出」就永远
    无法报出——而且没有任何报错。
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

