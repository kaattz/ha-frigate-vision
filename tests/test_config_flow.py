from __future__ import annotations

from unittest.mock import AsyncMock, patch

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
    # The menu routes to settings or a connectivity check, so provider fields
    # can be changed without re-adding the integration.
    assert set(result["menu_options"]) == {"settings", "test_connection"}
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

