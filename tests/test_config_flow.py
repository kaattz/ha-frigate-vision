from __future__ import annotations

from unittest.mock import AsyncMock, patch

from aiohttp import InvalidURL, web
from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision import config_flow

DOMAIN = "frigate_vision"


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
