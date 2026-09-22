from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.frigate_vision as integration
from custom_components.frigate_vision.clip_proxy import FrigateClipView
from custom_components.frigate_vision.media_source import EvidenceMediaView
from custom_components.frigate_vision.runtime import IntegrationRuntime


async def test_setup_registers_authenticated_media_view(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Both public routes must be registered and must demand authentication.

    The evidence image and the activity clip are the two things a notification
    links to. Both proxy camera content, so neither may be reachable without a
    Home Assistant login -- that is what keeps Frigate itself off the internet.
    """
    registered: list[object] = []
    monkeypatch.setattr(hass, "http", SimpleNamespace(register_view=registered.append))
    assert await integration.async_setup(hass, {})

    kinds = {type(view) for view in registered}
    assert EvidenceMediaView in kinds
    assert FrigateClipView in kinds
    for view in registered:
        assert view.requires_auth is True, f"{type(view).__name__} is unauthenticated"


async def test_setup_and_unload_entry(hass: HomeAssistant, monkeypatch) -> None:
    entry = MockConfigEntry(
        domain="frigate_vision",
        title="Front Door",
        data={},
        options={},
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, config_entries.ConfigEntryState.LOADED)
    monkeypatch.setattr(hass.config_entries, "async_forward_entry_setups", AsyncMock())
    monkeypatch.setattr(
        hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)
    )

    assert hasattr(integration, "async_setup_entry")
    assert hasattr(integration, "async_unload_entry")
    assert await integration.async_setup_entry(hass, entry)
    runtime = entry.runtime_data
    assert isinstance(runtime, IntegrationRuntime)
    assert runtime.running
    assert await integration.async_unload_entry(hass, entry)
    assert not runtime.running
    assert entry.runtime_data is None
