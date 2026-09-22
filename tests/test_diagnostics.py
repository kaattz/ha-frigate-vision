from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision.diagnostics import (
    async_get_config_entry_diagnostics,
)
from custom_components.frigate_vision.runtime import (
    EntryRuntime,
    IntegrationRuntime,
)
from custom_components.frigate_vision.store import ActivityStore


async def test_diagnostics_redacts_credentials_and_person_data(
    hass: HomeAssistant,
) -> None:
    async def handler(value: object) -> None:
        return None

    queue = EntryRuntime(queue_size=1, handler=handler)
    await queue.async_start()
    entry = MockConfigEntry(
        domain="frigate_vision",
        entry_id="entry_1",
        title="Front",
        data={
            "frigate": {"base_url": "http://secret", "username": "u", "password": "p"}
        },
    )
    entry.runtime_data = IntegrationRuntime(
        hass=hass, store=ActivityStore(hass, "entry_1"), queue=queue
    )
    result = await async_get_config_entry_diagnostics(hass, entry)
    text = str(result)
    assert "http://secret" not in text and "'p'" not in text
    assert result["loaded"] is True
    await queue.async_stop()
