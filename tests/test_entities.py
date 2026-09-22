from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision.binary_sensor import HealthySensor
from custom_components.frigate_vision.event import ActivityEvent
from custom_components.frigate_vision.runtime import (
    EntryRuntime,
    IntegrationRuntime,
)
from custom_components.frigate_vision.select import ProcessingModeSelect
from custom_components.frigate_vision.store import ActivityStore


async def test_entities_have_stable_unique_ids_and_memory_state(
    hass: HomeAssistant,
) -> None:
    async def handler(value: object) -> None:
        return None

    queue = EntryRuntime(queue_size=1, handler=handler)
    await queue.async_start()
    runtime = IntegrationRuntime(
        hass=hass, store=ActivityStore(hass, "entry_1"), queue=queue
    )
    entry = MockConfigEntry(
        domain="frigate_vision",
        title="Front",
        entry_id="entry_1",
        options={"processing_mode": "observe"},
    )
    select = ProcessingModeSelect(entry, runtime, "processing_mode")
    healthy = HealthySensor(entry, runtime, "healthy")
    event = ActivityEvent(entry, runtime, "activity")
    assert select.unique_id == "entry_1_processing_mode"
    assert select.current_option == "observe"
    assert healthy.is_on is True
    assert event.unique_id == "entry_1_activity"
    await queue.async_stop()
