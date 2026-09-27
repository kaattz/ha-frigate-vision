"""Entity identity and health surface.

The select entity this file used to cover was removed with the processing modes,
so only the health sensor and the activity event remain. The unique_id assertion
is the part worth keeping: it is the entity registry key, and a change to it
orphans the entity in every existing installation.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision.binary_sensor import HealthySensor
from custom_components.frigate_vision.event import ActivityEvent
from custom_components.frigate_vision.runtime import (
    EntryRuntime,
    IntegrationRuntime,
)
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
        options={},
    )
    healthy = HealthySensor(entry, runtime, "healthy")
    event = ActivityEvent(entry, runtime, "activity")
    assert healthy.unique_id == "entry_1_healthy"
    assert healthy.is_on is True
    assert event.unique_id == "entry_1_activity"
    # The event entity's types are part of its contract with the notification
    # blueprint, which filters on `activity_completed`.
    assert set(event.event_types) == {"activity_completed", "activity_failed"}
    await queue.async_stop()
