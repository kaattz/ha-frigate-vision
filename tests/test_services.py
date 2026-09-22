from __future__ import annotations

from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    ProcessingMode,
)
from custom_components.frigate_vision.runtime import (
    EntryRuntime,
    IntegrationRuntime,
)
from custom_components.frigate_vision.services import (
    async_register_services,
)
from custom_components.frigate_vision.store import ActivityStore


async def test_get_activity_returns_only_whitelisted_state(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.COMPLETED,
            processing_mode=ProcessingMode.SHADOW,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )

    async def handler(value: object) -> None:
        return None

    queue = EntryRuntime(queue_size=1, handler=handler)
    await queue.async_start()
    runtime = IntegrationRuntime(hass=hass, store=store, queue=queue)
    entry = MockConfigEntry(domain="frigate_vision", title="Front", entry_id="entry_1")
    entry.add_to_hass(hass)
    entry.runtime_data = runtime
    await async_register_services(hass)
    response = await hass.services.async_call(
        "frigate_vision",
        "get_activity",
        {"entry_id": "entry_1", "activity_id": "activity_1"},
        blocking=True,
        return_response=True,
    )
    assert response["classification"] == "visitor"
    assert "evidence_path" not in response
    assert "claimed_side_effects" not in response
    await queue.async_stop()
