from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision.delivery import (
    EVENT_ACTIVITY,
    DeliveryManager,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    ProcessingMode,
)
from custom_components.frigate_vision.store import (
    ActivityStore,
    StoreConflictError,
)


async def test_delivery_fires_once_and_requires_matching_ack(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    events = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=60)
    started = await manager.async_start("activity_1")
    await hass.async_block_till_done()
    assert len(events) == 1
    with pytest.raises(StoreConflictError, match="delivery_identity_mismatch"):
        await manager.async_ack("activity_1", "wrong")
    completed = await manager.async_ack("activity_1", started.delivery_attempt_id)
    assert completed.stage is ActivityStage.COMPLETED
    assert len(events) == 1
    await manager.async_stop()


async def test_delivery_timeout_becomes_unknown_without_refiring(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    events = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=0)
    await manager.async_start("activity_1")
    await hass.async_block_till_done()
    failed = store.get("activity_1")
    assert failed is not None and failed.stage is ActivityStage.FAILED
    assert failed.error_code == "delivery_outcome_unknown"
    assert len(events) == 1
    await manager.async_stop()


async def test_timeout_retries_temporary_store_write_failure(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    manager = DeliveryManager(hass, entry, store, ack_timeout=0.02)
    completed = asyncio.Event()
    unsubscribe = store.async_subscribe(lambda record: completed.set())
    await manager.async_start("activity_1")
    original = store._store.async_save  # type: ignore[attr-defined]
    failures = 0

    async def flaky(payload):
        nonlocal failures
        if failures == 0:
            failures += 1
            raise OSError("temporary")
        return await original(payload)

    store._store.async_save = AsyncMock(side_effect=flaky)  # type: ignore[method-assign]
    await asyncio.wait_for(completed.wait(), timeout=2)
    assert failures == 1
    assert store.get("activity_1").error_code == "delivery_outcome_unknown"  # type: ignore[union-attr]
    unsubscribe()
    await manager.async_stop()
