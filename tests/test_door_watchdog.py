"""Test that the stale-open watchdog fails a cycle without inventing a close."""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision import runtime as runtime_module
from custom_components.frigate_vision.door import DoorCycleCoordinator
from custom_components.frigate_vision.models import (
    ActivityStage,
    ProcessingMode,
)
from custom_components.frigate_vision.store import ActivityStore


async def test_door_watchdog_fails_stale_open(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    coordinator = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    opened = await coordinator.async_open(1000.0, "inside")
    assert opened.record.stage is ActivityStage.COLLECTING

    abandoned = await coordinator.async_abandon_stale_open(
        1000.0 + runtime_module.DOOR_OPEN_WATCHDOG_SECONDS + 1,
        threshold=runtime_module.DOOR_OPEN_WATCHDOG_SECONDS,
    )
    assert abandoned is not None
    assert abandoned.status == "abandoned"
    assert abandoned.record.stage is ActivityStage.FAILED
    assert abandoned.record.error_code == "door_open_too_long"


async def test_door_watchdog_keeps_recent_open(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    coordinator = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    await coordinator.async_open(1000.0, "outside")

    abandoned = await coordinator.async_abandon_stale_open(
        1000.0 + runtime_module.DOOR_OPEN_WATCHDOG_SECONDS - 1,
        threshold=runtime_module.DOOR_OPEN_WATCHDOG_SECONDS,
    )
    assert abandoned is None


async def test_door_watchdog_ignores_close_after_abandon(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    coordinator = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    await coordinator.async_open(1000.0, "inside")
    await coordinator.async_abandon_stale_open(
        1000.0 + runtime_module.DOOR_OPEN_WATCHDOG_SECONDS + 1,
        threshold=runtime_module.DOOR_OPEN_WATCHDOG_SECONDS,
    )

    with pytest.raises(Exception, match="close_without_open"):
        await coordinator.async_close(
            1000.0 + runtime_module.DOOR_OPEN_WATCHDOG_SECONDS + 60
        )
