from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.door import (
    DoorCycleCoordinator,
    DoorCycleError,
    DoorMapping,
    async_subscribe_lock,
    contact_is_open,
    doorbell_message,
    normalize_lock_event,
)
from custom_components.frigate_vision.models import (
    ActivityStage,
    ProcessingMode,
)
from custom_components.frigate_vision.store import ActivityStore

MAPPING = DoorMapping(
    action_attribute="锁动作",
    open_values=frozenset({"2"}),
    close_values=frozenset({"1"}),
    side_attribute="操作位置",
    inside_values=frozenset({"1"}),
    outside_values=frozenset({"2"}),
)


async def test_concurrent_open_during_save_keeps_one_cycle(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def save(payload):
        entered.set()
        await release.wait()

    store._store.async_save = AsyncMock(side_effect=save)
    coordinator = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    first = asyncio.create_task(coordinator.async_open(100, "outside"))
    await entered.wait()
    second = asyncio.create_task(coordinator.async_open(101, "outside"))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(first, second)
    assert len(store.all()) == 1
    assert [result.status for result in results] == ["opened", "duplicate_open"]


def test_loock_attributes_normalize_inside_outside_and_close() -> None:
    assert normalize_lock_event({"锁动作": 2, "操作位置": 1}, MAPPING) == (
        "open",
        "inside",
    )
    assert normalize_lock_event({"锁动作": 2, "操作位置": 2}, MAPPING) == (
        "open",
        "outside",
    )
    assert normalize_lock_event({"锁动作": 1, "操作位置": 3}, MAPPING) == (
        "close",
        "unknown",
    )


def test_missing_or_unknown_lock_attributes_fail_explicitly() -> None:
    with pytest.raises(DoorCycleError, match="door_attribute_missing"):
        normalize_lock_event({"锁动作": 2}, MAPPING)
    with pytest.raises(DoorCycleError, match="door_action_unknown"):
        normalize_lock_event({"锁动作": 9, "操作位置": 1}, MAPPING)
    with pytest.raises(DoorCycleError, match="door_side_unknown"):
        normalize_lock_event({"锁动作": 2, "操作位置": 3}, MAPPING)


async def test_open_duplicate_and_close_keep_one_cycle(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    coordinator = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    opened = await coordinator.async_open(100, "inside")
    assert opened.record.opening_side == "inside"
    with_doorbell = await coordinator.async_doorbell(102)
    assert with_doorbell is not None
    assert with_doorbell.doorbell_at == 102
    duplicate = await coordinator.async_open(105, "outside")
    assert duplicate.status == "duplicate_open"
    assert duplicate.record.activity_id == opened.record.activity_id
    assert duplicate.record.created_at == 100
    closed = await coordinator.async_close(120)
    assert closed.record.stage is ActivityStage.SEALED
    assert closed.record.door_closed_at == 120
    assert closed.record.door_remained_open is None
    assert closed.record.association_deadline == 130
    assert closed.record.finalization_deadline == 240


async def test_close_without_open_fails(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    coordinator = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    with pytest.raises(DoorCycleError, match="close_without_open"):
        await coordinator.async_close(120)


async def test_collecting_cycle_can_be_rebound_after_restart(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    first = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    opened = await first.async_open(100, "inside")
    restored = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        active_id=opened.record.activity_id,
    )
    closed = await restored.async_close(120)
    assert closed.record.stage is ActivityStage.SEALED


async def test_contact_evidence_distinguishes_continuous_open_and_reopen(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    continuous = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    await continuous.async_open(100, "inside")
    await continuous.async_contact(True, 101)
    await continuous.async_contact(False, 119)
    assert (await continuous.async_close(120)).record.door_remained_open is True

    reopened = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    await reopened.async_open(200, "inside")
    await reopened.async_contact(True, 201)
    await reopened.async_contact(False, 210)
    await reopened.async_contact(True, 211)
    await reopened.async_contact(False, 219)
    restored = DoorCycleCoordinator(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        active_id="door_entry_1_200000",
    )
    assert (await restored.async_close(220)).record.door_remained_open is False


async def test_lock_listener_unsubscribes_cleanly(hass: HomeAssistant) -> None:
    received: list[tuple[str, str]] = []

    async def callback(action: str, side: str, occurred_at: float) -> None:
        received.append((action, side))

    unsubscribe = async_subscribe_lock(hass, "event.front_lock", MAPPING, callback)
    hass.states.async_set(
        "event.front_lock", "2026-09-01T00:00:00+00:00", {"锁动作": 2, "操作位置": 1}
    )
    await hass.async_block_till_done()
    assert received == [("open", "inside")]
    unsubscribe()
    hass.states.async_set(
        "event.front_lock", "2026-09-01T00:00:01+00:00", {"锁动作": 1, "操作位置": 3}
    )
    await hass.async_block_till_done()
    assert received == [("open", "inside")]


async def test_lock_listener_reports_mapping_error(hass: HomeAssistant) -> None:
    errors: list[str] = []

    async def callback(action: str, side: str, occurred_at: float) -> None:
        return None

    unsubscribe = async_subscribe_lock(
        hass,
        "event.front_lock",
        MAPPING,
        callback,
        lambda exc: errors.append(str(exc)),
    )
    hass.states.async_set("event.front_lock", "bad", {"锁动作": 2})
    await hass.async_block_till_done()
    assert errors == ["door_attribute_missing"]
    unsubscribe()


def test_contact_and_doorbell_are_context_only() -> None:
    assert contact_is_open("on") is True
    assert contact_is_open("off") is False
    assert contact_is_open("unknown") is None
    assert contact_is_open("unavailable") is None
    message = doorbell_message("entry_1", "doorbell_1", 10)
    assert message.kind.value == "doorbell"
    assert message.occurred_at == 10


async def test_reconnected_lock_does_not_replay_previous_event(
    hass: HomeAssistant,
) -> None:
    timestamp = "2026-09-01T00:00:00+00:00"
    attrs = {"锁动作": 2, "操作位置": 1}
    hass.states.async_set("event.front_lock", timestamp, attrs)
    received = []

    async def receive(action, side, occurred_at):
        received.append(occurred_at)

    unsubscribe = async_subscribe_lock(hass, "event.front_lock", MAPPING, receive)
    hass.states.async_set("event.front_lock", "unavailable")
    await hass.async_block_till_done()
    hass.states.async_set("event.front_lock", timestamp, attrs)
    await hass.async_block_till_done()
    assert received == []
    hass.states.async_set("event.front_lock", "2026-09-01T00:00:01+00:00", attrs)
    await hass.async_block_till_done()
    assert received == [1788220801.0]
    unsubscribe()
