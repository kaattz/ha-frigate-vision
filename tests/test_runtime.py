from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.frigate_vision.runtime as runtime_module
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    BufferedIngress,
    IngressKind,
    IngressMessage,
    ProcessingMode,
)
from custom_components.frigate_vision.runtime import (
    EntryRuntime,
    IntegrationRuntime,
    QueueFullError,
    RecoveryAction,
    build_recovery_work,
)
from custom_components.frigate_vision.store import ActivityStore


class _NoopMediaManager:
    def __init__(self, hass, store, *args, **kwargs):
        self.store = store

    async def async_restore_registry(self):
        return None

    async def async_cleanup(self, **kwargs):
        return ()

    async def async_build(self, activity_id):
        return self.store.get(activity_id)


async def test_runtime_processes_messages_in_order() -> None:
    processed: list[int] = []

    async def handler(value: int) -> None:
        processed.append(value)

    runtime = EntryRuntime(queue_size=2, handler=handler)
    await runtime.async_start()
    runtime.enqueue(1)
    runtime.enqueue(2)
    await runtime.async_join()
    await runtime.async_stop()
    assert processed == [1, 2]


async def test_runtime_queue_full_does_not_drop_oldest() -> None:
    gate = asyncio.Event()

    async def handler(value: int) -> None:
        await gate.wait()

    runtime = EntryRuntime(queue_size=1, handler=handler)
    runtime.enqueue(1)
    with pytest.raises(QueueFullError, match="queue_full"):
        runtime.enqueue(2)
    gate.set()


async def test_runtime_handler_error_is_reported_and_worker_continues() -> None:
    processed: list[int] = []
    errors: list[str] = []

    async def handler(value: int) -> None:
        if value == 1:
            raise ValueError("bad message")
        processed.append(value)

    async def on_error(exc: Exception) -> None:
        errors.append(str(exc))

    runtime = EntryRuntime(queue_size=2, handler=handler, on_error=on_error)
    await runtime.async_start()
    runtime.enqueue(1)
    runtime.enqueue(2)
    await runtime.async_join()
    await runtime.async_stop()
    assert errors == ["bad message"]
    assert processed == [2]


async def test_runtime_stop_cancels_a_stuck_handler() -> None:
    gate = asyncio.Event()

    async def handler(value: int) -> None:
        await gate.wait()

    runtime = EntryRuntime(queue_size=1, handler=handler, stop_timeout=0.01)
    await runtime.async_start()
    runtime.enqueue(1)
    await asyncio.sleep(0)
    await runtime.async_stop()
    assert not runtime.running


def _entry(data: dict[str, object]) -> MockConfigEntry:
    return MockConfigEntry(
        domain="frigate_vision",
        title="Front Door",
        data=data,
        options={"processing_mode": "observe"},
        entry_id="entry_1",
    )


def _frigate_data() -> dict[str, object]:
    return {
        "base_url": "http://frigate.local",
        "auth_mode": "none",
        "mqtt_topic_prefix": "frigate",
        "camera": "front",
    }


def _door_data() -> dict[str, object]:
    return {
        "event_entity_id": "event.front_lock",
        "action_attribute": "action",
        "open_values": ["open"],
        "close_values": ["close"],
        "side_attribute": "side",
        "inside_values": ["inside"],
        "outside_values": ["outside"],
    }


async def test_runtime_rebinds_the_unique_collecting_cycle(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.COLLECTING,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=100,
        camera="front",
    )

    async def load(store: ActivityStore) -> None:
        store._activities = {record.activity_id: record}

    async def recover(store: ActivityStore, now: float) -> list[ActivityRecord]:
        return [record]

    client = SimpleNamespace(async_close=AsyncMock())
    monkeypatch.setattr(ActivityStore, "async_load", load)
    monkeypatch.setattr(ActivityStore, "async_recover", recover)
    monkeypatch.setattr(
        runtime_module.FrigateClient,
        "async_create",
        AsyncMock(return_value=client),
    )
    monkeypatch.setattr(
        runtime_module, "async_subscribe_lock", Mock(return_value=Mock())
    )
    monkeypatch.setattr(
        runtime_module, "async_subscribe_frigate", AsyncMock(return_value=Mock())
    )
    door_data = _door_data()
    door_data["contact_entity_id"] = "binary_sensor.front_contact"
    hass.states.async_set("binary_sensor.front_contact", "on")
    entry = _entry({"frigate": _frigate_data(), "door": door_data, "zones": {}})
    entry.add_to_hass(hass)
    runtime = await IntegrationRuntime.async_create(hass, entry, 10)
    assert runtime.door_coordinator is not None
    result = await runtime.door_coordinator.async_open(110, "inside")
    assert result.status == "duplicate_open"
    assert result.record.activity_id == record.activity_id
    closed = await runtime.door_coordinator.async_close(120)
    assert closed.record.door_remained_open is True
    await runtime.async_stop()


async def test_runtime_rejects_multiple_recovered_collecting_cycles(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.COLLECTING,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=100,
        camera="front",
    )
    second = ActivityRecord(
        activity_id="door_entry_1_101000",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.COLLECTING,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=101,
        updated_at=101,
        camera="front",
    )

    async def load(store: ActivityStore) -> None:
        store._activities = {first.activity_id: first, second.activity_id: second}

    async def recover(store: ActivityStore, now: float) -> list[ActivityRecord]:
        return [first, second]

    monkeypatch.setattr(ActivityStore, "async_load", load)
    monkeypatch.setattr(ActivityStore, "async_recover", recover)
    entry = _entry({"frigate": _frigate_data(), "door": _door_data(), "zones": {}})
    entry.add_to_hass(hass)
    with pytest.raises(RuntimeError, match="multiple_collecting_cycles"):
        await IntegrationRuntime.async_create(hass, entry, 10)


async def test_setup_failure_rolls_back_every_acquired_resource(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    unsubscribe_lock = Mock()
    client = SimpleNamespace(async_close=AsyncMock())
    stop = AsyncMock()
    monkeypatch.setattr(EntryRuntime, "async_start", AsyncMock())
    monkeypatch.setattr(EntryRuntime, "async_stop", stop)
    monkeypatch.setattr(
        runtime_module, "async_subscribe_lock", Mock(return_value=unsubscribe_lock)
    )
    monkeypatch.setattr(
        runtime_module.FrigateClient,
        "async_create",
        AsyncMock(return_value=client),
    )
    monkeypatch.setattr(
        runtime_module,
        "async_subscribe_frigate",
        AsyncMock(side_effect=RuntimeError("mqtt_subscribe_failed")),
    )
    entry = _entry({"frigate": _frigate_data(), "door": _door_data(), "zones": {}})
    entry.add_to_hass(hass)
    with pytest.raises(RuntimeError, match="mqtt_subscribe_failed"):
        await IntegrationRuntime.async_create(hass, entry, 10)
    unsubscribe_lock.assert_called_once_with()
    stop.assert_awaited_once_with()
    client.async_close.assert_awaited_once_with()


async def test_runtime_settles_persisted_review_after_restart(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle_after = time.time() + 0.5
    message = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_1",
        review_id="review_1",
        occurred_at=settle_after - 1,
        camera="front",
        detection_ids=("event_1",),
    )
    buffered = BufferedIngress(message=message, settle_after=settle_after)
    first_event = BufferedIngress(
        message=IngressMessage(
            kind=IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="event_2",
            event_id="event_2",
            event_type="new",
            occurred_at=settle_after - 2,
            camera="front",
        ),
        settle_after=settle_after,
    )
    second_event = BufferedIngress(
        message=replace(
            first_event.message,
            event_type="update",
            occurred_at=settle_after - 1.5,
        ),
        settle_after=settle_after + 0.5,
    )

    async def load(store: ActivityStore) -> None:
        store._ingress_buffer = {
            item.buffer_id: item for item in (buffered, first_event, second_event)
        }

    async def recover(store: ActivityStore, now: float) -> list[ActivityRecord]:
        return []

    client = SimpleNamespace(async_close=AsyncMock())
    monkeypatch.setattr(ActivityStore, "async_load", load)
    monkeypatch.setattr(ActivityStore, "async_recover", recover)
    monkeypatch.setattr(runtime_module, "MediaManager", _NoopMediaManager)
    monkeypatch.setattr(
        runtime_module.FrigateClient,
        "async_create",
        AsyncMock(return_value=client),
    )
    monkeypatch.setattr(
        runtime_module, "async_subscribe_frigate", AsyncMock(return_value=Mock())
    )
    entry = _entry({"frigate": _frigate_data(), "zones": {}})
    entry.add_to_hass(hass)
    runtime = await IntegrationRuntime.async_create(hass, entry, 10)
    tasks = tuple((runtime.settlement_tasks or {}).values())
    # One task per distinct (entry_id, source_id, kind). Three messages are
    # buffered but they collapse to two tasks, because both events share
    # source_id "event_2" -- the lookup keys on the source, not on the message.
    assert len(runtime.store.buffered_ingress()) == 3
    assert len(tasks) == 2
    await asyncio.gather(*tasks)
    assert runtime.store.get("review_entry_1_front_review_1") is not None

    # The second event's deadline (settle_after + 0.5) is later than the
    # review's, so it is still buffered when the review settles. Event messages
    # are discarded once their own deadline passes, so draining again clears it.
    correlation = runtime.correlation
    assert correlation is not None
    await correlation.async_settle_due(settle_after + 1)
    assert runtime.store.buffered_ingress() == ()
    await runtime.async_stop()


async def test_runtime_retries_safe_buffer_settlement_failure(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    settle_after = time.time() + 0.2
    message = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_retry",
        review_id="review_retry",
        occurred_at=settle_after - 1,
        camera="front",
        detection_ids=("event_retry",),
    )
    buffered = BufferedIngress(message=message, settle_after=settle_after)

    async def load(store: ActivityStore) -> None:
        store._ingress_buffer = {buffered.buffer_id: buffered}

    async def recover(store: ActivityStore, now: float) -> list[ActivityRecord]:
        return []

    original_settle = runtime_module.CorrelationEngine.async_settle_due
    calls = 0

    async def flaky_settle(
        engine: runtime_module.CorrelationEngine, now: float
    ) -> tuple[ActivityRecord, ...]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary store failure")
        return await original_settle(engine, now)

    client = SimpleNamespace(async_close=AsyncMock())
    monkeypatch.setattr(ActivityStore, "async_load", load)
    monkeypatch.setattr(ActivityStore, "async_recover", recover)
    monkeypatch.setattr(runtime_module, "MediaManager", _NoopMediaManager)
    monkeypatch.setattr(
        runtime_module.CorrelationEngine, "async_settle_due", flaky_settle
    )
    monkeypatch.setattr(runtime_module, "SETTLEMENT_RETRY_SECONDS", 0)
    monkeypatch.setattr(
        runtime_module.FrigateClient,
        "async_create",
        AsyncMock(return_value=client),
    )
    monkeypatch.setattr(
        runtime_module, "async_subscribe_frigate", AsyncMock(return_value=Mock())
    )
    entry = _entry({"frigate": _frigate_data(), "zones": {}})
    entry.add_to_hass(hass)
    runtime = await IntegrationRuntime.async_create(hass, entry, 10)
    task = next(iter((runtime.settlement_tasks or {}).values()))
    await task
    assert calls == 2
    assert runtime.store.get("review_entry_1_front_review_retry") is not None
    await runtime.async_stop()


def test_recovery_work_covers_every_safe_stage_without_resetting_deadlines() -> None:
    base = ActivityRecord(
        activity_id="activity_collecting",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.COLLECTING,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=1,
        updated_at=1,
        camera="front",
    )
    records = (
        base,
        replace(
            base,
            activity_id="activity_sealed",
            stage=ActivityStage.SEALED,
            association_deadline=10,
            finalization_deadline=20,
        ),
        replace(
            base,
            activity_id="activity_evidence",
            stage=ActivityStage.EVIDENCE_READY,
        ),
        replace(
            base,
            activity_id="activity_shadow",
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.SHADOW,
        ),
        replace(
            base,
            activity_id="activity_live",
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
        ),
    )
    work = build_recovery_work(records, now=15)
    assert [(item.activity_id, item.action, item.run_at) for item in work] == [
        ("activity_collecting", RecoveryAction.RESTORE_DOOR_CYCLE, 15),
        ("activity_sealed", RecoveryAction.FINALIZE_SEALED, 20),
        ("activity_evidence", RecoveryAction.COMPLETE_OBSERVE, 15),
        ("activity_shadow", RecoveryAction.COMPLETE_SHADOW, 15),
        ("activity_live", RecoveryAction.CONTINUE_DELIVERY, 15),
    ]


async def test_recovery_handoff_keeps_failed_work_until_consumer_succeeds(
    hass: HomeAssistant,
) -> None:
    work = runtime_module.RecoveryWork(
        "activity_1", RecoveryAction.CONTINUE_ANALYSIS, 10
    )

    async def queue_handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=ActivityStore(hass, "entry_1"),
        queue=EntryRuntime(queue_size=1, handler=queue_handler),
        recovery_work=(work,),
    )

    async def fail(item: runtime_module.RecoveryWork) -> None:
        raise OSError("consumer failed")

    with pytest.raises(OSError, match="consumer failed"):
        await runtime.async_process_recovery(fail)
    assert runtime.recovery_work == (work,)

    consumed: list[runtime_module.RecoveryWork] = []

    async def consume(item: runtime_module.RecoveryWork) -> None:
        consumed.append(item)

    await runtime.async_process_recovery(consume)
    assert consumed == [work]
    assert runtime.recovery_work == ()


async def test_runtime_consumes_recovered_sealed_with_media_manager(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = time.time()
    record = ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=now - 1,
    )

    async def load(store: ActivityStore) -> None:
        store._activities = {record.activity_id: record}

    async def recover(store: ActivityStore, current: float) -> list[ActivityRecord]:
        return [record]

    built: list[str] = []

    class FakeManager:
        def __init__(self, *args, **kwargs):
            pass

        async def async_restore_registry(self):
            return None

        async def async_cleanup(self, **kwargs):
            return ()

        async def async_build(self, activity_id: str):
            built.append(activity_id)
            return record

    client = SimpleNamespace(async_close=AsyncMock())
    monkeypatch.setattr(ActivityStore, "async_load", load)
    monkeypatch.setattr(ActivityStore, "async_recover", recover)
    monkeypatch.setattr(runtime_module, "MediaManager", FakeManager)
    monkeypatch.setattr(
        runtime_module.FrigateClient, "async_create", AsyncMock(return_value=client)
    )
    monkeypatch.setattr(
        runtime_module, "async_subscribe_frigate", AsyncMock(return_value=Mock())
    )
    entry = _entry(
        {
            "frigate": _frigate_data(),
            "zones": {"near": ["near"], "transition": ["mid"], "far": ["far"]},
        }
    )
    entry.add_to_hass(hass)
    runtime = await IntegrationRuntime.async_create(hass, entry, 10)
    await asyncio.gather(*(runtime.media_tasks or {}).values())
    assert built == [record.activity_id]
    await runtime.async_stop()


async def test_media_failure_during_stop_keeps_activity_sealed(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_1",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=1,
        updated_at=2,
        camera="front",
        finalization_deadline=2,
    )
    await store.async_create(record)
    started = asyncio.Event()
    release = asyncio.Event()

    class Manager:
        async def async_build(self, activity_id):
            started.set()
            await release.wait()
            raise runtime_module.MediaError("recording_gap")

    async def handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        media_manager=Manager(),
        media_tasks={},
    )
    entry = _entry({})
    entry.add_to_hass(hass)
    runtime._schedule_media(entry, record)
    await started.wait()
    runtime.stopping = True
    release.set()
    await asyncio.gather(*(runtime.media_tasks or {}).values())
    assert store.get("activity_1").stage is ActivityStage.SEALED  # type: ignore[union-attr]


async def test_a_provider_502_abandons_the_activity_without_retrying(
    hass: HomeAssistant,
) -> None:
    """A 502 from the provider discards the activity instead of trying again.

    This is the behaviour that decides which vision model is safe to deploy, so it
    is pinned here rather than left to be re-derived.

    Measured on this deployment's proxy: the slowest models time out upstream and
    return `provider_http_502` on 15-50% of calls while the fastest returns none.
    The retry path covers only `SAFE_ANALYSIS_PRECHECK_ERRORS`
    (`vision_not_configured`, `provider_unavailable`), so a 502 is recorded and the
    loop returns -- the activity is never analysed and nothing ever tells the user
    it was skipped.

    That makes a model's failure rate as important as its accuracy: a model that is
    70% accurate when it answers, but answers 65% of the time, delivers less than
    one that is 40% accurate and always answers.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_http_502",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.EVIDENCE_READY,
        # LIVE, not OBSERVE: an observe-mode activity short-circuits before the
        # provider is ever called, so it could not test the retry policy at all.
        processing_mode=ProcessingMode.LIVE,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=time.time() + 60,
    )
    await store.async_create(record)

    class Vision:
        def __init__(self) -> None:
            self.calls = 0

        async def async_analyze(self, activity_id: str):
            self.calls += 1
            raise runtime_module.VisionError("provider_http_502")

    vision = Vision()

    async def handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        # analysis_tasks defaults to None, and `_schedule_analysis` returns at once
        # when it is None (production sets `{}` at construction). Passing it here
        # is what makes the analysis path reachable at all.
        analysis_tasks={},
    )
    runtime.vision = vision  # type: ignore[assignment]
    entry = _entry({})
    entry.add_to_hass(hass)

    runtime._schedule_analysis(entry, record)
    await asyncio.gather(*(runtime.analysis_tasks or {}).values())

    assert vision.calls == 1, (
        "a 502 must not be retried -- if this became >1 the retry policy changed "
        "and the model comparison's conclusions need revisiting"
    )
    assert runtime.last_error is not None
    assert "502" in str(runtime.last_error)
    # The activity is left where it was: sealed evidence, no classification. It
    # will not be picked up again, which is why the failure rate matters.
    assert store.get(record.activity_id).classification is None  # type: ignore[union-attr]


async def test_runtime_loads_without_door_configuration(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def load(store: ActivityStore) -> None:
        return None

    async def recover(store: ActivityStore, now: float) -> list[ActivityRecord]:
        return []

    client = SimpleNamespace(async_close=AsyncMock())
    monkeypatch.setattr(ActivityStore, "async_load", load)
    monkeypatch.setattr(ActivityStore, "async_recover", recover)
    monkeypatch.setattr(runtime_module, "MediaManager", _NoopMediaManager)
    monkeypatch.setattr(
        runtime_module.FrigateClient,
        "async_create",
        AsyncMock(return_value=client),
    )
    monkeypatch.setattr(
        runtime_module, "async_subscribe_frigate", AsyncMock(return_value=Mock())
    )
    entry = _entry({"frigate": _frigate_data(), "zones": {}})
    entry.add_to_hass(hass)
    runtime = await IntegrationRuntime.async_create(hass, entry, 10)
    assert runtime.door_coordinator is None
    assert runtime.frigate_client is client
    assert runtime.correlation is not None
    await runtime.async_stop()
