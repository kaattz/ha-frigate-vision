from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.frigate_vision.runtime as runtime_module
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    BufferedIngress,
    IngressKind,
    IngressMessage,
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
        options={},
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


async def test_setup_failure_rolls_back_every_acquired_resource(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    client = SimpleNamespace(async_close=AsyncMock())
    stop = AsyncMock()
    monkeypatch.setattr(EntryRuntime, "async_start", AsyncMock())
    monkeypatch.setattr(EntryRuntime, "async_stop", stop)
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
    entry = _entry({"frigate": _frigate_data(), "zones": {}})
    entry.add_to_hass(hass)
    with pytest.raises(RuntimeError, match="mqtt_subscribe_failed"):
        await IntegrationRuntime.async_create(hass, entry, 10)
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
    """Every non-terminal stage maps to exactly the work that finishes it.

    The mapping changed when the door cycle and the processing modes were removed:
    `COLLECTING` no longer has a restore action (nothing produces that stage any
    more) and `EVIDENCE_READY` / `ANALYSIS_DONE` each have a single successor
    rather than one per mode. The deadlines still travel untouched, which is what
    keeps a restart from extending an activity's own finalization window.
    """
    base = ActivityRecord(
        activity_id="activity_sealed",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=1,
        updated_at=1,
        camera="front",
    )
    records = (
        replace(
            base,
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
            activity_id="activity_done",
            stage=ActivityStage.ANALYSIS_DONE,
        ),
        # A stage with no downstream work must be skipped, not guessed at.
        replace(
            base,
            activity_id="activity_terminal",
            stage=ActivityStage.COMPLETED,
        ),
    )
    work = build_recovery_work(records, now=15)
    assert [(item.activity_id, item.action, item.run_at) for item in work] == [
        ("activity_sealed", RecoveryAction.FINALIZE_SEALED, 20),
        ("activity_evidence", RecoveryAction.CONTINUE_ANALYSIS, 15),
        ("activity_done", RecoveryAction.CONTINUE_DELIVERY, 15),
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
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
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
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
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


async def test_a_frigate_404_does_not_raise_the_connection_card(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frigate answering 404 is not Frigate being unreachable.

    Measured on this deployment: the evidence plan derives its frames from the
    *detections*, but the build is scheduled at the *review's* end. A detection
    can outlive its review, so `last` and `postroll` are requested up to ~8s
    before the moments they depict are recorded, and Frigate answers 404 for an
    instant it has no segment for yet -- true of 11 of 11 real reviews measured,
    gaps 5.4-8.0s.

    That is a recording gap on a Frigate that is up and answering. Raising
    `frigate_unavailable` sent the user to check a service that was never down,
    which is what "无法连接 Frigate" told them.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_404",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=time.time() - 1,
    )
    await store.async_create(record)

    calls: list[str] = []

    class Manager:
        async def async_build(self, activity_id: str):
            calls.append(activity_id)
            raise runtime_module.FrigateApiError("http_404")

    async def handler(message: object) -> None:
        return None

    entry = _entry({})
    entry.add_to_hass(hass)
    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        media_manager=Manager(),
        media_tasks={},
        entry_id=entry.entry_id,
    )
    # One attempt: this test is about the classification, not the retry budget.
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_SECONDS", 0)
    runtime._schedule_media(entry, record)
    await asyncio.gather(*(runtime.media_tasks or {}).values())

    registry = ir.async_get(hass)
    assert (
        "frigate_vision",
        f"{entry.entry_id}_frigate_unavailable",
    ) not in registry.issues, (
        "Frigate 回答了 404 说明它可达；报「无法连接 Frigate」把录像缺口说成了连接故障"
    )
    # The failure is still recorded, and still visible: an activity was lost.
    assert runtime.last_error == "http_404"
    assert calls == ["activity_404"]
    failed = store.get(record.activity_id)
    assert failed is not None and failed.stage is ActivityStage.FAILED
    assert failed.error_code == "media_retry_exhausted"


async def test_a_frigate_404_is_still_retried_as_a_transient_gap(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing frame is usually not-yet-written, so it must still be retried.

    This is the other half of the fix: the code is no longer *reported* as a
    connection failure, but it must not stop being treated as transient either.
    Frigate closes recording segments every 10s, so an instant at the live edge
    becomes servable moments later -- the same retry that papers over the gap
    today has to stay.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_404_retry",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=time.time() - 1,
    )
    await store.async_create(record)

    calls: list[str] = []

    class Manager:
        async def async_build(self, activity_id: str):
            calls.append(activity_id)
            raise runtime_module.FrigateApiError("http_404")

    async def handler(message: object) -> None:
        return None

    entry = _entry({})
    entry.add_to_hass(hass)
    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        media_manager=Manager(),
        media_tasks={},
        entry_id=entry.entry_id,
    )
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_ATTEMPTS", 3)
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_SECONDS", 0)
    runtime._schedule_media(entry, record)
    await asyncio.gather(*(runtime.media_tasks or {}).values())

    assert len(calls) == 3, "一次 404 就放弃会丢掉本来只是还没写完的帧"


async def test_a_genuine_connection_failure_still_raises_the_card(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fix must not over-correct: a real outage keeps its own repair.

    `frigate_unavailable` means the client could not reach Frigate at all. That
    card is correct and must survive, or the repair added for a real outage
    would have been replaced by silence.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_offline",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=time.time() - 1,
    )
    await store.async_create(record)

    class Manager:
        async def async_build(self, activity_id: str):
            raise runtime_module.FrigateApiError("frigate_unavailable")

    async def handler(message: object) -> None:
        return None

    entry = _entry({})
    entry.add_to_hass(hass)
    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        media_manager=Manager(),
        media_tasks={},
        entry_id=entry.entry_id,
    )
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_SECONDS", 0)
    runtime._schedule_media(entry, record)
    await asyncio.gather(*(runtime.media_tasks or {}).values())

    registry = ir.async_get(hass)
    assert (
        "frigate_vision",
        f"{entry.entry_id}_frigate_unavailable",
    ) in registry.issues, "真正连不上 Frigate 时必须保留这张修复卡片"


async def test_the_runtime_loop_does_not_retry_a_spent_provider_failure(
    hass: HomeAssistant,
) -> None:
    """The retry for a 5xx belongs to the HTTP layer, not to this loop.

    This test previously pinned the opposite conclusion -- "a 502 abandons the
    activity, and nothing ever tells the user it was skipped" -- which was a
    description of the bug rather than a requirement. The retry now lives in
    `async_request_with_retry`, and it has to: it must run while the analysis
    claim is still held. By the time an error reaches this loop the claim has
    been moved to FAILED, so a second `VisionClient` call here would be refused
    by the store as `side_effect_stage_conflict` rather than actually re-run.

    So the assertion that survives is the division of responsibility -- the loop
    abandons the activity after one `VisionClient` call -- and what is added is
    the part that was missing: the failure is now *visible*. It raises a repair,
    because a memory-only sensor that already held the same value left one lost
    activity indistinguishable from a quiet night.

    Separately, a model's failure rate still matters: the retry cap means a
    provider that answers 5xx often enough still loses activities, just far fewer
    of them.
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
            # What `VisionClient` raises once the HTTP layer has spent its
            # budget: the loop never sees the individual attempts.
            raise runtime_module.VisionError("provider_http_502")

    vision = Vision()

    async def handler(message: object) -> None:
        return None

    entry = _entry({})
    entry.add_to_hass(hass)
    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        # analysis_tasks defaults to None, and `_schedule_analysis` returns at once
        # when it is None (production sets `{}` at construction). Passing it here
        # is what makes the analysis path reachable at all.
        analysis_tasks={},
        entry_id=entry.entry_id,
    )
    runtime.vision = vision  # type: ignore[assignment]

    runtime._schedule_analysis(entry, record)
    await asyncio.gather(*(runtime.analysis_tasks or {}).values())

    assert vision.calls == 1, (
        "the loop must not re-run an analysis the HTTP layer already retried -- "
        "if this became >1 the retry has been duplicated, and the second call "
        "would be refused by the store rather than executed"
    )
    assert runtime.last_error == "provider_http_502"
    # The activity is left where it was: sealed evidence, no classification.
    assert store.get(record.activity_id).classification is None  # type: ignore[union-attr]

    registry = ir.async_get(hass)
    assert (
        "frigate_vision",
        f"{entry.entry_id}_provider_error",
    ) in registry.issues, "a lost activity must raise a repair, not pass silently"


async def test_a_store_conflict_does_not_raise_the_provider_alert(
    hass: HomeAssistant,
) -> None:
    """库存冲突不是视觉 provider 的问题，绝不能报 provider_error。

    `record_error` 的输入域是 worker 兜底路径喂进来的**任意**异常字符串
    （`runtime.py:250`），远宽于分析路径。一个 `StoreConflictError` 走到这里
    时，provider 根本没被调用过 —— 给用户报「视觉模型在报错」会把他送去查
    LLM，而真正坏掉的是库存。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()

    async def handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        entry_id="entry_1",
    )

    runtime.record_error("terminal_activity")

    registry = ir.async_get(hass)
    assert (
        "frigate_vision",
        "entry_1_provider_error",
    ) not in registry.issues, "库存冲突被误报成了视觉 provider 的故障"
    assert runtime.last_error == "terminal_activity"


async def test_a_frigate_failure_keeps_its_own_card_and_is_not_erased(
    hass: HomeAssistant,
) -> None:
    """clear_error(PROVIDER_ERROR) 不能抹掉 live 的 Frigate 故障。

    `frigate_unavailable` 有自己的修复项，所以 `record_error` 让它走「自己的
    卡片」那条分支并提前返回 —— 它**不**是 provider 侧的失败。而
    `clear_error(PROVIDER_ERROR)` 会顺带清 `last_error`，前提是那个码是
    provider 的错。旧实现用「非本地错误」的补集判定，Frigate 掉线恰好落在
    补集里，于是下一次分析成功就把一场仍在持续的 Frigate 故障从
    `sensor.<name>_last_error` 上抹掉了，而它的修复卡片还挂着。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()

    async def handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        entry_id="entry_1",
    )
    runtime.record_error("frigate_unavailable")
    assert runtime.last_error == "frigate_unavailable"
    registry = ir.async_get(hass)
    assert ("frigate_vision", "entry_1_frigate_unavailable") in registry.issues

    runtime.clear_error("provider_error")

    assert runtime.last_error == "frigate_unavailable", (
        "一场仍在持续的 Frigate 故障被一次成功的分析抹掉了"
    )
    assert (
        "frigate_vision",
        "entry_1_frigate_unavailable",
    ) in registry.issues, "Frigate 的修复卡片不该被 provider_error 收走"


async def test_runtime_loads_with_review_only_configuration(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """没有门锁配置时，运行时以 review-only 形态启动。

    门周期删除后这就是唯一的形态，但这个断言仍然值得保留：配置里残留的
    `door` 键（旧 entry 升级上来）必须被无视而不是让启动失败。
    """
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
    # A legacy entry still carrying its old door mapping must not fail setup.
    entry = _entry(
        {"frigate": _frigate_data(), "door": _door_data_legacy(), "zones": {}}
    )
    entry.add_to_hass(hass)
    runtime = await IntegrationRuntime.async_create(hass, entry, 10)
    assert runtime.frigate_client is client
    assert runtime.correlation is not None
    await runtime.async_stop()


async def test_a_media_failure_is_logged_so_it_can_be_diagnosed(
    hass: HomeAssistant,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A lost activity must leave a trace in the log, not only on a sensor.

    Diagnosed from a real deployment: two activities died in frame extraction and
    the only surviving evidence was `media_retry_exhausted` on the `Last error`
    sensor. `media.py` logs nothing at all and this failure branch logged nothing
    either, so the code named neither the operation that failed nor the exception
    that caused it -- an hour went into re-probing Frigate endpoints by hand, all
    of which turned out to answer 200.

    The log line is the difference between "something failed" and "this call
    failed with this exception".
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_logged",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=time.time() - 1,
    )
    await store.async_create(record)

    class Manager:
        async def async_build(self, activity_id: str):
            raise runtime_module.FrigateApiError("http_404")

    async def handler(message: object) -> None:
        return None

    entry = _entry({})
    entry.add_to_hass(hass)
    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        media_manager=Manager(),
        media_tasks={},
        entry_id=entry.entry_id,
    )
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_ATTEMPTS", 1)
    monkeypatch.setattr(runtime_module, "MEDIA_RETRY_SECONDS", 0)

    with caplog.at_level(logging.WARNING, logger="custom_components.frigate_vision"):
        runtime._schedule_media(entry, record)
        await asyncio.gather(*(runtime.media_tasks or {}).values())

    # Only our own logger's records count. The store writes its whole state at
    # DEBUG on every transition, which contains both the activity id and (via the
    # record) the closing error code -- matching against the raw capture would
    # pass without a single line of our own being emitted.
    ours = [
        record_
        for record_ in caplog.records
        if record_.name.startswith("custom_components.frigate_vision")
    ]
    assert ours, (
        "活动因取帧失败而丢失，却没有任何日志：只有传感器上的一个错误码，"
        "无法判断是哪一步、哪个异常"
    )
    message = "\n".join(record_.getMessage() for record_ in ours)
    assert "activity_logged" in message, "日志必须点名是哪条活动失败了"
    assert "http_404" in message, "日志必须带上底层异常，否则仍然无法定位"


def _door_data_legacy() -> dict:
    """A stored door mapping from before the removal, kept verbatim."""
    return {
        "event_entity_id": "event.front_door_lock",
        "action_attribute": "action",
        "open_values": ["1"],
        "close_values": ["2"],
        "side_attribute": "side",
        "inside_values": ["inside"],
        "outside_values": ["outside"],
        "contact_entity_id": "binary_sensor.front_contact",
        "doorbell_event_entity_id": "event.front_door_doorbell",
    }
