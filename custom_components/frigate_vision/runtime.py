"""Per-entry runtime and bounded queue."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Coroutine
from contextlib import suppress
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar, cast

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import Event, EventStateChangedData, HomeAssistant
from homeassistant.helpers.event import async_track_state_change_event

from .correlation import CorrelationEngine, CorrelationError, ZoneRoles
from .delivery import DeliveryManager
from .door import (
    DoorCycleCoordinator,
    DoorCycleError,
    DoorMapping,
    async_subscribe_lock,
    contact_is_open,
    doorbell_message,
)
from .frigate import (
    FrigateApiError,
    FrigateClient,
    FrigatePayloadError,
    async_subscribe_frigate,
    parse_event_payload,
    parse_review_payload,
)
from .media import MediaError, MediaManager
from .media_source import async_default_media_root
from .models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    IngressMessage,
    ProcessingMode,
)
from .repairs import ISSUES, async_clear_issue, async_set_issue
from .store import ActivityStore
from .vision import (
    VisionClient,
    VisionError,
    vision_config_from,
    vision_is_configured,
)

MessageT = TypeVar("MessageT")
_STOP = object()
SETTLEMENT_RETRY_SECONDS = 5.0
MEDIA_RETRY_SECONDS = 5.0
MEDIA_RETRY_ATTEMPTS = 3
IN_PROGRESS_RETRY_SECONDS = 10.0
IN_PROGRESS_RETRY_ATTEMPTS = 18
DOOR_OPEN_WATCHDOG_SECONDS = 1800.0
DOOR_OPEN_WATCHDOG_INTERVAL_SECONDS = 60.0
CLEANUP_INTERVAL_SECONDS = 86400.0
ANALYSIS_PRECHECK_RETRY_SECONDS = 30.0
SAFE_ANALYSIS_PRECHECK_ERRORS = {
    "vision_not_configured",
    "provider_unavailable",
}
TRANSIENT_MEDIA_ERRORS = {
    "recording_gap",
    "frigate_unavailable",
    "request_timeout",
    "http_404",
    "event_in_progress",
}
_LOGGER = logging.getLogger(__name__)


class QueueFullError(RuntimeError):
    """The entry queue is full."""


class RecoveryAction(StrEnum):
    """Deterministic handoff for safe persisted stages."""

    RESTORE_DOOR_CYCLE = "restore_door_cycle"
    FINALIZE_SEALED = "finalize_sealed"
    COMPLETE_OBSERVE = "complete_observe"
    CONTINUE_ANALYSIS = "continue_analysis"
    COMPLETE_SHADOW = "complete_shadow"
    CONTINUE_DELIVERY = "continue_delivery"


@dataclass(frozen=True, slots=True)
class RecoveryWork:
    activity_id: str
    action: RecoveryAction
    run_at: float


def build_recovery_work(
    records: tuple[ActivityRecord, ...] | list[ActivityRecord], *, now: float
) -> tuple[RecoveryWork, ...]:
    """Map safe persisted stages to downstream work without executing effects."""
    work: list[RecoveryWork] = []
    for record in records:
        if record.stage is ActivityStage.COLLECTING:
            action = RecoveryAction.RESTORE_DOOR_CYCLE
            run_at = now
        elif record.stage is ActivityStage.SEALED:
            if record.finalization_deadline is None:
                raise RuntimeError("recovery_deadline_missing")
            action = RecoveryAction.FINALIZE_SEALED
            run_at = record.finalization_deadline
        elif record.stage is ActivityStage.EVIDENCE_READY:
            action = (
                RecoveryAction.COMPLETE_OBSERVE
                if record.processing_mode is ProcessingMode.OBSERVE
                else RecoveryAction.CONTINUE_ANALYSIS
            )
            run_at = now
        elif record.stage is ActivityStage.ANALYSIS_DONE:
            if record.processing_mode is ProcessingMode.SHADOW:
                action = RecoveryAction.COMPLETE_SHADOW
            elif record.processing_mode is ProcessingMode.LIVE:
                action = RecoveryAction.CONTINUE_DELIVERY
            else:
                raise RuntimeError("invalid_recovery_mode")
            run_at = now
        else:
            continue
        work.append(RecoveryWork(record.activity_id, action, run_at))
    return tuple(work)


class EntryRuntime[MessageT]:
    """Run one ordered worker for one config entry."""

    def __init__(
        self,
        *,
        queue_size: int,
        handler: Callable[[MessageT], Awaitable[None]],
        on_error: Callable[[Exception], Awaitable[None]] | None = None,
        stop_timeout: float = 5.0,
        task_factory: Callable[[Coroutine[object, object, None]], asyncio.Task[None]]
        | None = None,
    ) -> None:
        if queue_size < 1:
            raise ValueError("invalid_queue_size")
        self._queue: asyncio.Queue[MessageT | object] = asyncio.Queue(queue_size)
        self._handler = handler
        self._on_error = on_error
        self._stop_timeout = stop_timeout
        self._task_factory = task_factory
        self._accepting = True
        self._worker: asyncio.Task[None] | None = None

    async def async_start(self) -> None:
        if self._worker is not None:
            raise RuntimeError("runtime_already_started")
        coroutine = self._async_worker()
        self._worker = (
            self._task_factory(coroutine)
            if self._task_factory is not None
            else asyncio.create_task(coroutine)
        )

    @property
    def running(self) -> bool:
        return self._worker is not None and not self._worker.done()

    def enqueue(self, message: MessageT) -> None:
        if not self._accepting:
            raise RuntimeError("runtime_stopping")
        try:
            self._queue.put_nowait(message)
        except asyncio.QueueFull as exc:
            raise QueueFullError("queue_full") from exc

    async def async_join(self) -> None:
        await self._queue.join()

    async def async_stop(self) -> None:
        if self._worker is None:
            return
        self._accepting = False
        try:
            await asyncio.wait_for(self._queue.join(), timeout=self._stop_timeout)
        except TimeoutError:
            _LOGGER.error("Entry worker did not drain before unload; cancelling")
            self._worker.cancel()
            with suppress(asyncio.CancelledError):
                await self._worker
        else:
            await self._queue.put(_STOP)
            await self._worker
        self._worker = None

    async def _async_worker(self) -> None:
        while True:
            value = await self._queue.get()
            try:
                if value is _STOP:
                    return
                try:
                    await self._handler(cast(MessageT, value))
                except Exception as exc:  # noqa: BLE001
                    if self._on_error is None:
                        _LOGGER.exception("Entry worker rejected a message")
                    else:
                        await self._on_error(exc)
            finally:
                self._queue.task_done()


@dataclass(slots=True)
class IntegrationRuntime:
    """Resources owned by one loaded config entry."""

    hass: HomeAssistant
    store: ActivityStore
    queue: EntryRuntime[object]
    last_error: str | None = None
    door_coordinator: DoorCycleCoordinator | None = None
    unsubscribe_callbacks: list[Callable[[], None]] | None = None
    correlation: CorrelationEngine | None = None
    frigate_client: FrigateClient | None = None
    contact_open: bool | None = None
    settlement_tasks: dict[str, asyncio.Task[None]] | None = None
    recovery_work: tuple[RecoveryWork, ...] = ()
    media_manager: MediaManager | None = None
    media_tasks: dict[str, asyncio.Task[None]] | None = None
    cleanup_task: asyncio.Task[None] | None = None
    door_watchdog_task: asyncio.Task[None] | None = None
    stopping: bool = False
    vision: VisionClient | None = None
    analysis_tasks: dict[str, asyncio.Task[None]] | None = None
    allowed_zones: frozenset[str] = frozenset()
    delivery: DeliveryManager | None = None
    entry_id: str = ""

    @classmethod
    async def async_create(
        cls, hass: HomeAssistant, entry: ConfigEntry, queue_size: int
    ) -> IntegrationRuntime:
        store = ActivityStore(hass, entry.entry_id)
        await store.async_load()
        recovery_now = time.time()
        recovered = await store.async_recover(recovery_now)
        recovery_work = build_recovery_work(recovered, now=recovery_now)
        runtime: IntegrationRuntime

        correlation: CorrelationEngine | None = None
        frigate_data = entry.data.get("frigate")
        active_id: str | None = None
        if isinstance(frigate_data, dict) and frigate_data:
            collecting = [
                record
                for record in recovered
                if record.entry_id == entry.entry_id
                and record.camera == frigate_data["camera"]
                and record.source is ActivitySource.DOOR_CYCLE
                and record.stage is ActivityStage.COLLECTING
            ]
            if len(collecting) > 1:
                raise RuntimeError("multiple_collecting_cycles")
            if collecting:
                active_id = collecting[0].activity_id
            correlation = CorrelationEngine(
                store,
                entry_id=entry.entry_id,
                camera=frigate_data["camera"],
                processing_mode=ProcessingMode(
                    entry.options.get("processing_mode", "observe")
                ),
                min_review_seconds=float(entry.options.get("min_review_seconds", 0)),
            )

        async def handle_message(message: object) -> None:
            if correlation is None or not isinstance(message, IngressMessage):
                raise CorrelationError("ingress_handler_not_configured")
            result = await correlation.async_handle(message)
            if result is None:
                runtime._schedule_ingress_settlement(entry, message)
            elif result.stage is ActivityStage.SEALED:
                runtime._schedule_media(entry, result)

        async def handle_error(exc: Exception) -> None:
            runtime.record_error(str(exc))

        queue = EntryRuntime(
            queue_size=queue_size,
            handler=handle_message,
            on_error=handle_error,
            task_factory=lambda coroutine: entry.async_create_background_task(
                hass,
                coroutine,
                f"{entry.domain}-{entry.entry_id}-worker",
            ),
        )
        runtime = cls(
            hass=hass,
            entry_id=entry.entry_id,
            store=store,
            queue=queue,
            correlation=correlation,
            settlement_tasks={},
            recovery_work=recovery_work,
            media_tasks={},
            analysis_tasks={},
        )
        runtime.delivery = DeliveryManager(hass, entry, store)
        # Provider settings may have been entered through the Options page
        # rather than the initial flow, so both are considered; testing
        # `entry.data` alone left a UI-configured entry with no client.
        if vision_is_configured(entry.data, entry.options):
            runtime.vision = VisionClient(
                hass, store, vision_config_from(entry.data, dict(entry.options))
            )
        runtime.unsubscribe_callbacks = []
        for recovered_record in recovered:
            if recovered_record.error_code in ISSUES:
                async_set_issue(hass, entry.entry_id, recovered_record.error_code)
        try:
            await queue.async_start()
            door_data = entry.data.get("door")
            if (
                isinstance(door_data, dict)
                and door_data
                and isinstance(frigate_data, dict)
            ):
                mapping = DoorMapping(
                    action_attribute=door_data["action_attribute"],
                    open_values=frozenset(door_data["open_values"]),
                    close_values=frozenset(door_data["close_values"]),
                    side_attribute=door_data["side_attribute"],
                    inside_values=frozenset(door_data["inside_values"]),
                    outside_values=frozenset(door_data["outside_values"]),
                )
                coordinator = DoorCycleCoordinator(
                    store,
                    entry_id=entry.entry_id,
                    camera=frigate_data["camera"],
                    processing_mode=ProcessingMode(
                        entry.options.get("processing_mode", "observe")
                    ),
                    active_id=active_id,
                )
                runtime.door_coordinator = coordinator
                runtime._start_door_watchdog(entry)

                async def handle_door(
                    action: str, side: str, occurred_at: float
                ) -> None:
                    try:
                        if action == "open":
                            result = await coordinator.async_open(occurred_at, side)
                            if runtime.contact_open is True:
                                await coordinator.async_contact(True, occurred_at)
                            if correlation is not None:
                                await correlation.async_replay_for_activity(
                                    result.record.activity_id
                                )
                        else:
                            result = await coordinator.async_close(occurred_at)
                            runtime._schedule_media(entry, result.record)
                        runtime.clear_error("door_mapping_invalid")
                        runtime.clear_error("door_open_too_long")
                    except (DoorCycleError, CorrelationError) as exc:
                        runtime.last_error = str(exc)

                runtime.unsubscribe_callbacks.append(
                    async_subscribe_lock(
                        hass,
                        door_data["event_entity_id"],
                        mapping,
                        handle_door,
                        lambda exc: runtime.record_error("door_mapping_invalid"),
                    )
                )
                doorbell_entity = door_data.get("doorbell_event_entity_id")
                if doorbell_entity:

                    async def handle_doorbell(
                        event: Event[EventStateChangedData],
                    ) -> None:
                        occurred_at = event.time_fired.timestamp()
                        attached = await coordinator.async_doorbell(occurred_at)
                        if attached is None:
                            queue.enqueue(
                                doorbell_message(
                                    entry.entry_id,
                                    str(doorbell_entity).replace(".", "_"),
                                    occurred_at,
                                    frigate_data["camera"],
                                )
                            )

                    runtime.unsubscribe_callbacks.append(
                        async_track_state_change_event(
                            hass, [doorbell_entity], handle_doorbell
                        )
                    )
                contact_entity = door_data.get("contact_entity_id")
                if contact_entity:
                    current_contact = hass.states.get(contact_entity)
                    if current_contact is not None:
                        runtime.contact_open = contact_is_open(current_contact.state)
                        if runtime.contact_open is None:
                            runtime.last_error = "door_contact_unavailable"
                        else:
                            if runtime.last_error == "door_contact_unavailable":
                                runtime.last_error = None
                            if active_id is not None:
                                await coordinator.async_contact(
                                    runtime.contact_open, time.time()
                                )

                    async def handle_contact(
                        event: Event[EventStateChangedData],
                    ) -> None:
                        new_state = event.data.get("new_state")
                        if new_state is not None:
                            runtime.contact_open = contact_is_open(new_state.state)
                            if runtime.contact_open is None:
                                runtime.last_error = "door_contact_unavailable"
                                return
                            if runtime.last_error == "door_contact_unavailable":
                                runtime.last_error = None
                            await coordinator.async_contact(
                                runtime.contact_open, event.time_fired.timestamp()
                            )

                    runtime.unsubscribe_callbacks.append(
                        async_track_state_change_event(
                            hass, [contact_entity], handle_contact
                        )
                    )
            zones_data = entry.data.get("zones")
            if isinstance(frigate_data, dict) and isinstance(zones_data, dict):
                client = await FrigateClient.async_create(hass, frigate_data)
                runtime.frigate_client = client
                allowed_zones = set().union(
                    zones_data.get("near", []),
                    zones_data.get("transition", []),
                    zones_data.get("far", []),
                )
                runtime.allowed_zones = frozenset(allowed_zones)
                analyze_all_person_reviews = bool(
                    entry.options.get("analyze_all_far_reviews", True)
                )

                def enqueue_event(payload: str) -> None:
                    try:
                        message = parse_event_payload(
                            payload,
                            entry_id=entry.entry_id,
                            camera=frigate_data["camera"],
                            allowed_zones=allowed_zones,
                        )
                        if message is not None:
                            runtime.clear_error("frigate_unavailable")
                            queue.enqueue(message)
                    except (FrigatePayloadError, RuntimeError) as exc:
                        runtime.last_error = str(exc)

                def enqueue_review(payload: str) -> None:
                    try:
                        message = parse_review_payload(
                            payload,
                            entry_id=entry.entry_id,
                            camera=frigate_data["camera"],
                            allowed_zones=allowed_zones,
                            analyze_all_person_reviews=(analyze_all_person_reviews),
                        )
                        if message is not None:
                            runtime.clear_error("frigate_unavailable")
                            queue.enqueue(message)
                    except (FrigatePayloadError, RuntimeError) as exc:
                        runtime.last_error = str(exc)

                runtime.unsubscribe_callbacks.append(
                    await async_subscribe_frigate(
                        hass,
                        frigate_data["mqtt_topic_prefix"],
                        enqueue_event,
                        enqueue_review,
                    )
                )
                roles = ZoneRoles(
                    near=frozenset(zones_data.get("near", [])),
                    transition=frozenset(zones_data.get("transition", [])),
                    far=frozenset(zones_data.get("far", [])),
                )
                media_root = await async_default_media_root(hass)
                runtime.media_manager = MediaManager(
                    hass, store, client, media_root, roles
                )
                await runtime.media_manager.async_restore_registry()
                await runtime.media_manager.async_cleanup(
                    retention_days=int(entry.options.get("media_retention_days", 7))
                )
                runtime.clear_error("media_cleanup_failed")
                runtime._start_periodic_cleanup(
                    entry,
                    int(entry.options.get("media_retention_days", 7)),
                )
            if correlation is not None:
                if active_id is not None:
                    await correlation.async_replay_for_activity(active_id)
                for buffered in store.buffered_ingress():
                    runtime._schedule_ingress_settlement(entry, buffered.message)
            if runtime.media_manager is not None:
                for record in store.all():
                    if record.stage is ActivityStage.SEALED:
                        runtime._schedule_media(entry, record)
                    elif record.stage is ActivityStage.EVIDENCE_READY:
                        runtime._schedule_analysis(entry, record)
                    elif (
                        record.stage is ActivityStage.ANALYSIS_DONE
                        and record.processing_mode is ProcessingMode.SHADOW
                    ):
                        await store.async_complete_mode(
                            record.activity_id, mode="shadow", updated_at=time.time()
                        )
                    elif (
                        record.stage is ActivityStage.ANALYSIS_DONE
                        and record.processing_mode is ProcessingMode.LIVE
                        and runtime.delivery is not None
                    ):
                        await runtime.delivery.async_start(record.activity_id)
            return runtime
        except Exception:
            await runtime.async_stop()
            raise

    @property
    def running(self) -> bool:
        return self.queue.running

    @property
    def is_healthy(self) -> bool:
        if not self.running or self.stopping:
            return False
        if self.correlation is not None and self.frigate_client is None:
            return False
        return True

    def _schedule_ingress_settlement(
        self, entry: ConfigEntry, message: IngressMessage
    ) -> None:
        if self.correlation is None or self.settlement_tasks is None:
            return
        if self.stopping:
            return
        buffered = next(
            (
                item
                for item in self.store.buffered_ingress()
                if item.message.entry_id == message.entry_id
                and item.message.source_id == message.source_id
                and item.message.kind is message.kind
            ),
            None,
        )
        if buffered is None or buffered.buffer_id in self.settlement_tasks:
            return
        correlation = self.correlation
        if correlation is None:
            return

        async def settle() -> None:
            delay = max(0.0, buffered.settle_after - time.time())
            if delay:
                await asyncio.sleep(delay)
            while any(
                item.buffer_id == buffered.buffer_id
                for item in self.store.buffered_ingress()
            ):
                try:
                    settled = await correlation.async_settle_due(time.time())
                    for record in settled:
                        if record.stage is ActivityStage.SEALED:
                            self._schedule_media(entry, record)
                except OSError as exc:
                    self.last_error = str(exc)
                    await asyncio.sleep(SETTLEMENT_RETRY_SECONDS)
                    continue
                except Exception:  # noqa: BLE001
                    self.record_error("media_cleanup_failed")
                    return
                return

        task = entry.async_create_background_task(
            self.hass,
            settle(),
            f"{entry.domain}-{entry.entry_id}-ingress-settlement",
        )
        self.settlement_tasks[buffered.buffer_id] = task
        task.add_done_callback(
            lambda _task: (
                self.settlement_tasks.pop(buffered.buffer_id, None)
                if self.settlement_tasks is not None
                else None
            )
        )

    def _schedule_media(self, entry: ConfigEntry, record: ActivityRecord) -> None:
        if self.media_manager is None or self.media_tasks is None:
            return
        if self.stopping:
            return
        if record.activity_id in self.media_tasks:
            return
        if record.finalization_deadline is None:
            self.last_error = "media_deadline_missing"
            return
        manager = self.media_manager
        run_at = record.finalization_deadline

        async def build() -> None:
            delay = max(0.0, run_at - time.time())
            if delay:
                await asyncio.sleep(delay)
            attempts = 0
            while True:
                try:
                    completed = await manager.async_build(record.activity_id)
                    self.clear_error("frigate_unavailable")
                    self._schedule_analysis(entry, completed)
                    return
                except (OSError, FrigateApiError, MediaError) as exc:
                    if self.stopping:
                        return
                    code = str(exc)
                    if code in {
                        "frigate_unavailable",
                        "request_timeout",
                        "http_404",
                    }:
                        self.record_error("frigate_unavailable")
                    attempts += 1
                    if code == "event_in_progress":
                        retry_attempts = IN_PROGRESS_RETRY_ATTEMPTS
                        retry_seconds = IN_PROGRESS_RETRY_SECONDS
                    else:
                        retry_attempts = MEDIA_RETRY_ATTEMPTS
                        retry_seconds = MEDIA_RETRY_SECONDS
                    if (
                        (isinstance(exc, OSError) or code in TRANSIENT_MEDIA_ERRORS)
                        and attempts < retry_attempts
                        and not self.stopping
                    ):
                        await asyncio.sleep(retry_seconds)
                        continue
                    self.last_error = code
                    current = self.store.get(record.activity_id)
                    if current is not None and current.stage is ActivityStage.SEALED:
                        error_code = (
                            "media_retry_exhausted"
                            if isinstance(exc, OSError)
                            or code in TRANSIENT_MEDIA_ERRORS
                            else code
                        )
                        while not self.stopping:
                            try:
                                await self.store.async_transition(
                                    record.activity_id,
                                    ActivityStage.SEALED,
                                    ActivityStage.FAILED,
                                    updated_at=time.time(),
                                    error_code=error_code,
                                )
                                break
                            except OSError as store_exc:
                                self.last_error = str(store_exc)
                                await asyncio.sleep(MEDIA_RETRY_SECONDS)
                            except Exception as store_exc:  # noqa: BLE001
                                self.last_error = str(store_exc)
                                break
                    return

        task = entry.async_create_background_task(
            self.hass,
            build(),
            f"{entry.domain}-{entry.entry_id}-{record.activity_id}-media",
        )
        self.media_tasks[record.activity_id] = task
        task.add_done_callback(
            lambda _task: (
                self.media_tasks.pop(record.activity_id, None)
                if self.media_tasks is not None
                else None
            )
        )

    def _start_door_watchdog(self, entry: ConfigEntry) -> None:
        coordinator = self.door_coordinator
        if coordinator is None or self.door_watchdog_task is not None:
            return

        async def watchdog_loop() -> None:
            while not self.stopping:
                await asyncio.sleep(DOOR_OPEN_WATCHDOG_INTERVAL_SECONDS)
                if self.stopping:
                    return
                try:
                    result = await coordinator.async_abandon_stale_open(
                        time.time(),
                        threshold=DOOR_OPEN_WATCHDOG_SECONDS,
                    )
                except Exception as exc:  # noqa: BLE001
                    self.last_error = str(exc)
                    continue
                if result is not None:
                    self.record_error("door_open_too_long")

        self.door_watchdog_task = entry.async_create_background_task(
            self.hass,
            watchdog_loop(),
            f"{entry.domain}-{entry.entry_id}-door-watchdog",
        )

    def _start_periodic_cleanup(self, entry: ConfigEntry, retention_days: int) -> None:
        if self.media_manager is None or self.cleanup_task is not None:
            return
        manager = self.media_manager

        async def cleanup_loop() -> None:
            while True:
                await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)
                try:
                    await manager.async_cleanup(retention_days=retention_days)
                    self.clear_error("media_cleanup_failed")
                except Exception:  # noqa: BLE001
                    self.record_error("media_cleanup_failed")

        self.cleanup_task = entry.async_create_background_task(
            self.hass,
            cleanup_loop(),
            f"{entry.domain}-{entry.entry_id}-media-cleanup",
        )

    def _schedule_analysis(self, entry: ConfigEntry, record: ActivityRecord) -> None:
        if self.stopping or self.analysis_tasks is None:
            return
        if record.activity_id in self.analysis_tasks:
            return

        async def analyze() -> None:
            if self.stopping:
                return
            if record.processing_mode is ProcessingMode.OBSERVE:
                try:
                    await self.store.async_complete_mode(
                        record.activity_id, mode="observe", updated_at=time.time()
                    )
                except Exception as exc:  # noqa: BLE001
                    self.last_error = str(exc)
                return
            while not self.stopping:
                if self.vision is None:
                    error = VisionError("vision_not_configured")
                else:
                    try:
                        analyzed = await self.vision.async_analyze(record.activity_id)
                        if analyzed.processing_mode is ProcessingMode.SHADOW:
                            await self.store.async_complete_mode(
                                analyzed.activity_id,
                                mode="shadow",
                                updated_at=time.time(),
                            )
                        elif self.delivery is not None:
                            await self.delivery.async_start(analyzed.activity_id)
                        self.clear_error("vision_not_configured")
                        return
                    except VisionError as exc:
                        error = exc
                    except Exception as exc:  # noqa: BLE001
                        # Recorded on the sensor as a bare string, which is all
                        # the UI needs -- but the traceback is the only way to
                        # tell an unexpected failure apart from a handled one.
                        _LOGGER.exception(
                            "Analysis failed unexpectedly for %s",
                            record.activity_id,
                        )
                        self.last_error = str(exc)
                        return
                self.record_error(str(error))
                current = self.store.get(record.activity_id)
                if (
                    str(error) in SAFE_ANALYSIS_PRECHECK_ERRORS
                    and current is not None
                    and current.stage is ActivityStage.EVIDENCE_READY
                ):
                    await asyncio.sleep(ANALYSIS_PRECHECK_RETRY_SECONDS)
                    continue
                return

        task = entry.async_create_background_task(
            self.hass,
            analyze(),
            f"{entry.domain}-{entry.entry_id}-{record.activity_id}-analysis",
        )
        self.analysis_tasks[record.activity_id] = task
        task.add_done_callback(
            lambda _task: (
                self.analysis_tasks.pop(record.activity_id, None)
                if self.analysis_tasks is not None
                else None
            )
        )

    def async_schedule_record(self, entry: ConfigEntry, record: ActivityRecord) -> None:
        if record.stage is ActivityStage.SEALED:
            self._schedule_media(entry, record)
        elif record.stage is ActivityStage.EVIDENCE_READY:
            self._schedule_analysis(entry, record)

    async def async_process_recovery(
        self, handler: Callable[[RecoveryWork], Awaitable[None]]
    ) -> None:
        """Hand safe recovery work to a downstream stage consumer."""
        while self.recovery_work:
            item = self.recovery_work[0]
            await handler(item)
            self.recovery_work = self.recovery_work[1:]

    def record_error(self, code: str) -> None:
        self.last_error = code
        if code in ISSUES and self.entry_id:
            async_set_issue(self.hass, self.entry_id, code)

    def clear_error(self, code: str) -> None:
        if code in ISSUES and self.entry_id:
            async_clear_issue(self.hass, self.entry_id, code)
        if self.last_error == code:
            self.last_error = None

    async def async_stop(self) -> None:
        self.stopping = True
        analysis_tasks = tuple((self.analysis_tasks or {}).values())
        for task in analysis_tasks:
            task.cancel()
        if analysis_tasks:
            await asyncio.gather(*analysis_tasks, return_exceptions=True)
        self.analysis_tasks = {}
        for unsubscribe in self.unsubscribe_callbacks or []:
            try:
                unsubscribe()
            except Exception:  # noqa: BLE001
                _LOGGER.exception("Failed to unsubscribe entry resource")
        self.unsubscribe_callbacks = []
        await self.queue.async_stop()
        settlement_tasks = tuple((self.settlement_tasks or {}).values())
        for task in settlement_tasks:
            task.cancel()
        if settlement_tasks:
            await asyncio.gather(*settlement_tasks, return_exceptions=True)
        self.settlement_tasks = {}
        media_tasks = tuple((self.media_tasks or {}).values())
        for task in media_tasks:
            task.cancel()
        if media_tasks:
            await asyncio.gather(*media_tasks, return_exceptions=True)
        self.media_tasks = {}
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            await asyncio.gather(self.cleanup_task, return_exceptions=True)
            self.cleanup_task = None
        if self.door_watchdog_task is not None:
            self.door_watchdog_task.cancel()
            await asyncio.gather(self.door_watchdog_task, return_exceptions=True)
            self.door_watchdog_task = None
        if self.frigate_client is not None:
            await self.frigate_client.async_close()
        if self.delivery is not None:
            await self.delivery.async_stop()
