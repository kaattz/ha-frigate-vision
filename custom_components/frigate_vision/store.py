"""Persistent activity store."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store

from .models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    BufferedIngress,
    IngressMessage,
    ModelValidationError,
    analysis_key,
    attempt_activity_id,
    retry_is_safe,
)

STORE_VERSION = 1
TERMINAL_STAGES = {ActivityStage.COMPLETED, ActivityStage.FAILED}
TRANSITIONS = {
    ActivityStage.COLLECTING: {ActivityStage.SEALED, ActivityStage.FAILED},
    ActivityStage.SEALED: {ActivityStage.EVIDENCE_READY, ActivityStage.FAILED},
    ActivityStage.EVIDENCE_READY: {
        ActivityStage.ANALYSIS_STARTED,
        ActivityStage.COMPLETED,
        ActivityStage.FAILED,
    },
    ActivityStage.ANALYSIS_STARTED: {
        ActivityStage.ANALYSIS_DONE,
        ActivityStage.FAILED,
    },
    ActivityStage.ANALYSIS_DONE: {
        ActivityStage.DELIVERY_STARTED,
        ActivityStage.COMPLETED,
        ActivityStage.FAILED,
    },
    ActivityStage.DELIVERY_STARTED: {
        ActivityStage.COMPLETED,
        ActivityStage.FAILED,
    },
}


class StoreConflictError(RuntimeError):
    """A compare-and-set or identity constraint failed."""


class ActivityStore:
    """Versioned activities for one config entry."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        *,
        max_activities: int = 1000,
        max_ingress_buffer: int = 200,
    ) -> None:
        if max_activities < 1 or max_ingress_buffer < 1:
            raise ValueError("invalid_store_capacity")
        self._entry_id = entry_id
        self._max_activities = max_activities
        self._max_ingress_buffer = max_ingress_buffer
        self._store: Store[dict[str, Any]] = Store(
            hass,
            STORE_VERSION,
            f"frigate_vision.{entry_id}",
            atomic_writes=True,
        )
        self._lock = asyncio.Lock()
        self._activities: dict[str, ActivityRecord] = {}
        self._ingress_buffer: dict[str, BufferedIngress] = {}
        self._listeners: list[Callable[[ActivityRecord], None]] = []

    def async_subscribe(
        self, listener: Callable[[ActivityRecord], None]
    ) -> Callable[[], None]:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def _notify(self, record: ActivityRecord) -> None:
        if record.stage in TERMINAL_STAGES:
            for listener in tuple(self._listeners):
                listener(record)

    async def async_load(self) -> None:
        raw = await self._store.async_load()
        if raw is None:
            return
        if not {"schema_version", "activities"} <= set(raw) or set(raw) - {
            "schema_version",
            "activities",
            "ingress_buffer",
        }:
            raise ModelValidationError("invalid_store")
        if raw["schema_version"] != STORE_VERSION or not isinstance(
            raw["activities"], dict
        ):
            raise ModelValidationError("unsupported_store_version")
        loaded: dict[str, ActivityRecord] = {}
        for activity_id, payload in raw["activities"].items():
            if not isinstance(payload, dict):
                raise ModelValidationError("invalid_store_activity")
            record = ActivityRecord.from_dict(payload)
            if record.activity_id != activity_id or record.entry_id != self._entry_id:
                raise ModelValidationError("store_identity_mismatch")
            loaded[activity_id] = record
        raw_buffer = raw.get("ingress_buffer", {})
        if not isinstance(raw_buffer, dict):
            raise ModelValidationError("invalid_ingress_buffer")
        buffered: dict[str, BufferedIngress] = {}
        for buffer_id, payload in raw_buffer.items():
            if not isinstance(payload, dict):
                raise ModelValidationError("invalid_buffered_ingress")
            item = BufferedIngress.from_dict(payload)
            if item.buffer_id != buffer_id or item.message.entry_id != self._entry_id:
                raise ModelValidationError("store_identity_mismatch")
            buffered[buffer_id] = item
        if len(buffered) > self._max_ingress_buffer:
            raise ModelValidationError("ingress_buffer_capacity")
        self._activities = loaded
        self._ingress_buffer = buffered

    async def async_create(self, record: ActivityRecord) -> ActivityRecord:
        async with self._lock:
            if record.entry_id != self._entry_id:
                raise StoreConflictError("entry_mismatch")
            existing = self._activities.get(record.activity_id)
            if existing is not None:
                if existing.identity() != record.identity():
                    raise StoreConflictError("identity_conflict")
                return existing
            candidate = {**self._activities, record.activity_id: record}
            if len(candidate) > self._max_activities:
                terminal = sorted(
                    (
                        item
                        for item in self._activities.values()
                        if item.stage in TERMINAL_STAGES
                    ),
                    key=lambda item: (item.updated_at, item.activity_id),
                )
                while len(candidate) > self._max_activities and terminal:
                    candidate.pop(terminal.pop(0).activity_id)
                if len(candidate) > self._max_activities:
                    raise StoreConflictError("history_capacity")
            await self._async_save(candidate)
            self._activities = candidate
            self._notify(record)
            return record

    async def async_transition(
        self,
        activity_id: str,
        expected_stage: ActivityStage,
        next_stage: ActivityStage,
        *,
        updated_at: float,
        error_code: str | None = None,
        association_deadline: float | None = None,
        finalization_deadline: float | None = None,
        door_closed_at: float | None = None,
        door_remained_open: bool | None = None,
    ) -> ActivityRecord:
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage in TERMINAL_STAGES:
                raise StoreConflictError("terminal_activity")
            if existing.stage is not expected_stage:
                raise StoreConflictError("stage_conflict")
            if next_stage not in TRANSITIONS.get(existing.stage, set()):
                raise StoreConflictError("invalid_transition")
            if next_stage is ActivityStage.FAILED and error_code is None:
                raise StoreConflictError("failure_code_required")
            if next_stage is not ActivityStage.FAILED and error_code is not None:
                raise StoreConflictError("unexpected_error_code")
            updated = replace(
                existing,
                stage=next_stage,
                updated_at=updated_at,
                error_code=error_code,
                association_deadline=(
                    association_deadline
                    if association_deadline is not None
                    else existing.association_deadline
                ),
                finalization_deadline=(
                    finalization_deadline
                    if finalization_deadline is not None
                    else existing.finalization_deadline
                ),
                door_closed_at=(
                    door_closed_at
                    if door_closed_at is not None
                    else existing.door_closed_at
                ),
                door_remained_open=(
                    door_remained_open
                    if door_remained_open is not None
                    else existing.door_remained_open
                ),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            self._notify(updated)
            return updated

    def get(self, activity_id: str) -> ActivityRecord | None:
        return self._activities.get(activity_id)

    def all(self) -> tuple[ActivityRecord, ...]:
        return tuple(self._activities.values())

    def buffered_ingress(self) -> tuple[BufferedIngress, ...]:
        return tuple(
            sorted(
                self._ingress_buffer.values(),
                key=lambda item: (
                    item.message.occurred_at,
                    item.message.kind.value,
                    item.message.source_id,
                ),
            )
        )

    async def async_buffer_ingress(
        self, message: IngressMessage, *, settle_after: float
    ) -> BufferedIngress:
        item = BufferedIngress(message=message, settle_after=settle_after)
        async with self._lock:
            if message.entry_id != self._entry_id:
                raise StoreConflictError("entry_mismatch")
            existing = self._ingress_buffer.get(item.buffer_id)
            if existing is not None:
                if existing.message != message:
                    raise StoreConflictError("identity_conflict")
                return existing
            if len(self._ingress_buffer) >= self._max_ingress_buffer:
                raise StoreConflictError("ingress_buffer_capacity")
            buffered = {**self._ingress_buffer, item.buffer_id: item}
            await self._async_save(self._activities, buffered)
            self._ingress_buffer = buffered
            return item

    async def async_remove_buffered_ingress(self, buffer_id: str) -> bool:
        async with self._lock:
            if buffer_id not in self._ingress_buffer:
                return False
            buffered = dict(self._ingress_buffer)
            del buffered[buffer_id]
            await self._async_save(self._activities, buffered)
            self._ingress_buffer = buffered
            return True

    async def async_merge_context(
        self,
        activity_id: str,
        *,
        detection_ids: tuple[str, ...] = (),
        review_ids: tuple[str, ...] = (),
        zone_update: tuple[float, tuple[str, ...]] | None = None,
        detection_zone_update: tuple[str, float, tuple[str, ...]] | None = None,
        doorbell_at: float | None = None,
        doorbell_times: tuple[float, ...] = (),
        updated_at: float,
    ) -> ActivityRecord:
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage in TERMINAL_STAGES:
                raise StoreConflictError("terminal_activity")
            zone_updates = existing.zone_updates
            if zone_update is not None:
                occurred_at, zones = zone_update
                by_timestamp = {
                    timestamp: set(existing_zones)
                    for timestamp, existing_zones in zone_updates
                }
                by_timestamp.setdefault(occurred_at, set()).update(zones)
                zone_updates = tuple(
                    (timestamp, tuple(sorted(merged_zones)))
                    for timestamp, merged_zones in sorted(by_timestamp.items())
                )
            detection_zone_updates = existing.detection_zone_updates
            if detection_zone_update is not None:
                detection_id, occurred_at, zones = detection_zone_update
                by_detection_time = {
                    (existing_id, timestamp): set(existing_zones)
                    for existing_id, timestamp, existing_zones in detection_zone_updates
                }
                by_detection_time.setdefault((detection_id, occurred_at), set()).update(
                    zones
                )
                detection_zone_updates = tuple(
                    (existing_id, timestamp, tuple(sorted(merged_zones)))
                    for (existing_id, timestamp), merged_zones in sorted(
                        by_detection_time.items(),
                        key=lambda item: (item[0][1], item[0][0]),
                    )
                )
            merged_doorbell_times = tuple(
                sorted(
                    {
                        *existing.doorbell_times,
                        *(
                            (existing.doorbell_at,)
                            if existing.doorbell_at is not None
                            else ()
                        ),
                        *((doorbell_at,) if doorbell_at is not None else ()),
                        *doorbell_times,
                    }
                )
            )
            updated = replace(
                existing,
                detection_ids=tuple(
                    dict.fromkeys((*existing.detection_ids, *detection_ids))
                ),
                review_ids=tuple(dict.fromkeys((*existing.review_ids, *review_ids))),
                zone_updates=zone_updates,
                detection_zone_updates=detection_zone_updates,
                updated_at=max(existing.updated_at, updated_at),
                doorbell_at=(
                    merged_doorbell_times[0] if merged_doorbell_times else None
                ),
                doorbell_times=merged_doorbell_times,
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            self._notify(updated)
            return updated

    async def async_start_side_effect(
        self,
        activity_id: str,
        key: str,
        expected_stage: ActivityStage,
        next_stage: ActivityStage,
        *,
        updated_at: float,
    ) -> bool:
        """Atomically claim an uncertain side effect and enter its started stage."""
        required_prefix = {
            ActivityStage.ANALYSIS_STARTED: f"analysis:{activity_id}:",
            ActivityStage.DELIVERY_STARTED: f"delivery:{activity_id}:",
        }.get(next_stage)
        if required_prefix is None or not key.startswith(required_prefix):
            raise StoreConflictError("side_effect_key_mismatch")
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if key in existing.claimed_side_effects:
                if existing.stage is next_stage:
                    return False
                raise StoreConflictError("side_effect_stage_conflict")
            if existing.stage in TERMINAL_STAGES:
                raise StoreConflictError("terminal_activity")
            if existing.stage is not expected_stage:
                raise StoreConflictError("stage_conflict")
            if next_stage not in TRANSITIONS.get(expected_stage, set()):
                raise StoreConflictError("invalid_transition")
            updated = replace(
                existing,
                stage=next_stage,
                claimed_side_effects=(*existing.claimed_side_effects, key),
                updated_at=max(existing.updated_at, updated_at),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            return True

    async def async_update_contact(
        self, activity_id: str, *, is_open: bool, updated_at: float
    ) -> ActivityRecord:
        """Persist door-contact continuity evidence for a collecting cycle."""
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage is not ActivityStage.COLLECTING:
                raise StoreConflictError("stage_conflict")
            contact_seen_open = existing.contact_seen_open
            contact_seen_close = existing.contact_seen_close
            contact_reopened = existing.contact_reopened
            if is_open:
                contact_reopened = contact_reopened or contact_seen_close
                contact_seen_open = True
            elif contact_seen_open:
                contact_seen_close = True
            updated = replace(
                existing,
                contact_seen_open=contact_seen_open,
                contact_seen_close=contact_seen_close,
                contact_reopened=contact_reopened,
                updated_at=max(existing.updated_at, updated_at),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            return updated

    async def async_complete_media(
        self,
        activity_id: str,
        *,
        key: str,
        evidence_mode: str,
        evidence_revision: int,
        evidence_path: str,
        evidence_media_url: str,
        sample_times: tuple[float, ...],
        updated_at: float,
        selection_source: str | None = None,
    ) -> ActivityRecord:
        """Atomically register a validated stable media file as evidence ready."""
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            expected_key = f"media:{activity_id}:{evidence_revision}"
            if key != expected_key:
                raise StoreConflictError("side_effect_key_mismatch")
            if key in existing.claimed_side_effects:
                if existing.stage is ActivityStage.EVIDENCE_READY:
                    return existing
                raise StoreConflictError("side_effect_stage_conflict")
            if existing.stage is not ActivityStage.SEALED:
                raise StoreConflictError("stage_conflict")
            updated = replace(
                existing,
                stage=ActivityStage.EVIDENCE_READY,
                evidence_mode=evidence_mode,
                evidence_revision=evidence_revision,
                evidence_path=evidence_path,
                evidence_media_url=evidence_media_url,
                sample_times=sample_times,
                selection_source=selection_source,
                claimed_side_effects=(*existing.claimed_side_effects, key),
                updated_at=max(existing.updated_at, updated_at),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            return updated

    async def async_expire_evidence(
        self, activity_id: str, *, expired_at: float
    ) -> ActivityRecord:
        """Persist retention expiry before deleting registered files."""
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage not in TERMINAL_STAGES:
                raise StoreConflictError("stage_conflict")
            if existing.evidence_expired_at is not None:
                return existing
            updated = replace(existing, evidence_expired_at=expired_at)
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            return updated

    async def async_create_retry(
        self, activity_id: str, *, now: float
    ) -> ActivityRecord:
        async with self._lock:
            if "_attempt_" in activity_id:
                raise StoreConflictError("retry_requires_root_activity")
            original = self._activities.get(activity_id)
            if (
                original is None
                or original.stage is not ActivityStage.FAILED
                or original.error_code is None
                or not retry_is_safe(original.error_code)
            ):
                raise StoreConflictError("retry_not_safe")
            prefix = f"{activity_id}_attempt_"
            attempts = [
                item
                for item in self._activities.values()
                if item.activity_id.startswith(prefix)
            ]
            if any(item.stage not in TERMINAL_STAGES for item in attempts):
                raise StoreConflictError("retry_already_active")
            retry_id = attempt_activity_id(activity_id, len(attempts) + 1)
            has_evidence = (
                original.evidence_path is not None
                and original.evidence_expired_at is None
            )
            retry = replace(
                original,
                activity_id=retry_id,
                source=original.source,
                stage=(
                    ActivityStage.EVIDENCE_READY
                    if has_evidence
                    else ActivityStage.SEALED
                ),
                updated_at=now,
                error_code=None,
                claimed_side_effects=((f"media:{retry_id}:1",) if has_evidence else ()),
                prompt_version=None,
                classification=None,
                description=None,
                confidence=None,
            )
            candidate = {**self._activities, retry_id: retry}
            if len(candidate) > self._max_activities:
                raise StoreConflictError("history_capacity")
            await self._async_save(candidate)
            self._activities = candidate
            return retry

    async def async_complete_analysis(
        self,
        activity_id: str,
        *,
        scene_mode: str,
        prompt_version: str,
        classification: str,
        description: str,
        confidence: int,
        updated_at: float,
    ) -> ActivityRecord:
        # Derived from the same helper the claim used. Rebuilding the string here
        # is what let the two spellings drift apart: the claim carries the scene
        # (`analysis:<id>:<scene>:<version>`) and a hand-written
        # `analysis:<id>:<version>` never matched it, so every analysis failed
        # with `side_effect_key_mismatch` *after* the side effect was claimed.
        key = analysis_key(activity_id, scene_mode, prompt_version)
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage is not ActivityStage.ANALYSIS_STARTED:
                raise StoreConflictError("stage_conflict")
            if key not in existing.claimed_side_effects:
                raise StoreConflictError("side_effect_key_mismatch")
            updated = replace(
                existing,
                stage=ActivityStage.ANALYSIS_DONE,
                prompt_version=prompt_version,
                classification=classification,
                description=description,
                confidence=confidence,
                updated_at=max(existing.updated_at, updated_at),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            return updated

    async def async_start_delivery(
        self, activity_id: str, *, attempt_id: str, updated_at: float
    ) -> ActivityRecord:
        key = f"delivery:{activity_id}:{attempt_id}"
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage is not ActivityStage.ANALYSIS_DONE:
                raise StoreConflictError("stage_conflict")
            if existing.processing_mode.value != "live":
                raise StoreConflictError("delivery_mode_forbidden")
            if any(
                value.startswith(f"delivery:{activity_id}:")
                for value in existing.claimed_side_effects
            ):
                raise StoreConflictError("delivery_already_started")
            updated = replace(
                existing,
                stage=ActivityStage.DELIVERY_STARTED,
                delivery_attempt_id=attempt_id,
                claimed_side_effects=(*existing.claimed_side_effects, key),
                updated_at=max(existing.updated_at, updated_at),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            return updated

    async def async_complete_mode(
        self, activity_id: str, *, mode: str, updated_at: float
    ) -> ActivityRecord:
        expected = {
            "observe": ActivityStage.EVIDENCE_READY,
            "shadow": ActivityStage.ANALYSIS_DONE,
        }.get(mode)
        if expected is None:
            raise StoreConflictError("invalid_completion_kind")
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage is not expected or existing.processing_mode.value != mode:
                raise StoreConflictError("stage_conflict")
            updated = replace(
                existing,
                stage=ActivityStage.COMPLETED,
                completion_kind=mode,
                updated_at=max(existing.updated_at, updated_at),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            self._notify(updated)
            return updated

    async def async_ack_delivery(
        self, activity_id: str, *, attempt_id: str, updated_at: float
    ) -> ActivityRecord:
        async with self._lock:
            existing = self._activities.get(activity_id)
            if existing is None:
                raise StoreConflictError("activity_missing")
            if existing.stage is not ActivityStage.DELIVERY_STARTED:
                raise StoreConflictError("stage_conflict")
            if existing.delivery_attempt_id != attempt_id:
                raise StoreConflictError("delivery_identity_mismatch")
            updated = replace(
                existing,
                stage=ActivityStage.COMPLETED,
                completion_kind="live",
                updated_at=max(existing.updated_at, updated_at),
            )
            candidate = {**self._activities, activity_id: updated}
            await self._async_save(candidate)
            self._activities = candidate
            self._notify(updated)
            return updated

    async def async_recover(self, now: float) -> list[ActivityRecord]:
        """Audit nonterminal activities without repeating uncertain side effects."""
        async with self._lock:
            recovered: list[ActivityRecord] = []
            candidate = dict(self._activities)
            changed = False
            collecting_by_camera: dict[str, list[str]] = {}
            for item in candidate.values():
                if (
                    item.source is ActivitySource.DOOR_CYCLE
                    and item.stage is ActivityStage.COLLECTING
                ):
                    collecting_by_camera.setdefault(item.camera, []).append(
                        item.activity_id
                    )
            conflicts = {
                activity_id
                for ids in collecting_by_camera.values()
                if len(ids) > 1
                for activity_id in ids
            }
            for activity_id, record in list(candidate.items()):
                error_code = None
                if activity_id in conflicts:
                    error_code = "conflicting_door_cycles"
                elif record.stage is ActivityStage.ANALYSIS_STARTED:
                    error_code = "analysis_outcome_unknown"
                elif record.stage is ActivityStage.DELIVERY_STARTED:
                    error_code = "delivery_outcome_unknown"
                if error_code is not None:
                    record = replace(
                        record,
                        stage=ActivityStage.FAILED,
                        updated_at=now,
                        error_code=error_code,
                    )
                    candidate[activity_id] = record
                    changed = True
                if record.stage not in TERMINAL_STAGES or error_code is not None:
                    recovered.append(record)
            if changed:
                await self._async_save(candidate)
                self._activities = candidate
            return recovered

    async def _async_save(
        self,
        activities: dict[str, ActivityRecord],
        ingress_buffer: dict[str, BufferedIngress] | None = None,
    ) -> None:
        buffered = self._ingress_buffer if ingress_buffer is None else ingress_buffer
        await self._store.async_save(
            {
                "schema_version": STORE_VERSION,
                "activities": {
                    key: value.to_dict() for key, value in activities.items()
                },
                "ingress_buffer": {
                    key: value.to_dict() for key, value in buffered.items()
                },
            }
        )
