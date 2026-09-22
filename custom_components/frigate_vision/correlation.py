"""Deterministic Review ownership and zone direction evidence."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace

from .models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    BufferedIngress,
    IngressKind,
    IngressMessage,
    ProcessingMode,
    review_activity_id,
)
from .store import ActivityStore


class CorrelationError(RuntimeError):
    """Evidence cannot be assigned uniquely."""


@dataclass(frozen=True, slots=True)
class ZoneRoles:
    near: frozenset[str]
    transition: frozenset[str]
    far: frozenset[str]

    def classify(self, zones: Iterable[str]) -> str | None:
        values = set(zones)
        matched = [
            role
            for role, configured in (
                ("near", self.near),
                ("transition", self.transition),
                ("far", self.far),
            )
            if values & configured
        ]
        return matched[0] if len(matched) == 1 else None


@dataclass(frozen=True, slots=True)
class ZoneAnchor:
    occurred_at: float
    role: str


def anchor_sequence(
    updates: Sequence[tuple[float, Sequence[str]]], roles: ZoneRoles
) -> list[ZoneAnchor]:
    anchors: list[ZoneAnchor] = []
    for occurred_at, zones in sorted(updates, key=lambda item: item[0]):
        role = roles.classify(zones)
        if role is None or (anchors and anchors[-1].role == role):
            continue
        anchors.append(ZoneAnchor(occurred_at, role))
    return anchors


def infer_direction(anchors: Sequence[ZoneAnchor]) -> str:
    """Infer a candidate direction from stable role anchors only.

    REQ-6 defines the boundary, not the far zone itself: near -> non-near is
    an outbound candidate and non-near -> near is an inbound candidate. A
    single anchor (or a sequence that never leaves near) stays ambiguous
    because the person may simply stand at the door.
    """
    roles = [anchor.role for anchor in anchors]
    if len(roles) < 2:
        return "ambiguous"
    if roles[0] != "near" and roles[-1] == "near":
        return "inbound"
    if roles[0] == "near" and roles[-1] != "near":
        return "outbound"
    if roles[0] == "near" and roles[-1] == "near":
        if any(role != "near" for role in roles[1:-1]):
            return "roundtrip"
    return "ambiguous"


def find_review_owner(
    activities: Iterable[ActivityRecord], detection_ids: set[str], camera: str
) -> ActivityRecord | None:
    """Return the single non-terminal activity that owns a detection id.

    Terminal activities cannot absorb new context: the Store rejects any
    write to them. A Review that overlaps only terminal activities is an
    independent entry and must get its own activity instead of failing.
    """
    matches = [
        activity
        for activity in activities
        if activity.camera == camera
        and activity.stage not in {ActivityStage.COMPLETED, ActivityStage.FAILED}
        and detection_ids.intersection(activity.detection_ids)
    ]
    if len(matches) > 1:
        raise CorrelationError("ambiguous_review_ownership")
    return matches[0] if matches else None


class CorrelationEngine:
    """Attach Frigate evidence to exactly one activity."""

    def __init__(
        self,
        store: ActivityStore,
        *,
        entry_id: str,
        camera: str,
        processing_mode: ProcessingMode,
        clock: Callable[[], float] = time.time,
        review_settle_seconds: float = 10,
        min_review_seconds: float = 0.0,
    ) -> None:
        self._store = store
        self._entry_id = entry_id
        self._camera = camera
        self._processing_mode = processing_mode
        self._clock = clock
        self._review_settle_seconds = review_settle_seconds
        self._min_review_seconds = min_review_seconds

    @property
    def camera(self) -> str:
        return self._camera

    async def async_handle(self, message: IngressMessage) -> ActivityRecord | None:
        if message.entry_id != self._entry_id or message.camera != self._camera:
            raise CorrelationError("ingress_identity_mismatch")
        if message.kind is IngressKind.FRIGATE_EVENT:
            if message.event_id is None:
                raise CorrelationError("event_id_missing")
            related = [
                record
                for record in self._store.all()
                if record.camera == self._camera
                and record.stage in {ActivityStage.COLLECTING, ActivityStage.SEALED}
                and record.created_at - 30 <= message.occurred_at
                and (
                    record.finalization_deadline is None
                    or message.occurred_at <= record.finalization_deadline
                )
            ]
            candidates = [
                record
                for record in related
                if (
                    (
                        message.event_id in record.detection_ids
                        and (
                            record.finalization_deadline is None
                            or message.occurred_at <= record.finalization_deadline
                        )
                    )
                    or (
                        message.event_id not in record.detection_ids
                        and (
                            record.stage is ActivityStage.COLLECTING
                            or (
                                record.association_deadline is not None
                                and message.occurred_at <= record.association_deadline
                            )
                        )
                    )
                )
            ]
            if len(candidates) != 1:
                if not candidates:
                    if related:
                        raise CorrelationError("event_owner_missing")
                    await self._store.async_buffer_ingress(
                        message,
                        settle_after=max(self._clock(), message.occurred_at) + 30,
                    )
                    return None
                raise CorrelationError("ambiguous_event_ownership")
            updated = await self._store.async_merge_context(
                candidates[0].activity_id,
                detection_ids=(message.event_id,),
                detection_zone_update=(
                    message.event_id,
                    message.occurred_at,
                    message.current_zones,
                ),
                updated_at=message.occurred_at,
            )
            await self._async_attach_matching_reviews(updated)
            return updated
        if message.kind is IngressKind.DOORBELL:
            candidates = [
                record
                for record in self._store.all()
                if record.camera == self._camera
                and record.stage in {ActivityStage.COLLECTING, ActivityStage.SEALED}
                and record.created_at - 30 <= message.occurred_at
                and (
                    record.finalization_deadline is None
                    or message.occurred_at <= record.finalization_deadline
                )
            ]
            if len(candidates) > 1:
                raise CorrelationError("ambiguous_doorbell_ownership")
            if candidates:
                return await self._store.async_merge_context(
                    candidates[0].activity_id,
                    doorbell_at=message.occurred_at,
                    updated_at=message.occurred_at,
                )
            await self._store.async_buffer_ingress(
                message,
                settle_after=max(self._clock(), message.occurred_at)
                + self._review_settle_seconds
                + 120,
            )
            return None
        if message.kind is not IngressKind.FRIGATE_REVIEW or message.review_id is None:
            raise CorrelationError("unsupported_ingress_kind")
        for record in self._store.all():
            if message.review_id in record.review_ids:
                return record
        # A very short review is a person passing the frame, not an errand worth
        # reporting. Dropping it here also avoids fetching frames and calling the
        # vision model for something the user does not want to hear about.
        # Measured over 233 stored reviews: 9% run under 10s.
        if self._min_review_seconds > 0 and message.started_at is not None:
            span = message.occurred_at - message.started_at
            if span < self._min_review_seconds:
                return None
        try:
            owner = find_review_owner(
                self._store.all(), set(message.detection_ids), self._camera
            )
        except CorrelationError as exc:
            if str(exc) != "ambiguous_review_ownership":
                raise
            return await self._async_create_failed_review(
                message, "ambiguous_review_ownership"
            )
        if owner is not None:
            record = await self._store.async_merge_context(
                owner.activity_id,
                detection_ids=message.detection_ids,
                review_ids=(message.review_id,),
                updated_at=message.occurred_at,
            )
            return await self._async_attach_doorbell_context(
                record, message.occurred_at
            )
        await self._store.async_buffer_ingress(
            replace(
                message,
                processing_mode=(message.processing_mode or self._processing_mode),
            ),
            settle_after=max(self._clock(), message.occurred_at)
            + self._review_settle_seconds,
        )
        return None

    async def async_replay_for_activity(
        self, activity_id: str
    ) -> tuple[ActivityRecord, ...]:
        activity = self._store.get(activity_id)
        if activity is None:
            raise CorrelationError("activity_missing")
        replayed: list[ActivityRecord] = []
        for buffered in self._store.buffered_ingress():
            message = buffered.message
            if (
                message.kind is not IngressKind.FRIGATE_EVENT
                or message.camera != activity.camera
                or message.event_id is None
                or not activity.created_at - 30
                <= message.occurred_at
                <= activity.created_at
            ):
                continue
            activity = await self._store.async_merge_context(
                activity.activity_id,
                detection_ids=(message.event_id,),
                detection_zone_update=(
                    message.event_id,
                    message.occurred_at,
                    message.current_zones,
                ),
                updated_at=message.occurred_at,
            )
            await self._store.async_remove_buffered_ingress(buffered.buffer_id)
            replayed.append(activity)
        await self._async_attach_matching_reviews(activity)
        return tuple(replayed)

    async def async_settle_due(self, now: float) -> tuple[ActivityRecord, ...]:
        settled: list[ActivityRecord] = []
        for buffered in self._store.buffered_ingress():
            message = buffered.message
            if (
                message.kind is not IngressKind.FRIGATE_REVIEW
                and buffered.settle_after <= now
            ):
                await self._store.async_remove_buffered_ingress(buffered.buffer_id)
                continue
            if (
                message.kind is not IngressKind.FRIGATE_REVIEW
                or buffered.settle_after > now
                or message.review_id is None
            ):
                continue
            existing = next(
                (
                    record
                    for record in self._store.all()
                    if message.review_id in record.review_ids
                ),
                None,
            )
            if existing is not None:
                record = existing
                if record.stage in {
                    ActivityStage.COMPLETED,
                    ActivityStage.FAILED,
                }:
                    for doorbell in self._matching_doorbells(message.occurred_at):
                        await self._store.async_remove_buffered_ingress(
                            doorbell.buffer_id
                        )
                    await self._store.async_remove_buffered_ingress(buffered.buffer_id)
                    settled.append(record)
                    continue
            else:
                try:
                    owner = find_review_owner(
                        self._store.all(), set(message.detection_ids), self._camera
                    )
                except CorrelationError as exc:
                    if str(exc) != "ambiguous_review_ownership":
                        raise
                    record = await self._async_create_failed_review(
                        message, "ambiguous_review_ownership"
                    )
                    await self._store.async_remove_buffered_ingress(buffered.buffer_id)
                    settled.append(record)
                    continue
                if owner is not None:
                    record = await self._store.async_merge_context(
                        owner.activity_id,
                        detection_ids=message.detection_ids,
                        review_ids=(message.review_id,),
                        updated_at=message.occurred_at,
                    )
                else:
                    record = await self._async_create_standalone(message)
            record = await self._async_attach_doorbell_context(
                record, message.occurred_at
            )
            await self._store.async_remove_buffered_ingress(buffered.buffer_id)
            settled.append(record)
        return tuple(settled)

    async def _async_attach_matching_reviews(self, activity: ActivityRecord) -> None:
        for buffered in self._store.buffered_ingress():
            message = buffered.message
            if (
                message.kind is not IngressKind.FRIGATE_REVIEW
                or message.review_id is None
                or not set(message.detection_ids).intersection(activity.detection_ids)
            ):
                continue
            updated = await self._store.async_merge_context(
                activity.activity_id,
                detection_ids=message.detection_ids,
                review_ids=(message.review_id,),
                updated_at=message.occurred_at,
            )
            activity = await self._async_attach_doorbell_context(
                updated, message.occurred_at
            )
            await self._store.async_remove_buffered_ingress(buffered.buffer_id)

    async def _async_create_standalone(self, message: IngressMessage) -> ActivityRecord:
        if message.review_id is None:
            raise CorrelationError("review_id_missing")
        record = ActivityRecord(
            activity_id=review_activity_id(
                self._entry_id, self._camera, message.review_id
            ),
            entry_id=self._entry_id,
            source=(
                ActivitySource.MANUAL_REVIEW
                if message.manual
                else ActivitySource.STANDALONE_REVIEW
            ),
            stage=ActivityStage.SEALED,
            processing_mode=(message.processing_mode or self._processing_mode),
            created_at=(
                message.started_at
                if message.started_at is not None
                else message.occurred_at
            ),
            updated_at=message.occurred_at,
            camera=self._camera,
            review_ids=(message.review_id,),
            detection_ids=message.detection_ids,
            finalization_deadline=message.occurred_at,
        )
        return await self._store.async_create(record)

    async def _async_create_failed_review(
        self, message: IngressMessage, error_code: str
    ) -> ActivityRecord:
        if message.review_id is None:
            raise CorrelationError("review_id_missing")
        record = ActivityRecord(
            activity_id=review_activity_id(
                self._entry_id, self._camera, message.review_id
            ),
            entry_id=self._entry_id,
            source=(
                ActivitySource.MANUAL_REVIEW
                if message.manual
                else ActivitySource.STANDALONE_REVIEW
            ),
            stage=ActivityStage.FAILED,
            processing_mode=(message.processing_mode or self._processing_mode),
            created_at=(
                message.started_at
                if message.started_at is not None
                else message.occurred_at
            ),
            updated_at=message.occurred_at,
            camera=self._camera,
            review_ids=(message.review_id,),
            detection_ids=message.detection_ids,
            finalization_deadline=message.occurred_at,
            error_code=error_code,
        )
        return await self._store.async_create(record)

    async def _async_attach_doorbell_context(
        self, record: ActivityRecord, review_ended_at: float
    ) -> ActivityRecord:
        matches = self._matching_doorbells(review_ended_at)
        if not matches:
            return record
        updated = await self._store.async_merge_context(
            record.activity_id,
            doorbell_times=tuple(matched.message.occurred_at for matched in matches),
            updated_at=review_ended_at,
        )
        for matched in matches:
            await self._store.async_remove_buffered_ingress(matched.buffer_id)
        return updated

    def _matching_doorbells(self, review_ended_at: float) -> list[BufferedIngress]:
        return [
            buffered
            for buffered in self._store.buffered_ingress()
            if buffered.message.kind is IngressKind.DOORBELL
            and buffered.message.camera == self._camera
            and review_ended_at - 120
            <= buffered.message.occurred_at
            <= review_ended_at + 10
        ]
