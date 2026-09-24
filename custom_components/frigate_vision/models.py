"""Typed activity and ingress models."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Self

SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,192}$")
SAFE_SIDE_EFFECT_KEY = re.compile(r"^[A-Za-z0-9_.:-]{1,256}$")
# 分类值允许中文：用户要能写「宠物」而不必被迫写 ASCII。
# 比 SAFE_ID 宽，但仍拒绝控制字符与换行——那些会真的破坏通知显示和日志。
SAFE_CLASSIFICATION = re.compile(r"^[^\x00-\x1f\x7f]{1,192}$")


class ModelValidationError(ValueError):
    """A persisted or incoming model is invalid."""


class ActivitySource(StrEnum):
    DOOR_CYCLE = "door_cycle"
    STANDALONE_REVIEW = "standalone_review"
    MANUAL_REVIEW = "manual_review"


class ActivityStage(StrEnum):
    COLLECTING = "collecting"
    SEALED = "sealed"
    EVIDENCE_READY = "evidence_ready"
    ANALYSIS_STARTED = "analysis_started"
    ANALYSIS_DONE = "analysis_done"
    DELIVERY_STARTED = "delivery_started"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingMode(StrEnum):
    OBSERVE = "observe"
    SHADOW = "shadow"
    LIVE = "live"


class IngressKind(StrEnum):
    DOOR = "door"
    DOORBELL = "doorbell"
    FRIGATE_EVENT = "frigate_event"
    FRIGATE_REVIEW = "frigate_review"


@dataclass(frozen=True, slots=True)
class IngressMessage:
    kind: IngressKind
    entry_id: str
    source_id: str
    occurred_at: float
    started_at: float | None = None
    camera: str | None = None
    event_id: str | None = None
    event_type: str | None = None
    review_id: str | None = None
    current_zones: tuple[str, ...] = ()
    entered_zones: tuple[str, ...] = ()
    detection_ids: tuple[str, ...] = ()
    processing_mode: ProcessingMode | None = None
    manual: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.manual, bool):
            raise ModelValidationError("invalid_manual_flag")
        for value in (
            self.entry_id,
            self.source_id,
            self.camera,
            self.event_id,
            self.review_id,
        ):
            if value is not None and not SAFE_ID.fullmatch(value):
                raise ModelValidationError("invalid_ingress_id")
        if self.event_type is not None and self.event_type not in {
            "new",
            "update",
            "end",
        }:
            raise ModelValidationError("invalid_event_type")
        if not math.isfinite(self.occurred_at) or self.occurred_at < 0:
            raise ModelValidationError("invalid_timestamp")
        if self.started_at is not None and (
            not math.isfinite(self.started_at)
            or self.started_at < 0
            or self.started_at > self.occurred_at
        ):
            raise ModelValidationError("invalid_timestamp_order")
        for values in (
            self.current_zones,
            self.entered_zones,
            self.detection_ids,
        ):
            if len(values) > 50 or len(values) != len(set(values)):
                raise ModelValidationError("invalid_ingress_zones")
            if any(not SAFE_ID.fullmatch(value) for value in values):
                raise ModelValidationError("invalid_ingress_id")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        payload["processing_mode"] = (
            self.processing_mode.value if self.processing_mode is not None else None
        )
        payload["occurred_at"] = float(self.occurred_at)
        if self.started_at is not None:
            payload["started_at"] = float(self.started_at)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        allowed = {
            "kind",
            "entry_id",
            "source_id",
            "occurred_at",
            "started_at",
            "camera",
            "event_id",
            "event_type",
            "review_id",
            "current_zones",
            "entered_zones",
            "detection_ids",
            "processing_mode",
            "manual",
        }
        if set(payload) - allowed:
            raise ModelValidationError("unknown_ingress_field")
        try:
            return cls(
                kind=IngressKind(payload["kind"]),
                entry_id=str(payload["entry_id"]),
                source_id=str(payload["source_id"]),
                occurred_at=float(payload["occurred_at"]),
                started_at=(
                    float(payload["started_at"])
                    if payload.get("started_at") is not None
                    else None
                ),
                camera=(str(payload["camera"]) if payload.get("camera") else None),
                event_id=(
                    str(payload["event_id"])
                    if payload.get("event_id") is not None
                    else None
                ),
                event_type=(
                    str(payload["event_type"])
                    if payload.get("event_type") is not None
                    else None
                ),
                review_id=(
                    str(payload["review_id"])
                    if payload.get("review_id") is not None
                    else None
                ),
                current_zones=tuple(
                    str(value) for value in payload.get("current_zones", [])
                ),
                entered_zones=tuple(
                    str(value) for value in payload.get("entered_zones", [])
                ),
                detection_ids=tuple(
                    str(value) for value in payload.get("detection_ids", [])
                ),
                processing_mode=(
                    ProcessingMode(payload["processing_mode"])
                    if payload.get("processing_mode") is not None
                    else None
                ),
                manual=payload.get("manual", False),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ModelValidationError):
                raise
            raise ModelValidationError("invalid_ingress") from exc


@dataclass(frozen=True, slots=True)
class BufferedIngress:
    """Persisted transport-order buffer entry."""

    message: IngressMessage
    settle_after: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.settle_after)
            or self.settle_after < self.message.occurred_at
        ):
            raise ModelValidationError("invalid_settle_after")

    @property
    def buffer_id(self) -> str:
        digest = hashlib.sha256(
            json.dumps(
                self.message.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:32]
        return _derived_id(
            "ingress",
            self.message.kind.value,
            digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message.to_dict(),
            "settle_after": self.settle_after,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        if set(payload) != {"message", "settle_after"} or not isinstance(
            payload.get("message"), dict
        ):
            raise ModelValidationError("invalid_buffered_ingress")
        try:
            return cls(
                message=IngressMessage.from_dict(payload["message"]),
                settle_after=float(payload["settle_after"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ModelValidationError):
                raise
            raise ModelValidationError("invalid_buffered_ingress") from exc


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    activity_id: str
    entry_id: str
    source: ActivitySource
    stage: ActivityStage
    processing_mode: ProcessingMode
    created_at: float
    updated_at: float
    camera: str
    schema_version: int = 1
    error_code: str | None = None
    association_deadline: float | None = None
    finalization_deadline: float | None = None
    review_ids: tuple[str, ...] = ()
    detection_ids: tuple[str, ...] = ()
    zone_updates: tuple[tuple[float, tuple[str, ...]], ...] = ()
    detection_zone_updates: tuple[tuple[str, float, tuple[str, ...]], ...] = ()
    opening_side: str = "unknown"
    door_closed_at: float | None = None
    door_remained_open: bool | None = None
    contact_seen_open: bool = False
    contact_seen_close: bool = False
    contact_reopened: bool = False
    doorbell_at: float | None = None
    doorbell_times: tuple[float, ...] = ()
    claimed_side_effects: tuple[str, ...] = ()
    evidence_mode: str | None = None
    evidence_revision: int = 0
    evidence_path: str | None = None
    evidence_media_url: str | None = None
    sample_times: tuple[float, ...] = ()
    selection_source: str | None = None
    evidence_expired_at: float | None = None
    prompt_version: str | None = None
    classification: str | None = None
    description: str | None = None
    confidence: int | None = None
    delivery_attempt_id: str | None = None
    completion_kind: str | None = None

    def __post_init__(self) -> None:
        for value, code in (
            (self.activity_id, "invalid_activity_id"),
            (self.entry_id, "invalid_entry_id"),
            (self.camera, "invalid_camera"),
        ):
            if not SAFE_ID.fullmatch(value):
                raise ModelValidationError(code)
        if self.schema_version != 1:
            raise ModelValidationError("unsupported_schema_version")
        if not all(
            math.isfinite(value) and value >= 0
            for value in (self.created_at, self.updated_at)
        ):
            raise ModelValidationError("invalid_timestamp")
        if self.updated_at < self.created_at:
            raise ModelValidationError("invalid_timestamp_order")
        if self.error_code is not None and not SAFE_ID.fullmatch(self.error_code):
            raise ModelValidationError("invalid_error_code")
        for values in (self.review_ids, self.detection_ids):
            if (
                len(values) > 100
                or len(values) != len(set(values))
                or any(not SAFE_ID.fullmatch(value) for value in values)
            ):
                raise ModelValidationError("invalid_activity_ids")
        if len(self.claimed_side_effects) != len(set(self.claimed_side_effects)) or any(
            not SAFE_SIDE_EFFECT_KEY.fullmatch(value)
            for value in self.claimed_side_effects
        ):
            raise ModelValidationError("invalid_side_effect_keys")
        if self.evidence_revision < 0:
            raise ModelValidationError("invalid_evidence_revision")
        if self.evidence_mode is not None and not SAFE_ID.fullmatch(self.evidence_mode):
            raise ModelValidationError("invalid_evidence_mode")
        for evidence_location in (self.evidence_path, self.evidence_media_url):
            if evidence_location is not None and (
                not evidence_location
                or len(evidence_location) > 1024
                or "\x00" in evidence_location
            ):
                raise ModelValidationError("invalid_evidence_location")
        if (
            # 0 means "not planned yet"; otherwise the count must be one a contact
            # sheet can be built from -- a whole number of 3-column rows, which
            # covers the 2x3 and 3x3 layouts the planner produces.
            len(self.sample_times) not in {0, 3, 6, 9, 12}
            or tuple(sorted(set(self.sample_times))) != self.sample_times
            or any(not math.isfinite(value) or value < 0 for value in self.sample_times)
        ):
            raise ModelValidationError("invalid_sample_times")
        if self.selection_source is not None and self.selection_source not in {
            "path_motion",
            "image_change",
            "zone_anchor",
        }:
            raise ModelValidationError("invalid_selection_source")
        if self.evidence_expired_at is not None and (
            not math.isfinite(self.evidence_expired_at)
            or self.evidence_expired_at < self.created_at
        ):
            raise ModelValidationError("invalid_evidence_expiry")
        if self.prompt_version is not None and not SAFE_ID.fullmatch(
            self.prompt_version
        ):
            raise ModelValidationError("invalid_prompt_version")
        if self.classification is not None and not SAFE_CLASSIFICATION.fullmatch(
            self.classification
        ):
            raise ModelValidationError("invalid_classification")
        if self.description is not None and (
            not self.description.strip() or len(self.description) > 500
        ):
            raise ModelValidationError("invalid_description")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not isinstance(self.confidence, int)
            or not 0 <= self.confidence <= 100
        ):
            raise ModelValidationError("invalid_confidence")
        for delivery_value in (self.delivery_attempt_id, self.completion_kind):
            if delivery_value is not None and not SAFE_ID.fullmatch(delivery_value):
                raise ModelValidationError("invalid_delivery_identity")
        if self.opening_side not in {"inside", "outside", "unknown"}:
            raise ModelValidationError("invalid_opening_side")
        if self.door_remained_open is not None and not isinstance(
            self.door_remained_open, bool
        ):
            raise ModelValidationError("invalid_door_state")
        if not all(
            isinstance(value, bool)
            for value in (
                self.contact_seen_open,
                self.contact_seen_close,
                self.contact_reopened,
            )
        ):
            raise ModelValidationError("invalid_door_state")
        if self.door_closed_at is not None and (
            not math.isfinite(self.door_closed_at)
            or self.door_closed_at < self.created_at
        ):
            raise ModelValidationError("invalid_activity_timestamp")
        earliest_doorbell = max(0.0, self.created_at - 120)
        if self.doorbell_at is not None and (
            not math.isfinite(self.doorbell_at) or self.doorbell_at < earliest_doorbell
        ):
            raise ModelValidationError("invalid_activity_timestamp")
        if (
            len(self.doorbell_times) > 100
            or tuple(sorted(set(self.doorbell_times))) != self.doorbell_times
            or any(
                not math.isfinite(timestamp) or timestamp < earliest_doorbell
                for timestamp in self.doorbell_times
            )
            or (
                self.doorbell_at is not None
                and (
                    not self.doorbell_times
                    or self.doorbell_at != self.doorbell_times[0]
                )
            )
        ):
            raise ModelValidationError("invalid_doorbell_times")
        previous_time = -1.0
        for occurred_at, zones in self.zone_updates:
            if (
                not math.isfinite(occurred_at)
                or occurred_at < previous_time
                or len(zones) != len(set(zones))
                or any(not SAFE_ID.fullmatch(zone) for zone in zones)
            ):
                raise ModelValidationError("invalid_zone_updates")
            previous_time = occurred_at
        if len(self.detection_zone_updates) > 1000:
            raise ModelValidationError("invalid_detection_zone_updates")
        previous_detection_key: tuple[float, str] | None = None
        for detection_id, occurred_at, zones in self.detection_zone_updates:
            detection_key = (occurred_at, detection_id)
            if (
                not SAFE_ID.fullmatch(detection_id)
                or not math.isfinite(occurred_at)
                or occurred_at < 0
                or detection_id not in self.detection_ids
                or len(zones) != len(set(zones))
                or any(not SAFE_ID.fullmatch(zone) for zone in zones)
                or (
                    previous_detection_key is not None
                    and detection_key <= previous_detection_key
                )
            ):
                raise ModelValidationError("invalid_detection_zone_updates")
            previous_detection_key = detection_key
        deadlines = [
            value
            for value in (self.association_deadline, self.finalization_deadline)
            if value is not None
        ]
        if any(
            not math.isfinite(value) or value < self.created_at for value in deadlines
        ):
            raise ModelValidationError("invalid_deadline")
        if (
            self.association_deadline is not None
            and self.finalization_deadline is not None
            and self.association_deadline > self.finalization_deadline
        ):
            raise ModelValidationError("invalid_deadline_order")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["source"] = self.source.value
        payload["stage"] = self.stage.value
        payload["processing_mode"] = self.processing_mode.value
        return payload

    def identity(self) -> tuple[object, ...]:
        return (
            self.activity_id,
            self.entry_id,
            self.source,
            self.processing_mode,
            self.created_at,
            self.camera,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Self:
        allowed = {
            "schema_version",
            "activity_id",
            "entry_id",
            "source",
            "stage",
            "processing_mode",
            "created_at",
            "updated_at",
            "camera",
            "error_code",
            "association_deadline",
            "finalization_deadline",
            "review_ids",
            "detection_ids",
            "zone_updates",
            "detection_zone_updates",
            "opening_side",
            "door_closed_at",
            "door_remained_open",
            "contact_seen_open",
            "contact_seen_close",
            "contact_reopened",
            "doorbell_at",
            "doorbell_times",
            "claimed_side_effects",
            "evidence_mode",
            "evidence_revision",
            "evidence_path",
            "evidence_media_url",
            "sample_times",
            "selection_source",
            "evidence_expired_at",
            "prompt_version",
            "classification",
            "description",
            "confidence",
            "delivery_attempt_id",
            "completion_kind",
        }
        if set(payload) - allowed:
            raise ModelValidationError("unknown_activity_field")
        try:
            source = ActivitySource(payload["source"])
        except (KeyError, ValueError) as exc:
            raise ModelValidationError("invalid_source") from exc
        try:
            stage = ActivityStage(payload["stage"])
        except (KeyError, ValueError) as exc:
            raise ModelValidationError("invalid_stage") from exc
        try:
            mode = ProcessingMode(payload["processing_mode"])
        except (KeyError, ValueError) as exc:
            raise ModelValidationError("invalid_processing_mode") from exc
        try:
            return cls(
                schema_version=int(payload["schema_version"]),
                activity_id=str(payload["activity_id"]),
                entry_id=str(payload["entry_id"]),
                source=source,
                stage=stage,
                processing_mode=mode,
                created_at=float(payload["created_at"]),
                updated_at=float(payload["updated_at"]),
                camera=str(payload["camera"]),
                error_code=(
                    str(payload["error_code"])
                    if payload.get("error_code") is not None
                    else None
                ),
                association_deadline=(
                    float(payload["association_deadline"])
                    if payload.get("association_deadline") is not None
                    else None
                ),
                finalization_deadline=(
                    float(payload["finalization_deadline"])
                    if payload.get("finalization_deadline") is not None
                    else None
                ),
                review_ids=tuple(str(value) for value in payload.get("review_ids", [])),
                detection_ids=tuple(
                    str(value) for value in payload.get("detection_ids", [])
                ),
                zone_updates=tuple(
                    (
                        float(update[0]),
                        tuple(str(zone) for zone in update[1]),
                    )
                    for update in payload.get("zone_updates", [])
                ),
                detection_zone_updates=tuple(
                    (
                        str(update[0]),
                        float(update[1]),
                        tuple(str(zone) for zone in update[2]),
                    )
                    for update in payload.get("detection_zone_updates", [])
                ),
                opening_side=str(payload.get("opening_side", "unknown")),
                door_closed_at=(
                    float(payload["door_closed_at"])
                    if payload.get("door_closed_at") is not None
                    else None
                ),
                door_remained_open=payload.get("door_remained_open"),
                contact_seen_open=payload.get("contact_seen_open", False),
                contact_seen_close=payload.get("contact_seen_close", False),
                contact_reopened=payload.get("contact_reopened", False),
                doorbell_at=(
                    float(payload["doorbell_at"])
                    if payload.get("doorbell_at") is not None
                    else None
                ),
                doorbell_times=tuple(
                    float(value)
                    for value in payload.get(
                        "doorbell_times",
                        (
                            [payload["doorbell_at"]]
                            if payload.get("doorbell_at") is not None
                            else []
                        ),
                    )
                ),
                claimed_side_effects=tuple(
                    str(value) for value in payload.get("claimed_side_effects", [])
                ),
                evidence_mode=(
                    str(payload["evidence_mode"])
                    if payload.get("evidence_mode") is not None
                    else None
                ),
                evidence_revision=int(payload.get("evidence_revision", 0)),
                evidence_path=(
                    str(payload["evidence_path"])
                    if payload.get("evidence_path") is not None
                    else None
                ),
                evidence_media_url=(
                    str(payload["evidence_media_url"])
                    if payload.get("evidence_media_url") is not None
                    else None
                ),
                sample_times=tuple(
                    float(value) for value in payload.get("sample_times", [])
                ),
                selection_source=(
                    str(payload["selection_source"])
                    if payload.get("selection_source") is not None
                    else None
                ),
                evidence_expired_at=(
                    float(payload["evidence_expired_at"])
                    if payload.get("evidence_expired_at") is not None
                    else None
                ),
                prompt_version=(
                    str(payload["prompt_version"])
                    if payload.get("prompt_version") is not None
                    else None
                ),
                classification=(
                    str(payload["classification"])
                    if payload.get("classification") is not None
                    else None
                ),
                description=(
                    str(payload["description"])
                    if payload.get("description") is not None
                    else None
                ),
                confidence=(
                    int(payload["confidence"])
                    if payload.get("confidence") is not None
                    else None
                ),
                delivery_attempt_id=(
                    str(payload["delivery_attempt_id"])
                    if payload.get("delivery_attempt_id") is not None
                    else None
                ),
                completion_kind=(
                    str(payload["completion_kind"])
                    if payload.get("completion_kind") is not None
                    else None
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ModelValidationError):
                raise
            raise ModelValidationError("invalid_activity") from exc


def door_activity_id(entry_id: str, opened_at: float) -> str:
    return _derived_id("door", entry_id, str(round(opened_at * 1000)))


def review_activity_id(entry_id: str, camera: str, review_id: str) -> str:
    return _derived_id("review", entry_id, camera, review_id)


def attempt_activity_id(activity_id: str, attempt: int) -> str:
    if attempt < 1:
        raise ModelValidationError("invalid_attempt")
    return _derived_id(activity_id, "attempt", str(attempt))


def _derived_id(*parts: str) -> str:
    value = "_".join(parts)
    if not SAFE_ID.fullmatch(value):
        raise ModelValidationError("invalid_derived_id")
    return value


def media_key(activity_id: str, revision: int) -> str:
    if revision < 1 or not SAFE_ID.fullmatch(activity_id):
        raise ModelValidationError("invalid_media_key")
    return f"media:{activity_id}:{revision}"


def analysis_key(activity_id: str, scene_mode: str, prompt_version: str) -> str:
    """Return the cache key for one scene's analysis of one activity.

    The scene is part of the key, not just the prompt version. One activity can
    legitimately be analysed under different scenes, and two scenes may share a
    version string -- without the scene in the key, the second analysis would be
    suppressed as already-done and would inherit the first scene's answer.
    """
    if (
        not SAFE_ID.fullmatch(activity_id)
        or not SAFE_ID.fullmatch(scene_mode)
        or not SAFE_ID.fullmatch(prompt_version)
    ):
        raise ModelValidationError("invalid_analysis_key")
    return f"analysis:{activity_id}:{scene_mode}:{prompt_version}"


def delivery_key(activity_id: str, attempt_id: str) -> str:
    if not SAFE_ID.fullmatch(activity_id) or not SAFE_ID.fullmatch(attempt_id):
        raise ModelValidationError("invalid_delivery_key")
    return f"delivery:{activity_id}:{attempt_id}"


def retry_is_safe(error_code: str) -> bool:
    return error_code in {
        "evidence_incomplete",
        "frigate_unavailable",
        "unexpected_content_type",
        "invalid_json",
        "media_retry_exhausted",
        "door_open_too_long",
    }
