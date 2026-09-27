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
# 分类值的最大长度。与下面的正则同源，不在别处再写一遍字面量：`scenes.py` 的
# 标签名校验要用同一个界，两处各写一个数字就会漂移。
#
# 刻意不复用 `SAFE_ID` 的 192——那两个数字只是碰巧相等，收紧其中一个不应该
# 牵动另一个。
MAX_CLASSIFICATION_LENGTH = 192
# 分类值允许中文：用户要能写「宠物」而不必被迫写 ASCII。
# 比 SAFE_ID 宽，但仍拒绝控制字符与各类行终止符——那些会真的破坏通知显示和
# 日志。除 ASCII 控制字符（含 DEL）外还要排除 C1 区（\x7f-\x9f）与 U+2028/
# U+2029：后两者是 JavaScript 的行终止符，会真的终止 JS 语句，而分类值会
# 流向 HA 前端模板与蓝图 Jinja2 模板。
#
# 这是分类值的**唯一**字符集定义：`parse_scene_labels` 直接引用它，所以解析器
# 接受的标签名与存储层接受的值不会漂移。漂移的后果不在解析处显现——畸形标签要
# 等到写入 `ActivityRecord` 时才抛错，那时 provider 已经计费，而且这个异常不是
# `VisionError`，会落进通用分支变成 `analysis_outcome_unknown`。
SAFE_CLASSIFICATION = re.compile(
    rf"^[^\x00-\x1f\x7f-\x9f\u2028\u2029]{{1,{MAX_CLASSIFICATION_LENGTH}}}$"
)
# One person box per MQTT update is cheap (four floats), but a door cycle can stay
# open for 30 minutes and Frigate re-publishes on every position change. The cap
# bounds a single activity's stored evidence; 1000 is what the per-detection zone
# sequence already uses for the same reason.
MAX_BOX_UPDATES = 1000


class ModelValidationError(ValueError):
    """A persisted or incoming model is invalid."""


def is_person_box(value: Any) -> bool:
    """Whether `value` is a usable normalised person box.

    A box is `[x, y, w, h]` as fractions of the frame. Zero or negative width or
    height covers no pixels, so `crop_person_box` would raise on it; rejecting it
    here keeps that failure at the boundary rather than at crop time.

    Shared by the model and the MQTT parser on purpose. If the parser accepted
    something the model rejects, the model's exception would be caught by the
    event parser's own handler and **the whole event would be lost** -- a door
    cycle would silently miss a zone update because one box was odd.
    """
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return False
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        return False
    return all(math.isfinite(item) for item in value) and value[2] > 0 and value[3] > 0


def person_box(value: Any) -> tuple[float, float, float, float] | None:
    """Return a validated box as a float tuple, or None when unusable.

    Used on the read path, where absent and unreadable both mean "no box" rather
    than an error.
    """
    if not is_person_box(value):
        return None
    return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))


def _required_box(value: Any) -> tuple[float, float, float, float]:
    """Coerce a persisted box, rejecting an unreadable one.

    A stored `box_updates` entry is evidence this integration wrote itself, so a
    malformed one is corruption rather than a Frigate quirk. Raising here makes
    `ActivityRecord.from_dict` report it instead of storing a hole.
    """
    box = person_box(value)
    if box is None:
        raise ModelValidationError("invalid_box_updates")
    return box


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
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.manual, bool):
            raise ModelValidationError("invalid_manual_flag")
        if self.box is not None and not is_person_box(self.box):
            raise ModelValidationError("invalid_box")
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
        if self.box is None:
            # Omitted rather than written as null: `BufferedIngress.buffer_id` is a
            # hash of this dict, and `ActivityStore.async_load` re-derives each
            # buffer_id and requires it to match the stored key. A new key would
            # change every id, so a restart would reject its own buffer as
            # `store_identity_mismatch` -- for messages that predate this field,
            # that is every message currently undecided.
            payload.pop("box", None)
        else:
            payload["box"] = [float(value) for value in self.box]
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
            "box",
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
                # A legacy payload has no box key at all; `None` is also what an
                # unreadable value degrades to, so both load the same way.
                box=person_box(payload.get("box")),
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
    box_updates: tuple[tuple[float, tuple[float, float, float, float]], ...] = ()
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
        # Strictly increasing, unlike zone_updates which merges same-timestamp
        # entries into a set. A box is a single value with no union, so two
        # records at one instant would be ambiguous: "the box at t" would depend
        # on which one a reader picked, and the crop would land on the wrong
        # moment. The writer replaces instead (see ActivityStore.async_merge_context).
        if len(self.box_updates) > MAX_BOX_UPDATES:
            raise ModelValidationError("invalid_box_updates")
        previous_box_time = -1.0
        for occurred_at, box in self.box_updates:
            if (
                not math.isfinite(occurred_at)
                or occurred_at < 0
                or occurred_at <= previous_box_time
                or not is_person_box(box)
            ):
                raise ModelValidationError("invalid_box_updates")
            previous_box_time = occurred_at
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
            "box_updates",
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
                # Strict here, unlike the MQTT read path: a stored box that cannot
                # be read means the persisted evidence is corrupt, and silently
                # dropping it would hide that behind a missing close-up crop.
                box_updates=tuple(
                    (
                        float(update[0]),
                        _required_box(update[1]),
                    )
                    for update in payload.get("box_updates", [])
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


# The prefix `async_request` builds its error code from, and the statuses it is
# safe to try again on its own.
#
# 502/503/504 are the gateway-and-availability family: 503 is "busy, come back
# later", 502/504 are an upstream that timed out or answered with nonsense. In all
# three the provider is telling us it did not process the request, so a retry
# cannot double-bill and cannot duplicate a side effect.
#
# Deliberately not "every 5xx": a 500 is the provider admitting an internal fault
# with no statement about whether the request was processed, which is the same
# uncertainty that makes `analysis_outcome_unknown` unsafe to replay. It stays
# out of the automatic loop and is left to the explicit, human-initiated retry.
PROVIDER_STATUS_PREFIX = "provider_http_"
AUTOMATIC_RETRY_STATUSES = frozenset({502, 503, 504})

# The status meaning "you are over your quota" -- measured here as a *daily*
# free-tier cap (`...free_tier_requests, limit: 20`).
#
# Replayable by hand but never retried automatically, and the two halves have
# different reasons. It is a provider-side, request-independent failure, like a
# 5xx, so nothing was billed and a replay is safe. But it is not a burst that a
# 26-second backoff can outlast: the window is a day, so an automatic retry would
# burn the next window's budget instead of recovering. A person who can see the
# quota and decide when to try again is the right actor.
QUOTA_STATUS = 429


def provider_status(error_code: str) -> int | None:
    """Read the HTTP status back out of a `provider_http_<status>` code.

    Returns None for anything that is not that shape, so a caller cannot mistake
    an unrelated code for a status.
    """
    if not error_code.startswith(PROVIDER_STATUS_PREFIX):
        return None
    tail = error_code[len(PROVIDER_STATUS_PREFIX) :]
    if not tail.isdigit():
        return None
    return int(tail)


def is_server_error(error_code: str) -> bool:
    """Whether the provider answered 5xx -- a fault on its side, not the request's."""
    status = provider_status(error_code)
    return status is not None and 500 <= status <= 599


def is_provider_side_failure(error_code: str) -> bool:
    """Whether the provider itself failed, so the request was never processed.

    Covers 5xx and 429. Both are the provider refusing to do the work for a reason
    that has nothing to do with the request: an internal fault, an overloaded
    gateway, or an exhausted quota. Nothing is billed in any of them, so a replay
    cannot duplicate a side effect -- which is what makes them safe to offer to a
    human, and what distinguishes them from `analysis_outcome_unknown`.

    This is the predicate behind the "the provider is failing and activities are
    being lost" repair, so it must be at least as wide as anything that can
    silently discard an activity.
    """
    return is_server_error(error_code) or provider_status(error_code) == QUOTA_STATUS


def is_automatically_retryable(error_code: str) -> bool:
    """Whether a failed analysis may be tried again without a human deciding."""
    status = provider_status(error_code)
    return status is not None and status in AUTOMATIC_RETRY_STATUSES


def retry_is_safe(error_code: str) -> bool:
    return (
        error_code
        in {
            "evidence_incomplete",
            "frigate_unavailable",
            "unexpected_content_type",
            "invalid_json",
            "media_retry_exhausted",
            "door_open_too_long",
            # The provider was never reached, so nothing can have been billed.
            # Excluded from this set until now, which left a connection failure
            # as the one unrecoverable transient error: it could not be replayed
            # even by hand.
            "provider_unavailable",
        }
        # Any 5xx. Wider than the automatic set on purpose: a person asking for
        # one more attempt has weighed the cost, and the worst case is a single
        # wasted call rather than a loop.
        or is_server_error(error_code)
        # An exhausted quota is the same shape of decision: nothing was billed, so
        # the replay is safe, and only a human can judge when the window resets.
        # Without this the provider's own "Please retry in 39s" advice pointed at a
        # route the integration refused.
        or provider_status(error_code) == QUOTA_STATUS
    )
