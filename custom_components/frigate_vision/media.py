"""Deterministic evidence planning and local contact-sheet generation."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from homeassistant.core import HomeAssistant
from PIL import Image, ImageChops, ImageOps, ImageStat, UnidentifiedImageError

from .correlation import ZoneRoles, anchor_sequence, infer_direction
from .frigate import FrigateApiError
from .media_source import DATA_MEDIA_REGISTRY
from .models import ActivityRecord, ActivitySource, ActivityStage, media_key
from .pathing import (
    InvalidPathData,
    MotionPath,
    parse_path_data,
    probe_gap_times,
    select_motion_times,
)
from .store import ActivityStore


class MediaError(RuntimeError):
    """Evidence cannot be generated without fabricating a frame."""


@dataclass(frozen=True, slots=True)
class EventWindow:
    event_id: str
    start_time: float
    end_time: float


@dataclass(frozen=True, slots=True)
class EvidencePlan:
    mode: str
    sample_times: tuple[float, ...] = ()
    first_time: float = 0
    change_candidates: tuple[float, ...] = ()
    last_time: float = 0
    postroll_time: float = 0
    selection_source: str | None = None
    motion_times: tuple[float, ...] = ()


class FrigateMediaClient(Protocol):
    async def async_get_event(self, event_id: str, camera: str) -> dict[str, Any]: ...

    async def async_get_recordings(
        self, camera: str, after: float, before: float
    ) -> list[dict[str, Any]]: ...

    async def async_get_snapshot(
        self, camera: str, timestamp: float, height: int
    ) -> bytes: ...


def _strict_three(values: Sequence[float]) -> tuple[float, float, float]:
    if len(values) < 2:
        raise MediaError("direction_ambiguous")
    start = float(values[0])
    end = float(values[-1])
    middle = float(values[len(values) // 2]) if len(values) >= 3 else (start + end) / 2
    if not start < middle < end:
        middle = (start + end) / 2
    if not start < middle < end:
        raise MediaError("invalid_sample_order")
    return start, middle, end


def _visible_sample_times(
    updates: Sequence[tuple[float, Sequence[str]]], roles: ZoneRoles
) -> list[float]:
    """Prefer anchor points, then any frame with a visible subject."""
    anchors = sorted({anchor.occurred_at for anchor in anchor_sequence(updates, roles)})
    if len(anchors) >= 2:
        return anchors
    visible = sorted({occurred_at for occurred_at, zones in updates if zones})
    if len(visible) >= 2:
        return visible
    return sorted({occurred_at for occurred_at, _ in updates})


def _conservative_door_plan(
    updates_by_event: Mapping[str, Sequence[tuple[float, Sequence[str]]]],
    roles: ZoneRoles,
) -> EvidencePlan | None:
    """Build the 3-frame fallback when direction anchors are inconclusive.

    REQ-6 forbids inventing a direction. The rules layer may still emit the
    time-ordered near/path/far candidates and let the vision model decide; a
    whole cycle must not fail only because zone updates stayed near the door.
    """
    times: list[float] = []
    for updates in updates_by_event.values():
        times.extend(_visible_sample_times(updates, roles))
    ordered = sorted({value for value in times})
    if len(ordered) < 3:
        return None
    return EvidencePlan(
        mode="door_single",
        sample_times=_strict_three(ordered),
        selection_source="zone_anchor",
    )


def plan_evidence(
    record: ActivityRecord,
    events: Mapping[str, EventWindow],
    roles: ZoneRoles,
    motion_paths: Mapping[str, MotionPath] | None = None,
) -> EvidencePlan:
    """Plan real timestamps without assigning visual semantics."""
    if record.source in {
        ActivitySource.STANDALONE_REVIEW,
        ActivitySource.MANUAL_REVIEW,
    }:
        if not events or set(events) != set(record.detection_ids):
            raise MediaError("event_window_missing")
        review_start = record.created_at
        review_end = record.updated_at
        person_start = min(event.start_time for event in events.values())
        person_end = max(event.end_time for event in events.values())
        if (
            review_end <= review_start
            or person_end <= person_start
            or person_end <= review_start
            or person_start >= review_end
        ):
            raise MediaError("invalid_review_time")
        person_duration = person_end - person_start
        edge = min(0.2, person_duration * 0.05)
        first = person_start + edge
        last = person_end - edge
        candidates = tuple(
            first + (last - first) * ratio / 10 for ratio in range(1, 10)
        )
        if len(candidates) != 9 or not first < last:
            raise MediaError("review_too_short")
        if motion_paths:
            paths = [
                motion_paths[event_id]
                for event_id in events
                if event_id in motion_paths
            ]
            selected = select_motion_times(paths, lower=first, upper=last)
            if selected is not None:
                return EvidencePlan(
                    mode="review_six",
                    first_time=first,
                    last_time=last,
                    postroll_time=person_end + 2.8,
                    selection_source="path_motion",
                    motion_times=selected,
                )
        return EvidencePlan(
            mode="review_six",
            first_time=first,
            change_candidates=candidates,
            last_time=last,
            postroll_time=person_end + 2.8,
            selection_source="image_change",
        )

    if record.source is not ActivitySource.DOOR_CYCLE:
        raise MediaError("unsupported_activity_source")
    if len(record.detection_ids) not in {1, 2}:
        raise MediaError("ambiguous_segment_count")
    tracks: dict[str, list[tuple[float, tuple[str, ...]]]] = {
        event_id: [] for event_id in record.detection_ids
    }
    for event_id, occurred_at, zones in record.detection_zone_updates:
        if event_id in tracks:
            tracks[event_id].append((occurred_at, zones))
    if set(events) != set(record.detection_ids):
        raise MediaError("event_window_missing")

    if len(record.detection_ids) == 2:
        segments: list[tuple[float, float, float]] = []
        ordered_ids = sorted(
            record.detection_ids, key=lambda value: events[value].start_time
        )
        directions: list[str] = []
        for event_id in ordered_ids:
            updates = sorted(tracks[event_id])
            observations = [value[0] for value in updates]
            window = events[event_id]
            if any(
                value < window.start_time or value > window.end_time
                for value in observations
            ):
                raise MediaError("event_timeline_outside_window")
            distinct = sorted(set(observations))
            if len(distinct) < 2:
                directions.append("ambiguous")
                continue
            directions.append(infer_direction(anchor_sequence(updates, roles)))
            segments.append(
                _strict_three(
                    (
                        max(window.start_time, distinct[0]),
                        *distinct[1:-1],
                        min(window.end_time, distinct[-1]),
                    )
                )
            )
        if directions == ["outbound", "inbound"] and len(segments) == 2:
            samples = tuple(value for segment in segments for value in segment)
            if tuple(sorted(set(samples))) == samples:
                return EvidencePlan(
                    mode="door_roundtrip",
                    sample_times=samples,
                    selection_source="zone_anchor",
                )
        fallback = _conservative_door_plan(tracks, roles)
        if fallback is None:
            raise MediaError("direction_ambiguous")
        return fallback

    event_id = record.detection_ids[0]
    updates = sorted(tracks[event_id])
    direction = infer_direction(anchor_sequence(updates, roles))
    times = _visible_sample_times(updates, roles)
    window = events[event_id]
    if any(value < window.start_time or value > window.end_time for value in times):
        raise MediaError("event_timeline_outside_window")
    if direction in {"outbound", "inbound"}:
        return EvidencePlan(
            mode="door_single",
            sample_times=_strict_three(times),
            selection_source="zone_anchor",
        )
    if direction != "roundtrip" or len(times) < 3:
        fallback = _conservative_door_plan({event_id: updates}, roles)
        if fallback is None:
            raise MediaError("direction_ambiguous")
        return fallback
    far_indexes = [
        index
        for index, (_, zones) in enumerate(updates)
        if roles.classify(zones) == "far"
    ]
    if not far_indexes:
        raise MediaError("direction_ambiguous")
    pivot_index = far_indexes[0]
    # `times` is the anchor sequence, where consecutive same-role samples are
    # collapsed, while `pivot_index` was computed over `updates`, which keeps
    # every sample. Indexing one list with the other's position lands early
    # whenever a role repeats (a duplicated far sample, for instance), so the
    # pivot is located in the list actually being sliced.
    pivot_time = updates[pivot_index][0]
    if pivot_time not in times:
        raise MediaError("direction_ambiguous")
    pivot_index = times.index(pivot_time)
    before_times = times[: pivot_index + 1]
    after_times = times[pivot_index + 1 :]
    # Two anchors per leg are enough: _strict_three derives the middle frame.
    # Demanding three rejected legs the helper handles, failing cycles that had
    # a usable outbound/return split.
    if len(before_times) < 2 or len(after_times) < 2:
        raise MediaError("insufficient_reversal_anchors")
    before = _strict_three(before_times)
    after = _strict_three(after_times)
    samples = (*before, *after)
    if tuple(sorted(set(samples))) != samples:
        raise MediaError("invalid_sample_order")
    return EvidencePlan(
        mode="door_roundtrip",
        sample_times=samples,
        selection_source="zone_anchor",
    )


def recordings_cover(
    sample_times: Sequence[float], recordings: Sequence[Mapping[str, Any]]
) -> bool:
    try:
        intervals = [
            (float(item["start_time"]), float(item["end_time"])) for item in recordings
        ]
    except KeyError, TypeError, ValueError:
        return False
    return all(
        any(start <= timestamp <= end for start, end in intervals)
        for timestamp in sample_times
    )


def select_review_change_frames(
    frames: Sequence[tuple[float, Path]], *, count: int = 3
) -> list[tuple[float, Path]]:
    if count <= 0 or len(frames) < count + 1:
        # Too few frames to score anything: a frame-availability problem, not a
        # statement about the scene.
        raise MediaError("insufficient_review_candidates")
    scored: list[tuple[float, float, Path, bytes]] = []
    previous: Image.Image | None = None
    try:
        for timestamp, path in sorted(frames):
            with Image.open(path) as image:
                current = ImageOps.fit(image.convert("L"), (64, 36))
                current.load()
            if previous is not None:
                score = float(
                    ImageStat.Stat(ImageChops.difference(previous, current)).mean[0]
                )
                if score >= 1.0:
                    scored.append((score, timestamp, path, current.tobytes()))
            previous = current
    except (OSError, UnidentifiedImageError) as exc:
        raise MediaError("invalid_review_candidate_frame") from exc
    unique: list[tuple[float, float, Path, bytes]] = []
    fingerprints: set[bytes] = set()
    for item in sorted(scored, key=lambda value: (-value[0], value[1])):
        if item[3] in fingerprints:
            continue
        fingerprints.add(item[3])
        unique.append(item)
        if len(unique) == count:
            break
    selected = sorted(unique, key=lambda item: item[1])
    if len(selected) != count:
        # Every frame was downloaded, but none of them differs from its
        # neighbour: the camera recorded a static scene, so there is no visible
        # change to build evidence from. Distinct from having too few frames.
        raise MediaError("insufficient_visible_change")
    return [(timestamp, path) for _, timestamp, path, _ in selected]


def _change_scores(
    frames: Sequence[tuple[float, Path]],
) -> list[tuple[float, float]]:
    """Score each frame by how much it differs from its predecessor.

    The same measure the fallback selector uses, applied to a probed stretch
    rather than to a whole review: a probe is worth taking where the picture
    changed, and that is what this reports. The first frame has no predecessor
    and is scored against the second, so every probe carries a number and none
    is silently dropped for being first.

    Returns (timestamp, score) in the order given. An unreadable frame is
    skipped rather than failing the activity -- probing is an improvement
    attempt, and the caller keeps its original picks if this yields nothing.
    """
    scored: list[tuple[float, float]] = []
    previous: Image.Image | None = None
    try:
        for timestamp, path in frames:
            current = _grayscale_signature(path)
            if previous is None:
                previous = current
                continue
            scored.append((timestamp, _mean_difference(previous, current)))
            previous = current
    except MediaError:
        return []
    return scored


def pair_times_with_roles(
    plan: EvidencePlan, probes: Sequence[float]
) -> tuple[tuple[float, ...], tuple[str, ...]]:
    """Return the sheet's timestamps and their roles, sorted together.

    A pure function on purpose. The pairing cannot be tested where it is used --
    inside an async method that fetches frames and writes files -- and a mutation
    that sorted the times while leaving the roles in insertion order passed every
    test, because nothing observed the two lists as a pair.

    That mutation is a real defect, not a hypothetical one:
    `_nudge_uncovered_frames` reads `roles[index]` to decide which frames the sheet
    cannot be built without. `first`, `last` and `postroll` are required, a probe
    is not, and probes land *inside* the motion span -- so a mispaired role lets a
    required frame be treated as droppable.

    Probes are included as ordinary cells; they are added to the sheet rather than
    replacing a motion pick, which is what grows it from 2x3 to 3x3.
    """
    paired = sorted(
        [
            (plan.first_time, "first"),
            *((moment, "motion") for moment in plan.motion_times),
            *((moment, "probe") for moment in probes),
            (plan.last_time, "last"),
            (plan.postroll_time, "postroll"),
        ]
    )
    return (
        tuple(moment for moment, _ in paired),
        tuple(role for _, role in paired),
    )


def build_contact_sheet(
    frame_paths: Sequence[Path],
    output_path: Path,
    *,
    columns: int = 3,
    cell_size: tuple[int, int] = (640, 360),
) -> None:
    if columns <= 0:
        raise MediaError("invalid_frame_count")
    # The row count follows the frame count rather than being fixed at two.
    # A nine-cell sheet is the same 3-column layout with three rows, and hardcoding
    # two would drop the third row silently -- which is how a malformed fixture of
    # mine once reported an inflated result. A partial final row is rejected
    # outright for the same reason.
    if not frame_paths or len(frame_paths) % columns:
        raise MediaError("invalid_frame_count")
    width, height = cell_size
    sheet = Image.new("RGB", (width * columns, height * (len(frame_paths) // columns)))
    try:
        for index, path in enumerate(frame_paths):
            try:
                with Image.open(path) as source:
                    if source.format != "JPEG":
                        raise MediaError("frame_not_jpeg")
                    source.load()
                    contained = ImageOps.contain(source.convert("RGB"), cell_size)
            except (OSError, UnidentifiedImageError) as exc:
                raise MediaError("frame_decode_failed") from exc
            cell = Image.new("RGB", cell_size)
            cell.paste(
                contained,
                ((width - contained.width) // 2, (height - contained.height) // 2),
            )
            sheet.paste(cell, ((index % columns) * width, (index // columns) * height))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(output_path, "JPEG", quality=85, optimize=True)
    finally:
        sheet.close()


def _grayscale_signature(path: Path) -> Image.Image:
    """Load a frame reduced to the comparison resolution."""
    try:
        with Image.open(path) as image:
            current = ImageOps.fit(image.convert("L"), (64, 36))
            current.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise MediaError("frame_decode_failed") from exc
    return current


def _mean_difference(left: Image.Image, right: Image.Image) -> float:
    return float(ImageStat.Stat(ImageChops.difference(left, right)).mean[0])


def validate_unique_frames(frame_paths: Sequence[Path]) -> None:
    """Reject exact and near-duplicate frames across the final artifact."""
    images: list[Image.Image] = []
    for path in frame_paths:
        current = _grayscale_signature(path)
        for previous in images:
            if _mean_difference(previous, current) < UNIQUE_FRAME_THRESHOLD:
                raise MediaError("duplicate_evidence_frame")
        images.append(current)


# Two frames closer than this mean greyscale distance would tell the vision
# model it saw two moments when it saw one.
UNIQUE_FRAME_THRESHOLD = 1.0

# The widest unobserved stretch the sheet will tolerate before probing it.
#
# Measured on this deployment: the motion picks leave holes of 12-39s, and every
# hole wider than this that was probed contained real change (mean greyscale
# difference 2.45-85.66, against a threshold of 1.0). So the holes are not empty
# corridors, and the detector is not seeing what the camera is.
#
# The bound is the whole economy of the feature: every probe costs a snapshot
# fetch, and a hole shorter than the shortest action worth seeing cannot be
# hiding one. A door opening, a hand-off, or a turn takes a few seconds, so a
# hole under this many seconds is already unlikely to swallow one whole.
MAX_SAMPLE_GAP = 12.0

# Instants tried inside the widest hole. Three probes span the hole without
# inviting a search: they are scored by the same change measure the fallback
# selector uses, and only the best-scoring ones can displace a motion pick.
GAP_PROBE_COUNT = 3

# Seconds between the last tracked person sample and the postroll frame. The
# postroll frame must stay after this point to keep showing the emptied scene.
POSTROLL_OFFSET_SECONDS = 2.8

# Offsets tried when a required frame falls inside a recording hole. Frigate
# writes recordings as fixed segments that can leave sub-second (sometimes
# seconds-long) gaps between them, and a frame the sheet needs cannot simply be
# discarded the way a change candidate can. A motion peak is a sampled instant
# rather than an exact moment, so shifting it by a second or two preserves its
# meaning while the neighbour bounds keep the frames spread apart.
RECORDING_NUDGE_OFFSETS = (
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -2.5,
    -3.0,
    -4.0,
    -5.0,
)

# The postroll frame means "the scene after the person left", so moving it
# later is always semantically safe; only moving it before the departure is not.
POSTROLL_FORWARD_OFFSETS = (6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0)

# Widest nudge the recordings query must anticipate.
MAX_RECORDING_NUDGE = max(
    [abs(value) for value in RECORDING_NUDGE_OFFSETS] + list(POSTROLL_FORWARD_OFFSETS)
)


# Offsets tried when a selected frame duplicates another. Small steps come
# first so the replacement stays as close as possible to the planned moment;
# the wider steps exist because a person standing still can keep the scene
# identical for many seconds.
UNIQUENESS_RETRY_OFFSETS = (
    0.25,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.5,
    4.0,
    5.0,
    6.0,
    8.0,
    11.0,
    15.0,
    -0.25,
    -0.5,
    -0.75,
    -1.0,
    -1.25,
    -1.5,
    -1.75,
    -2.0,
    -2.25,
    -2.5,
    -2.75,
    -3.0,
    -3.5,
    -4.0,
    -5.0,
    -6.0,
    -8.0,
    -11.0,
    -15.0,
)


# An IR-lit frame is strongly green-dominant with red suppressed. Measured on
# production frames: IR cells land near R-G = -150, colour frames near +2, and
# camera-transition frames near -35. The threshold sits between the IR frames
# and everything else so transitional (still legible) frames are kept.
INFRARED_RED_GREEN_THRESHOLD = -60.0

# A saturated frame carries no detail. The camera faces a lift door, and when
# that door opens the car's light overwhelms the sensor. Measured across every
# cell of the shadow-mode sheets: usable cells reach 14.7% near-white pixels and
# blown cells start at 42.7%, so the cut sits at the midpoint of that gap --
# about 15 points of margin either side.
OVEREXPOSED_WHITE_LEVEL = 235
OVEREXPOSED_WHITE_FRACTION = 0.30

# Offsets tried, in order, when a chosen timestamp yields an IR frame. The
# camera needs roughly two seconds to leave night mode after a person is
# detected, so small forward steps recover a colour frame without wandering far
# from the moment the plan asked for.
INFRARED_RETRY_OFFSETS = (
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    2.2,
    2.4,
    2.6,
    2.8,
    3.0,
    3.2,
    3.4,
    3.6,
    3.8,
    4.0,
    4.2,
    4.4,
    4.6,
    4.8,
    5.0,
    -0.1,
    -0.2,
    -0.3,
    -0.4,
    -0.5,
    -0.75,
    -1.0,
    -1.25,
    -1.5,
    -1.75,
    -2.0,
)


def frame_is_infrared(path: Path) -> bool:
    """Report whether a snapshot carries the camera's night-vision cast.

    The camera keeps an IR illuminator on in the dark and lags briefly behind
    person detection when switching back to colour, so the earliest sample of a
    review is routinely an IR frame: washed out, green, and low in detail.
    """
    try:
        with Image.open(path) as image:
            sample = ImageOps.fit(image.convert("RGB"), (64, 36))
            sample.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise MediaError("frame_decode_failed") from exc
    red, green, _blue = ImageStat.Stat(sample).mean
    return (red - green) < INFRARED_RED_GREEN_THRESHOLD


def frame_is_overexposed(path: Path) -> bool:
    """Report whether a snapshot is saturated and thus carries no evidence.

    The camera sits facing a lift door; when that door opens, the car's interior
    light shines straight into the lens and the sensor clips. The resulting frame
    is near-white with the detail burned out.

    The night-vision test does not catch this. It keys on a red/green imbalance,
    while a blown frame saturates all three channels together, so its red/green
    gap stays near zero and it passes as ordinary colour.

    Measured over the shadow-mode sheets: unusable cells had 42%-98% of pixels at
    or above the white level, while every usable cell stayed below 1%.

    A frame that is bright but still holds structure must survive, so the count
    requires *every* channel at the white level. A per-channel count would score
    a mainly-white frame highly even when dark edges give it real content, which
    is how a white door or wall -- ordinary here, and evidence in its own right
    -- would be discarded.

    The count runs at full resolution. Downsampling first, as the night-vision
    test does, averages saturated pixels together with their darker neighbours
    and pulls them back under the threshold: measured on a real sheet, a cell
    that was 42.7% saturated read as clean once reduced to 64x36. The colour cast
    survives averaging; clipped highlights do not.
    """
    try:
        with Image.open(path) as image:
            sample = image.convert("RGB")
            sample.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise MediaError("frame_decode_failed") from exc
    histogram = sample.histogram()
    pixels = sample.width * sample.height
    # A pixel is saturated only when all three channels are; the darkest channel
    # therefore decides, and it is the one whose histogram to read. Summing the
    # three would count a single saturated channel as a saturated pixel.
    darkest = min(
        sum(histogram[channel * 256 + OVEREXPOSED_WHITE_LEVEL : (channel + 1) * 256])
        for channel in range(3)
    )
    return (darkest / pixels) >= OVEREXPOSED_WHITE_FRACTION


class MediaManager:
    """Build and atomically register one local evidence artifact."""

    def __init__(
        self,
        hass: HomeAssistant,
        store: ActivityStore,
        client: FrigateMediaClient,
        root: Path,
        roles: ZoneRoles,
    ) -> None:
        self._hass = hass
        self._store = store
        self._client = client
        self._root = root.resolve()
        self._roles = roles
        self._locks: dict[str, asyncio.Lock] = {}

    async def async_build(self, activity_id: str) -> ActivityRecord:
        lock = self._locks.setdefault(activity_id, asyncio.Lock())
        async with lock:
            return await self._async_build_locked(activity_id)

    async def _async_build_locked(self, activity_id: str) -> ActivityRecord:
        record = self._store.get(activity_id)
        if record is None:
            raise MediaError("activity_missing")
        if record.stage is ActivityStage.EVIDENCE_READY:
            if record.evidence_path is None:
                raise MediaError("artifact_missing")
            expected = self._canonical_path(record)
            if Path(record.evidence_path) != expected:
                raise MediaError("invalid_output_path")
            metadata = expected.with_suffix(".json")
            recovered = await self._hass.async_add_executor_job(
                _read_existing, expected, metadata, record.activity_id
            )
            if recovered is None or recovered[:2] != (
                record.evidence_mode,
                record.sample_times,
            ):
                raise MediaError("artifact_missing")
            self._register(record, expected)
            return record
        if record.stage is not ActivityStage.SEALED:
            raise MediaError("stage_conflict")
        events: dict[str, EventWindow] = {}
        motion_paths: dict[str, MotionPath] = {}
        parse_paths = record.source in {
            ActivitySource.STANDALONE_REVIEW,
            ActivitySource.MANUAL_REVIEW,
        }
        for event_id in record.detection_ids:
            payload = await self._client.async_get_event(event_id, record.camera)
            start_time = float(payload["start_time"])
            end_time = float(payload["end_time"])
            events[event_id] = EventWindow(event_id, start_time, end_time)
            if not parse_paths:
                continue
            try:
                points = parse_path_data(payload)
            except InvalidPathData as exc:
                raise MediaError("invalid_path_data") from exc
            if points is not None:
                motion_paths[event_id] = MotionPath(start_time, end_time, points)
        plan = plan_evidence(record, events, self._roles, motion_paths)
        final = self._canonical_path(record)
        metadata = final.with_suffix(".json")
        recovered = await self._hass.async_add_executor_job(
            _read_existing, final, metadata, record.activity_id
        )
        if recovered is not None:
            mode, samples, stored_source = recovered
            if plan.motion_times:
                # A plan_version 3 artifact may have substituted an IR frame for
                # a neighbouring colour one, so the stored times can legitimately
                # differ from the freshly planned ones. Accept any set that keeps
                # the mode, source, frame count and ordering intact; otherwise
                # every restart would rebuild identical evidence.
                valid = (
                    mode == plan.mode
                    and stored_source == "path_motion"
                    and len(samples) == 6
                    and tuple(sorted(set(samples))) == samples
                )
            else:
                valid = mode == plan.mode and (
                    not plan.sample_times or samples == plan.sample_times
                )
            if valid:
                return await self._complete(record, final, mode, samples, stored_source)
            await self._hass.async_add_executor_job(_delete_artifact, final, self._root)

        times: tuple[float, ...]
        roles: tuple[str, ...]
        if plan.motion_times:
            times = (
                plan.first_time,
                *plan.motion_times,
                plan.last_time,
                plan.postroll_time,
            )
            roles = ("first", "motion", "motion", "motion", "last", "postroll")
        elif plan.sample_times:
            times = plan.sample_times
            roles = tuple("sample" for _ in times)
        else:
            times = (
                plan.first_time,
                *plan.change_candidates,
                plan.last_time,
                plan.postroll_time,
            )
            roles = (
                "first",
                *(("change",) * len(plan.change_candidates)),
                "last",
                "postroll",
            )
        requested = tuple(times)
        recordings = await self._client.async_get_recordings(
            record.camera,
            min(requested) - MAX_RECORDING_NUDGE,
            max(requested) + MAX_RECORDING_NUDGE,
        )
        # Look for evidence the path data cannot see, before anything is built.
        #
        # The detector reports where the *person* moved, and the middle cells are
        # chosen from that. But a door opening while the person stands still
        # moves no path point at all, so the sheet's largest hole can be exactly
        # where the decisive moment is. Measured here, every hole wider than
        # MAX_SAMPLE_GAP contained real change.
        #
        # The probes become *extra cells*, growing the sheet from 2x3 to 3x3,
        # rather than replacing a motion pick: spending extra cells on more
        # motion picks measured 16.4s worst gap against 11.5s for probed ones.
        # The probes are fetched into their own directory because the main
        # temporary directory is created further down.
        if plan.motion_times:
            probe_dir = Path(
                await self._hass.async_add_executor_job(
                    partial(tempfile.mkdtemp, dir=self._root)
                )
            )
            try:
                found = await self._async_probe_widest_gap(
                    record.camera,
                    probe_dir,
                    plan,
                    recordings,
                )
            finally:
                await self._hass.async_add_executor_job(
                    shutil.rmtree, probe_dir, True
                )
            if found:
                times, roles = pair_times_with_roles(plan, found)
                requested = times
        # A required frame inside a recording hole cannot be discarded the way a
        # change candidate can, so it is nudged to the nearest covered instant
        # first. Coverage is then checked on what actually must be present.
        requested, fixed, nudges = self._nudge_uncovered_frames(
            roles, requested, recordings
        )
        if not recordings_cover(tuple(fixed.values()), recordings):
            raise MediaError("recording_gap")
        await self._hass.async_add_executor_job(
            partial(self._root.mkdir, parents=True, exist_ok=True)
        )
        directory = await self._hass.async_add_executor_job(
            partial(tempfile.mkdtemp, dir=self._root)
        )
        temporary = Path(directory)
        try:
            covered = [
                (start, end) for start, end in self._coverage_intervals(recordings)
            ]

            def is_covered(timestamp: float) -> bool:
                return any(start <= timestamp <= end for start, end in covered)

            candidates: list[tuple[float, Path]] = []
            candidate_roles: list[str] = []
            for index, timestamp in enumerate(requested):
                if not is_covered(timestamp):
                    # A change candidate inside a recording hole: skip it, the
                    # selector has other points to choose from.
                    continue
                data = await self._client.async_get_snapshot(
                    record.camera, timestamp, 360
                )
                path = temporary / f"{index}.jpg"
                await self._hass.async_add_executor_job(path.write_bytes, data)
                candidates.append((timestamp, path))
                candidate_roles.append(roles[index])
            if plan.sample_times or plan.motion_times:
                selected = candidates
            else:
                # The search space is the change candidates only. Slicing by
                # position would let a dropped candidate pull `last` into the
                # selector's input, which then gets appended again as a
                # duplicate, so the roles are used to partition the frames.
                change_space = [
                    item
                    for item, role in zip(candidates, candidate_roles, strict=True)
                    if role == "change"
                ][:9]
                by_role = {
                    role: item
                    for item, role in zip(candidates, candidate_roles, strict=True)
                    if role != "change"
                }
                anchor = by_role["first"]
                changes = await self._hass.async_add_executor_job(
                    select_review_change_frames,
                    [anchor, *change_space],
                )
                selected = [
                    anchor,
                    *changes,
                    by_role["last"],
                    by_role["postroll"],
                ]
            selected, replacements = await self._async_replace_infrared(
                record.camera, temporary, selected, recordings
            )
            # After the night-vision repair, so a frame that is both IR and blown
            # is judged on its final form rather than repaired twice.
            selected, exposure_replacements = await self._async_replace_overexposed(
                record.camera, temporary, selected, recordings
            )
            selected, unique_replacements = await self._async_replace_duplicates(
                record.camera, temporary, selected, recordings
            )
            samples = tuple(timestamp for timestamp, _ in selected)
            if tuple(sorted(set(samples))) != samples:
                raise MediaError("invalid_sample_order")
            await self._hass.async_add_executor_job(
                validate_unique_frames, [path for _, path in selected]
            )
            sheet = temporary / "evidence.jpg"
            await self._hass.async_add_executor_job(
                build_contact_sheet, [path for _, path in selected], sheet
            )
            meta = temporary / "evidence.json"
            await self._hass.async_add_executor_job(
                meta.write_text,
                json.dumps(
                    {
                        "activity_id": record.activity_id,
                        "plan_version": 4,
                        "mode": plan.mode,
                        "selection_source": plan.selection_source,
                        "planned_times": list(requested),
                        "sample_times": samples,
                        "infrared_replacements": replacements,
                        "overexposure_replacements": exposure_replacements,
                        "uniqueness_replacements": unique_replacements,
                        "recording_nudges": nudges,
                    }
                ),
                "utf-8",
            )
            await self._hass.async_add_executor_job(
                partial(final.parent.mkdir, parents=True, exist_ok=True)
            )
            await self._hass.async_add_executor_job(os.replace, sheet, final)
            await self._hass.async_add_executor_job(os.replace, meta, metadata)
        finally:
            await self._hass.async_add_executor_job(shutil.rmtree, temporary, True)
        return await self._complete(
            record, final, plan.mode, samples, plan.selection_source
        )

    @staticmethod
    def _nudge_uncovered_frames(
        roles: Sequence[str],
        requested: Sequence[float],
        recordings: Sequence[Mapping[str, Any]],
    ) -> tuple[tuple[float, ...], dict[str, float], list[dict[str, Any]]]:
        """Move frames that fall in a recording hole to a covered instant.

        Frigate writes recordings as fixed-length segments and can leave small
        gaps between them. A frame the sheet needs cannot be dropped the way a
        change candidate can, so instead it shifts by a bounded offset that
        stays between its neighbours -- keeping the sheet strictly increasing
        and each frame near the moment the plan asked for.

        The postroll frame means "the scene after the person left", so moving it
        later is always safe; moving it earlier could cross back to before the
        departure, so it is only ever pushed forward.
        """
        intervals = MediaManager._coverage_intervals(recordings)

        def covered(timestamp: float) -> bool:
            return any(start <= timestamp <= end for start, end in intervals)

        result = list(requested)
        nudges: list[dict[str, Any]] = []
        for index, timestamp in enumerate(result):
            if covered(timestamp):
                continue
            offsets = (
                POSTROLL_FORWARD_OFFSETS
                if roles[index] == "postroll"
                else RECORDING_NUDGE_OFFSETS
            )
            lower = result[index - 1] if index > 0 else float("-inf")
            upper = result[index + 1] if index + 1 < len(result) else float("inf")
            for offset in offsets:
                candidate = timestamp + offset
                if not lower < candidate < upper or not covered(candidate):
                    continue
                result[index] = candidate
                nudges.append(
                    {
                        "index": index,
                        "role": roles[index],
                        "from": round(timestamp, 3),
                        "to": round(candidate, 3),
                        "offset": offset,
                    }
                )
                break

        fixed = {
            role: result[index]
            for index, role in enumerate(roles)
            if role in {"first", "last", "postroll"}
        }
        return tuple(result), fixed, nudges

    async def _async_probe_widest_gap(
        self,
        camera: str,
        temporary: Path,
        plan: EvidencePlan,
        recordings: Sequence[Mapping[str, Any]],
    ) -> tuple[float, ...] | None:
        """Instants worth adding as extra cells, found by visible change.

        The sheet's middle cells come from the person's motion path. The detector
        reports where the *person* moved, which is not where the *picture*
        changed: a door opening while the person stands still moves no path point
        at all, and the moment the entry door opened on this deployment fell
        inside a 33.3s hole for exactly that reason.

        These are *added* to the sheet rather than replacing a motion pick.
        Measured across 48 activities: spending the extra cells on more motion
        picks barely helps (mean worst gap 17.1s -> 16.4s) because a hole is by
        definition where the person was not moving, while three probed cells
        nearly halves it (17.1s -> 11.5s), improving 29 and worsening none.

        Returns None when there is nothing to add -- no hole wide enough, too few
        frames retrievable, or no visible change in the hole. A failure here must
        not cost the sheet: the caller keeps its six cells in every such case.
        """
        probes = probe_gap_times(
            (plan.first_time, *plan.motion_times, plan.last_time),
            max_gap=MAX_SAMPLE_GAP,
            probes=GAP_PROBE_COUNT,
            # The postroll cell sits beyond the activity on purpose, so the
            # stretch before it is not a hole to fill.
            within=(plan.first_time, plan.last_time),
        )
        if not probes:
            return None

        covered = self._coverage_intervals(recordings)
        fetchable = [
            moment
            for moment in probes
            if any(start <= moment <= end for start, end in covered)
        ]
        if len(fetchable) < GAP_PROBE_COUNT:
            # Only a full set of probes yields a whole number of rows. A short set
            # would leave a partial row, which cannot be laid out -- and padding
            # it with frames chosen for no reason would be worse than six cells.
            return None

        frames: list[tuple[float, Path]] = []
        for index, moment in enumerate(fetchable):
            try:
                data = await self._client.async_get_snapshot(camera, moment, 360)
            except (FrigateApiError, OSError):
                continue
            path = temporary / f"probe-{index}.jpg"
            await self._hass.async_add_executor_job(path.write_bytes, data)
            frames.append((moment, path))
        if len(frames) < GAP_PROBE_COUNT:
            return None

        # A night-vision probe is not comparable to a colour one, so a hole that
        # only yielded IR frames is left alone rather than trusted.
        usable = []
        for moment, path in frames:
            if await self._hass.async_add_executor_job(frame_is_infrared, path):
                continue
            usable.append(moment)
        if len(usable) < GAP_PROBE_COUNT:
            return None
        return tuple(sorted(usable))

    @staticmethod
    def _coverage_intervals(
        recordings: Sequence[Mapping[str, Any]],
    ) -> list[tuple[float, float]]:
        intervals: list[tuple[float, float]] = []
        for item in recordings:
            try:
                intervals.append((float(item["start_time"]), float(item["end_time"])))
            except KeyError, TypeError, ValueError:
                continue
        return intervals

    @staticmethod
    def _required_times(
        plan: EvidencePlan, requested: Sequence[float]
    ) -> tuple[float, ...]:
        """Return the timestamps the sheet cannot be built without.

        `first` and `last`/`postroll` are derived from the person window and
        always appear in the sheet; the change candidates do not, so a hole
        under one of them is recoverable. `sample_times` and `motion_times`
        are themselves the final selection and are all required.
        """
        if plan.sample_times:
            return plan.sample_times
        if plan.motion_times:
            return (
                plan.first_time,
                *plan.motion_times,
                plan.last_time,
                plan.postroll_time,
            )
        return (plan.first_time, plan.last_time, plan.postroll_time)

    async def _async_retry_frames(
        self,
        camera: str,
        temporary: Path,
        selected: Sequence[tuple[float, Path]],
        recordings: Sequence[Mapping[str, Any]],
        *,
        offsets: Sequence[float],
        tag: str,
        needed: Callable[[int, Path, list[tuple[float, Path]]], bool],
        acceptable: Callable[[Path, int, list[tuple[float, Path]]], bool],
    ) -> tuple[list[tuple[float, Path]], list[dict[str, Any]]]:
        """Re-fetch frames that fail `needed`, accepting the first good neighbour.

        Shared by the night-vision and uniqueness repairs: both pick a frame,
        find it unusable, and walk a fixed offset ladder for a nearby frame that
        passes. Each candidate must stay strictly between its neighbours so the
        sheet keeps increasing timestamps, must not reuse a taken timestamp, and
        must fall inside the recording coverage.

        A frame with no acceptable alternative is left in place; the caller
        decides what that means (the IR repair keeps it, the uniqueness repair
        lets the existing duplicate guard fail the activity).
        """
        covered: list[tuple[float, float]] = []
        for item in recordings:
            try:
                covered.append((float(item["start_time"]), float(item["end_time"])))
            except KeyError, TypeError, ValueError:
                continue

        result = list(selected)
        taken = {timestamp for timestamp, _ in selected}
        replacements: list[dict[str, Any]] = []

        # Process from the last frame backwards. Each frame's forward bound is
        # its successor's *current* position, so resolving the successor first
        # widens the room available here. Forward order would bound every frame
        # by its successor's original position, and a frame that must move far
        # (a night-vision frame waiting for colour, say) can be boxed in by a
        # successor that was itself about to move out of the way.
        order = list(range(len(selected) - 1, -1, -1))
        for index in order:
            timestamp, path = selected[index]
            if not await self._hass.async_add_executor_job(needed, index, path, result):
                continue
            lower = result[index - 1][0] if index > 0 else float("-inf")
            upper = result[index + 1][0] if index + 1 < len(result) else float("inf")
            # The ladder is a list of durations, but the room actually available
            # depends on where the neighbours landed, which is a repair result
            # rather than a planned value. A frame needing 4.20s that is offered
            # only 4.19s would be dropped for a margin far smaller than a frame
            # interval, so the tightest instant inside the bounds is also tried.
            ladder = list(offsets)
            if math.isfinite(upper):
                ladder.append(max(upper - timestamp - 0.01, 0.0))
            if math.isfinite(lower):
                ladder.append(min(lower - timestamp + 0.01, 0.0))
            for step, offset in enumerate(ladder):
                candidate = timestamp + offset
                if not lower < candidate < upper or candidate in taken:
                    continue
                if not any(start <= candidate <= end for start, end in covered):
                    continue
                try:
                    data = await self._client.async_get_snapshot(camera, candidate, 360)
                except OSError, FrigateApiError, MediaError:
                    # Retrying is a best-effort improvement: the frame we
                    # already hold is usable, and Frigate purges recordings in
                    # the background so a neighbouring instant can vanish.
                    # Letting that abort would trade an optional upgrade for the
                    # whole activity, so the next offset is simply tried.
                    continue
                trial = temporary / f"{tag}_{index}_{step}.jpg"
                await self._hass.async_add_executor_job(trial.write_bytes, data)
                if not await self._hass.async_add_executor_job(
                    acceptable, trial, index, result
                ):
                    continue
                # The candidate must be a different picture from its neighbours,
                # not merely a different timestamp. Frigate answers two nearby
                # instants with the same stored frame, so a timestamp bookkeeping
                # check passes while the sheet ends up with two identical cells
                # and the uniqueness guard then fails the whole activity.
                if await self._hass.async_add_executor_job(
                    self._collides_with_neighbours, index, trial, result
                ):
                    continue
                taken.discard(timestamp)
                taken.add(candidate)
                replacements.append(
                    {
                        "index": index,
                        "from": round(timestamp, 3),
                        "to": round(candidate, 3),
                        "offset": offset,
                        "reason": tag,
                    }
                )
                result[index] = (candidate, trial)
                break

        result.sort(key=lambda item: item[0])
        return result, replacements

    @staticmethod
    def _collides_with_neighbours(
        index: int, trial: Path, result: Sequence[tuple[float, Path]]
    ) -> bool:
        """Report whether a candidate repeats a neighbour's picture.

        Uses the same measure as the uniqueness guard, so a candidate this
        accepts cannot later be rejected by it. Only the immediate neighbours are
        compared: the sheet is sorted by time, so a candidate can only collide
        with what sits next to it.
        """
        try:
            candidate = _grayscale_signature(trial)
        except MediaError:
            return True
        for neighbour in (index - 1, index + 1):
            if not 0 <= neighbour < len(result):
                continue
            try:
                other = _grayscale_signature(result[neighbour][1])
            except MediaError:
                continue
            if _mean_difference(other, candidate) < UNIQUE_FRAME_THRESHOLD:
                return True
        return False

    async def _async_replace_infrared(
        self,
        camera: str,
        temporary: Path,
        selected: Sequence[tuple[float, Path]],
        recordings: Sequence[Mapping[str, Any]],
    ) -> tuple[list[tuple[float, Path]], list[dict[str, Any]]]:
        """Swap night-vision frames for a nearby colour frame."""
        return await self._async_retry_frames(
            camera,
            temporary,
            selected,
            recordings,
            offsets=INFRARED_RETRY_OFFSETS,
            tag="ir",
            needed=lambda _index, path, _result: frame_is_infrared(path),
            acceptable=lambda path, _index, _result: not frame_is_infrared(path),
        )

    async def _async_replace_overexposed(
        self,
        camera: str,
        temporary: Path,
        selected: Sequence[tuple[float, Path]],
        recordings: Sequence[Mapping[str, Any]],
    ) -> tuple[list[tuple[float, Path]], list[dict[str, Any]]]:
        """Swap a saturated frame for a nearby frame that kept its detail.

        A saturated frame carries no evidence: measured on the shadow sheets, one
        classification rested on a sheet whose first two cells were 96% and 82%
        near-white. Replacing them keeps six frames that can actually be judged.

        A frame with no acceptable alternative is left in place, matching the
        night-vision repair: the caller decides what that means, and the analysis
        still runs. The frame is recorded in `overexposure_replacements` so the
        evidence shows what could not be recovered.

        Night-vision frames are refused as replacements. Both defects come from
        the camera changing state, so an IR window often sits immediately before
        the blown window, and the ladder tries negative offsets first: accepting
        one would swap an unusable frame for a differently unusable frame and
        undo the night-vision repair that already ran.
        """
        return await self._async_retry_frames(
            camera,
            temporary,
            selected,
            recordings,
            offsets=INFRARED_RETRY_OFFSETS,
            tag="exposure",
            needed=lambda _index, path, _result: frame_is_overexposed(path),
            acceptable=lambda path, _index, _result: (
                not frame_is_overexposed(path) and not frame_is_infrared(path)
            ),
        )

    async def _async_replace_duplicates(
        self,
        camera: str,
        temporary: Path,
        selected: Sequence[tuple[float, Path]],
        recordings: Sequence[Mapping[str, Any]],
    ) -> tuple[list[tuple[float, Path]], list[dict[str, Any]]]:
        """Swap a frame that collides with another selection for a distinct one.

        The three fixed frames (first/last/postroll) are derived from the person
        window and never scored by the change selector, so on a still corridor
        one of them can land on a frame identical to another selection. The
        duplicate guard would then abort the whole activity, losing a real
        visit. Because the guard's own measure is the only reliable signal for
        this, the collision is resolved against that same measure.
        """
        signatures: dict[int, Image.Image] = {}

        def signature_of(index: int, path: Path) -> Image.Image:
            cached = signatures.get(index)
            if cached is None:
                cached = _grayscale_signature(path)
                signatures[index] = cached
            return cached

        def collides(index: int, path: Path, result: list[tuple[float, Path]]) -> bool:
            current = signature_of(index, path)
            for other, (_, other_path) in enumerate(result):
                if other == index:
                    continue
                if (
                    _mean_difference(signature_of(other, other_path), current)
                    < UNIQUE_FRAME_THRESHOLD
                ):
                    return True
            return False

        def acceptable(
            path: Path, index: int, result: list[tuple[float, Path]]
        ) -> bool:
            # Distinct from every other frame (the collision the guard measures).
            current = _grayscale_signature(path)
            for other, (_, other_path) in enumerate(result):
                if other == index:
                    continue
                if (
                    _mean_difference(_grayscale_signature(other_path), current)
                    < UNIQUE_FRAME_THRESHOLD
                ):
                    return False
            # Do not undo the night-vision repair. This runs after it, and a
            # pure uniqueness test will happily step backwards into the IR
            # region: measured on the 2026-09-20 08:27 review, index 0 was moved
            # to 08:27:49.177 (colour, R-G -32.6) and then relocated to
            # 08:27:48.772 (IR, R-G -152.8). A frame that is already IR may of
            # course stay IR -- there is nothing to protect in that case.
            if frame_is_infrared(path) and not frame_is_infrared(result[index][1]):
                return False
            return True

        return await self._async_retry_frames(
            camera,
            temporary,
            selected,
            recordings,
            offsets=UNIQUENESS_RETRY_OFFSETS,
            tag="uniq",
            needed=collides,
            acceptable=acceptable,
        )

    def _canonical_path(self, record: ActivityRecord) -> Path:
        final = (self._root / record.entry_id / f"{record.activity_id}.jpg").resolve()
        if final.parent != (self._root / record.entry_id).resolve():
            raise MediaError("invalid_output_path")
        return final

    async def async_restore_registry(self) -> None:
        """Rebuild the in-memory registry from validated persisted artifacts."""
        for record in self._store.all():
            if record.evidence_path is None or record.evidence_expired_at is not None:
                continue
            expected = self._canonical_path(record)
            if Path(record.evidence_path) != expected:
                raise MediaError("invalid_output_path")
            recovered = await self._hass.async_add_executor_job(
                _read_existing,
                expected,
                expected.with_suffix(".json"),
                record.activity_id,
            )
            if recovered is None or recovered[:2] != (
                record.evidence_mode,
                record.sample_times,
            ):
                raise MediaError("artifact_missing")
            self._register(record, expected)

    async def _complete(
        self,
        record: ActivityRecord,
        path: Path,
        mode: str,
        samples: tuple[float, ...],
        selection_source: str | None,
    ) -> ActivityRecord:
        completed = await self._store.async_complete_media(
            record.activity_id,
            key=media_key(record.activity_id, 1),
            evidence_mode=mode,
            evidence_revision=1,
            evidence_path=str(path),
            evidence_media_url=(
                f"media-source://frigate_vision/{record.entry_id}/{record.activity_id}"
            ),
            sample_times=samples,
            selection_source=selection_source,
            updated_at=time.time(),
        )
        self._register(record, path)
        return completed

    def _register(self, record: ActivityRecord, path: Path) -> None:
        registry: dict[str, Path] = self._hass.data.setdefault(DATA_MEDIA_REGISTRY, {})
        registry[f"{record.entry_id}/{record.activity_id}"] = path

    async def async_cleanup(
        self, *, retention_days: int, now: float | None = None
    ) -> tuple[str, ...]:
        """Delete expired registered files one at a time within the media root."""
        if retention_days < 1:
            raise MediaError("invalid_retention")
        cutoff = (time.time() if now is None else now) - retention_days * 86400
        removed: list[str] = []
        registry: dict[str, Path] = self._hass.data.setdefault(DATA_MEDIA_REGISTRY, {})
        for record in sorted(self._store.all(), key=lambda item: item.activity_id):
            if (
                record.stage not in {ActivityStage.COMPLETED, ActivityStage.FAILED}
                or record.updated_at >= cutoff
                or record.evidence_path is None
            ):
                continue
            try:
                path = await self._hass.async_add_executor_job(
                    _validated_cleanup_path,
                    Path(record.evidence_path),
                    self._root,
                )
            except ValueError as exc:
                raise MediaError("invalid_cleanup_path") from exc
            if record.evidence_expired_at is None:
                await self._store.async_expire_evidence(
                    record.activity_id,
                    expired_at=time.time() if now is None else now,
                )
            registry.pop(f"{record.entry_id}/{record.activity_id}", None)
            await self._hass.async_add_executor_job(_delete_artifact, path, self._root)
            removed.append(record.activity_id)
        return tuple(removed)


def _read_existing(
    final: Path, metadata: Path, activity_id: str
) -> tuple[str, tuple[float, ...], str | None] | None:
    if not final.is_file() or not metadata.is_file():
        return None
    try:
        with Image.open(final) as image:
            if image.format != "JPEG":
                return None
            image.load()
        payload = json.loads(metadata.read_text("utf-8"))
        if payload.get("activity_id") != activity_id or payload.get(
            "plan_version"
        ) not in {1, 2, 3, 4}:
            return None
        mode = str(payload["mode"])
        source_value = payload.get("selection_source")
        source = str(source_value) if source_value is not None else None
        samples = tuple(float(value) for value in payload["sample_times"])
    except OSError, ValueError, KeyError, TypeError, json.JSONDecodeError:
        return None
    return mode, samples, source


def _delete_artifact(path: Path, root: Path) -> None:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    if resolved.is_file():
        resolved.unlink()
    metadata = resolved.with_suffix(".json")
    metadata.relative_to(root.resolve())
    if metadata.is_file():
        metadata.unlink()


def _validated_cleanup_path(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(root.resolve())
    return resolved
