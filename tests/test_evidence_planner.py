from __future__ import annotations

from dataclasses import replace

import pytest

from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import (
    EventWindow,
    MediaError,
    plan_evidence,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    ProcessingMode,
)
from custom_components.frigate_vision.pathing import MotionPath


def _door(updates, detections=("event_1",)) -> ActivityRecord:
    return ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=130,
        camera="front",
        detection_ids=detections,
        detection_zone_updates=updates,
        door_closed_at=130,
        association_deadline=140,
        finalization_deadline=220,
    )


ROLES = ZoneRoles(
    near=frozenset({"near"}),
    transition=frozenset({"mid"}),
    far=frozenset({"far"}),
)


def test_single_direction_plans_three_strictly_ordered_frames() -> None:
    record = _door(
        (
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("mid",)),
            ("event_1", 120, ("far",)),
        )
    )
    plan = plan_evidence(record, {"event_1": EventWindow("event_1", 101, 121)}, ROLES)
    assert plan.mode == "door_single"
    assert plan.sample_times == (102, 110, 120)


def test_direction_reversal_and_two_events_plan_six_frames() -> None:
    reversal = _door(
        (
            ("event_1", 102, ("near",)),
            ("event_1", 106, ("mid",)),
            ("event_1", 110, ("far",)),
            ("event_1", 114, ("far",)),
            ("event_1", 116, ("mid",)),
            ("event_1", 120, ("near",)),
        )
    )
    assert (
        len(
            plan_evidence(
                reversal, {"event_1": EventWindow("event_1", 101, 121)}, ROLES
            ).sample_times
        )
        == 6
    )

    split = replace(
        _door(
            (
                ("event_1", 102, ("near",)),
                ("event_1", 110, ("far",)),
                ("event_2", 140, ("far",)),
                ("event_2", 148, ("near",)),
            ),
            detections=("event_1", "event_2"),
        ),
        door_closed_at=150,
        updated_at=150,
        association_deadline=160,
        finalization_deadline=240,
    )
    plan = plan_evidence(
        split,
        {
            "event_1": EventWindow("event_1", 101, 112),
            "event_2": EventWindow("event_2", 139, 149),
        },
        ROLES,
    )
    assert plan.mode == "door_roundtrip"
    assert len(plan.sample_times) == 6
    assert list(plan.sample_times) == sorted(plan.sample_times)

    # Two segments that do not form an outbound/inbound pair fall back to the
    # conservative plan rather than failing. A door cycle must not be lost just
    # because zone updates stayed near the door, so `door_single` is the
    # deliberate outcome -- the vision model then decides from the frames.
    # Here both events read near -> far, so no reversal is detectable.
    no_reversal = replace(
        split,
        detection_zone_updates=(
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("far",)),
            ("event_2", 140, ("near",)),
            ("event_2", 148, ("far",)),
        ),
    )
    fallback = plan_evidence(
        no_reversal,
        {
            "event_1": EventWindow("event_1", 101, 112),
            "event_2": EventWindow("event_2", 139, 149),
        },
        ROLES,
    )
    assert fallback.mode == "door_single"
    assert fallback.selection_source == "zone_anchor"
    assert len(fallback.sample_times) == 3
    assert list(fallback.sample_times) == sorted(fallback.sample_times)

    # An observation outside its own event window is a hard error: the
    # correlation itself is wrong, not merely inconclusive.
    outside = replace(
        split,
        detection_ids=("event_1",),
        detection_zone_updates=(
            ("event_1", 90, ("near",)),
            ("event_1", 110, ("far",)),
        ),
    )
    with pytest.raises(MediaError, match="event_timeline_outside_window"):
        plan_evidence(
            outside,
            {"event_1": EventWindow("event_1", 101, 112)},
            ROLES,
        )


def test_standalone_review_plans_change_candidates_and_postroll() -> None:
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.SHADOW,
        created_at=100,
        updated_at=120,
        camera="front",
        review_ids=("review_1",),
        detection_ids=("event_1",),
        finalization_deadline=120,
    )
    plan = plan_evidence(record, {"event_1": EventWindow("event_1", 103, 117)}, ROLES)
    assert plan.mode == "review_six"
    assert len(plan.change_candidates) == 9
    assert plan.first_time == pytest.approx(103.2)
    assert plan.last_time == pytest.approx(116.8)
    assert plan.postroll_time == pytest.approx(119.8)
    assert all(
        plan.first_time < value < plan.last_time for value in plan.change_candidates
    )


def test_ambiguous_single_detection_fails() -> None:
    record = _door((("event_1", 102, ("near",)),))
    with pytest.raises(MediaError, match="direction_ambiguous"):
        plan_evidence(record, {"event_1": EventWindow("event_1", 101, 121)}, ROLES)


def test_person_may_start_before_review_when_detection_id_matches() -> None:
    record = ActivityRecord(
        activity_id="review_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100.2,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=120,
    )
    plan = plan_evidence(record, {"event_1": EventWindow("event_1", 100, 119)}, ROLES)
    assert plan.first_time == pytest.approx(100.2)
    assert plan.postroll_time == pytest.approx(121.8)


def test_standalone_review_prefers_path_motion_selection() -> None:
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=120,
    )
    path = MotionPath(
        start_time=103,
        end_time=117,
        points=tuple((103 + index, float(index), 0.0) for index in range(7)),
    )
    plan = plan_evidence(
        record,
        {"event_1": EventWindow("event_1", 103, 117)},
        ROLES,
        {"event_1": path},
    )
    assert plan.selection_source == "path_motion"
    assert len(plan.motion_times) == 3
    assert plan.first_time < plan.motion_times[0]
    assert plan.motion_times[-1] < plan.last_time


def test_standalone_review_falls_back_when_path_points_are_insufficient() -> None:
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=120,
        camera="front",
        detection_ids=("event_1",),
        finalization_deadline=120,
    )
    path = MotionPath(
        start_time=103,
        end_time=117,
        points=((103.0, 0.0, 0.0), (104.0, 1.0, 0.0)),
    )
    plan = plan_evidence(
        record,
        {"event_1": EventWindow("event_1", 103, 117)},
        ROLES,
        {"event_1": path},
    )
    assert plan.selection_source == "image_change"
    assert plan.motion_times == ()
    assert len(plan.change_candidates) == 9


def test_door_cycle_plan_marks_zone_anchor_source() -> None:
    record = _door(
        (
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("mid",)),
            ("event_1", 120, ("far",)),
        )
    )
    plan = plan_evidence(record, {"event_1": EventWindow("event_1", 101, 121)}, ROLES)
    assert plan.selection_source == "zone_anchor"
    assert plan.motion_times == ()


def test_person_track_may_outlive_review_window() -> None:
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=103,
        updated_at=104,
        camera="front",
        review_ids=("review_1",),
        detection_ids=("event_1",),
        finalization_deadline=104,
    )
    plan = plan_evidence(record, {"event_1": EventWindow("event_1", 102.8, 110)}, ROLES)
    assert plan.selection_source == "image_change"
    assert len(plan.change_candidates) == 9
    assert plan.first_time < plan.last_time
    assert plan.postroll_time == 110 + 2.8
