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


def test_motion_picks_stay_spread_when_path_points_cluster() -> None:
    """Index thirds are not time thirds, and the difference loses evidence.

    Built from the *measured* movement list of activity 19:05 on this
    deployment, whose points cluster early: movements land at relative seconds
    0.2, 1.8, 5.6, 21.4, 23.3, 23.7, 34.1, ... Splitting that list by index gave
    picks at 1.8 / 34.5 / 42.5 -- a 32.7s hole between the first two -- and the
    moment the entry door opened (10s into the clip) fell inside it. With no
    frame showing the door, the model answered `home_arrival` for someone
    walking *out*.

    Dividing the window by time instead bounds that hole: on the same data the
    picks become 1.8 / 23.7 / 35.1, a 21.9s worst gap.
    """
    from custom_components.frigate_vision.pathing import MotionPath, select_motion_times

    base = 1790161387.731
    movements = (
        (0.20, 0.0127),
        (1.79, 0.0853),
        (5.60, 0.0611),
        (21.36, 0.0581),
        (23.34, 0.0851),
        (23.70, 0.0902),
        (34.08, 0.0797),
        (34.48, 0.0906),
        (35.08, 0.1516),
        (41.27, 0.1361),
        (41.49, 0.0093),
        (42.47, 0.1236),
        (42.88, 0.0672),
        (43.27, 0.0612),
        (45.05, 0.0706),
        (45.68, 0.0685),
        (46.26, 0.0793),
    )
    # A movement is recorded at the *later* of each pair, so rebuild the path
    # from consecutive positions whose distance matches.
    points = [(base, 0.0, 0.0)]
    x = 0.0
    for relative, distance in movements:
        x += distance
        points.append((base + relative, x, 0.0))
    path = MotionPath(start_time=base, end_time=base + 50.0, points=tuple(points))

    lower = 1790161387.733
    upper = 1790161439.773  # the clip window used in production
    picked = select_motion_times([path], lower=lower, upper=upper)
    assert picked is not None
    assert len(picked) == 3
    assert list(picked) == sorted(picked)

    # The window endpoints are the sheet's outer cells, so measure the gaps
    # across the whole span, not just between picks.
    span = [lower, *picked, upper]
    worst = max(b - a for a, b in zip(span, span[1:], strict=False))
    assert worst <= 25.0, (
        f"picks left a {worst:.1f}s hole (index thirds left 32.7s); a short "
        f"event inside it is invisible: {picked}"
    )


def test_motion_selection_spreads_picks_inside_a_quiet_window() -> None:
    """A quiet stretch must not let two picks bunch together.

    Movements spanning only part of the window leave the largest gap set by the
    quiet part either way, so a rule that compares only the *largest* gap cannot
    tell a good spread from a clustered one. Comparing the gaps worst-first
    does: on a fixture whose movements occupy 104-109 of a 103-117 window, the
    largest gap is the quiet tail regardless, and only the second-largest gap
    distinguishes 105/107/109 from 104/105/109.
    """
    from custom_components.frigate_vision.pathing import MotionPath, select_motion_times

    points = tuple((103.0 + index, float(index), 0.0) for index in range(7))
    path = MotionPath(start_time=103.0, end_time=117.0, points=points)

    picked = select_motion_times([path], lower=103.05, upper=116.95)
    assert picked is not None
    assert len(picked) == 3
    gaps = [b - a for a, b in zip(picked, picked[1:], strict=False)]
    # No two picks may sit on adjacent movement instants while an unused one
    # remains between them.
    assert min(gaps) >= 2.0, f"picks clustered on adjacent instants: {picked}"


def test_motion_selection_reports_when_it_cannot_fill_every_pick() -> None:
    """Too few distinct instants must fall back rather than repeat a frame.

    The caller has a real alternative (image-change selection), so returning
    None is how the fallback is reached; emitting fewer than `count` picks
    would be a silent short sheet.
    """
    from custom_components.frigate_vision.pathing import MotionPath, select_motion_times

    # Four points give three movements, which is exactly `count`; three points
    # give only two and cannot fill three picks.
    points = tuple((100.0 + index, float(index), 0.0) for index in range(3))
    path = MotionPath(start_time=100.0, end_time=110.0, points=points)

    assert select_motion_times([path], lower=99.0, upper=111.0) is None


def test_motion_selection_uses_the_window_not_the_movement_extent() -> None:
    """The window is what the picks must cover, not the movements' own extent.

    The sheet's outer cells are the person's first and last frames, so the three
    picks own everything between them. Anchoring the thirds to the movements'
    extent instead would let picks bunch at whichever end movement happened to
    favour, leaving the other end unobserved.
    """
    from custom_components.frigate_vision.pathing import MotionPath, select_motion_times

    # Movement present across the window, but with a much larger burst late.
    points = [(100.0 + index * 2.0, index * 0.02, 0.0) for index in range(11)]
    points.append((120.0, 5.0, 0.0))
    path = MotionPath(start_time=100.0, end_time=125.0, points=tuple(points))

    lower, upper = 100.0, 124.0
    picked = select_motion_times([path], lower=lower, upper=upper)
    assert picked is not None
    # Every third of the *window* must be represented, within its own bounds.
    third = (upper - lower) / 3
    for index, value in enumerate(picked):
        assert lower + third * index <= value <= lower + third * (index + 1), (
            f"pick {value} escaped its own third: {picked}"
        )
