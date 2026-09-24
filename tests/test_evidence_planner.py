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


def test_gap_probe_targets_the_widest_unobserved_stretch() -> None:
    """Probes must land in the sheet's largest hole, not spread everywhere.

    Measured on this deployment: the three motion picks leave holes of 12-39s,
    and every such hole that was probed contained real change (mean greyscale
    difference 2.45-85.66, against the component's own threshold of 1.0). So the
    holes are not empty corridors -- something happens in them that the sheet
    never shows.

    Cell count is fixed at six, so the hole cannot be closed by adding a cell;
    the picks have to be re-aimed. That needs frames, and frames cost fetches,
    so the probing has to be aimed at one place: the widest gap.
    """
    from custom_components.frigate_vision.pathing import probe_gap_times

    # One clearly widest hole: 50.0 -> 80.0 (30s), against 5-10s elsewhere.
    sheet = (0.0, 5.0, 15.0, 25.0, 50.0, 80.0)
    probes = probe_gap_times(sheet, max_gap=12.0, probes=4)

    assert len(probes) == 4
    for moment in probes:
        assert 50.0 < moment < 80.0, f"probe {moment} escaped the widest gap"
    assert list(probes) == sorted(probes)


def test_gap_probe_is_skipped_when_no_hole_is_wide_enough() -> None:
    """A tightly covered sheet must cost nothing.

    Every probe is a snapshot fetch, and most activities do not need any: a hole
    shorter than the shortest action worth seeing cannot be hiding one. Returning
    nothing is how the caller avoids the work.
    """
    from custom_components.frigate_vision.pathing import probe_gap_times

    sheet = (0.0, 5.0, 9.0, 13.0, 17.0, 20.0)
    assert probe_gap_times(sheet, max_gap=12.0, probes=4) == ()


def test_gap_probe_ignores_the_stretch_beyond_the_activity() -> None:
    """The gap after the person left is not a gap.

    The sheet's last two cells are the person's final frame and the emptied
    scene, which are *meant* to be far apart: nothing happens in between by
    definition. Probing there would spend fetches on an empty corridor and could
    displace a pick that was covering the activity itself.

    The fixture makes that the *only* qualifying hole, so a version that ignored
    `within` would probe the postroll stretch and this test would see it.
    """
    from custom_components.frigate_vision.pathing import probe_gap_times

    # Activity 0..20 is tightly covered (<=8s holes); the postroll cell sits
    # 40s later, creating a 40s hole that must not be probed.
    sheet = (0.0, 6.0, 12.0, 20.0, 60.0)

    # Without the bound, the postroll hole is the widest and would win.
    unbounded = probe_gap_times(sheet, max_gap=12.0, probes=4)
    assert unbounded and all(moment > 20.0 for moment in unbounded), (
        "fixture no longer isolates the postroll stretch"
    )

    # With the activity bound, there is no hole wide enough, so nothing is
    # fetched -- and crucially nothing beyond the activity is probed either.
    assert probe_gap_times(sheet, max_gap=12.0, probes=4, within=(0.0, 20.0)) == ()


def test_gap_probe_respects_the_activity_bound_when_a_hole_remains() -> None:
    """A qualifying hole inside the activity is still found when bounded."""
    from custom_components.frigate_vision.pathing import probe_gap_times

    # A 25s hole between 10 and 35, inside an activity that ends at 40.
    sheet = (0.0, 10.0, 35.0, 40.0, 90.0)
    probes = probe_gap_times(sheet, max_gap=12.0, probes=4, within=(0.0, 40.0))

    assert len(probes) == 4
    assert all(10.0 < moment < 35.0 for moment in probes), probes


def test_picks_can_be_re_aimed_at_probed_frames() -> None:
    """Probing is pointless unless the picks can actually move into the hole.

    The measured holes cover most of an activity -- 19:05's picks left a 17.0s
    hole in a 52.0s window -- so closing one necessarily means giving up a pick
    somewhere else. Offering the probed instants as extra candidates is what lets
    the same worst-gap rule decide that trade, rather than a second heuristic
    guessing which cell to sacrifice.
    """
    from custom_components.frigate_vision.pathing import MotionPath, select_motion_times

    # Movements cluster at the start; the tail from 30s on has none.
    points = [(100.0 + index * 0.5, index * 0.4, 0.0) for index in range(8)]
    path = MotionPath(start_time=100.0, end_time=160.0, points=tuple(points))

    without = select_motion_times([path], lower=100.0, upper=160.0)
    assert without is not None

    # Frames found by probing the hole the picks left, with the change each one
    # showed. Higher change means more happened there.
    extra = ((135.0, 12.0), (145.0, 30.0), (155.0, 8.0))
    with_probes = select_motion_times(
        [path], lower=100.0, upper=160.0, extra=extra
    )
    assert with_probes is not None

    def worst(picks) -> float:
        edges = [100.0, *picks, 160.0]
        return max(b - a for a, b in zip(edges, edges[1:], strict=False))

    assert worst(with_probes) < worst(without), (
        f"probing did not close the hole: {worst(without):.1f}s -> "
        f"{worst(with_probes):.1f}s"
    )


def test_relative_weights_make_two_sources_comparable() -> None:
    """Each source is scaled within itself, not against the other.

    A path movement is normalised frame coordinates (measured on this deployment
    at 0.01-0.15) and a greyscale change is 0-255 (measured at 2-85), so raw
    values from the two are not comparable. Scaling each source to its own
    maximum makes "how much happened here" mean the same thing in both.

    Tested directly because the ranking it feeds is a *tie-break*: the gap rule
    decides first, and two candidate sets tie on gaps only when their gap
    sequences are identical, which real float timestamps essentially never
    produce. A test routed through `select_motion_times` would therefore be
    asserting on a branch it cannot reach.
    """
    from custom_components.frigate_vision.pathing import _relative_weights

    movements = [(1.0, 0.05), (2.0, 0.10), (3.0, 0.025)]
    changes = [(1.0, 40.0), (2.0, 80.0), (3.0, 20.0)]

    weights = _relative_weights(movements)
    assert weights == {1.0: 0.5, 2.0: 1.0, 3.0: 0.25}

    # The same *shape* in the other source yields the same weights, which is the
    # point: 80 is to changes what 0.10 is to movements.
    assert _relative_weights(changes) == weights

    # A source where nothing happened must not divide by zero, and must not
    # pretend some instant was more notable than another.
    assert _relative_weights([(1.0, 0.0), (2.0, 0.0)]) == {1.0: 0.0, 2.0: 0.0}
    assert _relative_weights([]) == {}


def test_probed_instants_can_win_a_slot_on_coverage_merit() -> None:
    """A probe is a candidate like any other, judged by the same gap rule.

    This is what probing actually buys: the detector reports where the *person*
    moved, and a door opening while the person stands still produces no movement
    at all. Adding the probed instants as candidates lets the existing rule close
    a hole it could not otherwise close.
    """
    from custom_components.frigate_vision.pathing import MotionPath, select_motion_times

    # Movement only in the first quarter of the window.
    points = [(100.0 + index * 2.0, index * 0.05, 0.0) for index in range(8)]
    path = MotionPath(start_time=100.0, end_time=160.0, points=tuple(points))

    without = select_motion_times([path], lower=100.0, upper=160.0)
    assert without is not None

    # Three instants found by probing the hole, with the visible change each
    # showed. The most recent one is the strongest, so it should be preferred.
    extra = ((130.0, 4.0), (140.0, 9.0), (150.0, 30.0))
    with_probes = select_motion_times(
        [path], lower=100.0, upper=160.0, extra=extra
    )
    assert with_probes is not None

    def worst(picks) -> float:
        edges = [100.0, *picks, 160.0]
        return max(b - a for a, b in zip(edges, edges[1:], strict=False))

    assert worst(with_probes) < worst(without), (
        f"probing did not close the hole: {worst(without):.1f}s -> "
        f"{worst(with_probes):.1f}s"
    )
    probed = {moment for moment, _ in extra}
    assert probed & set(with_probes), (
        f"no probed instant was selected, so the hole was closed by motion "
        f"alone: {with_probes}"
    )


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
