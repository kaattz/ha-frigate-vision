"""Preview selection must keep movement phases alongside door changes."""

from dataclasses import replace

import pytest

from custom_components.frigate_vision.changes import ScoredFrame, Segment
from custom_components.frigate_vision.media import EvidencePlan, choose_cells


def _fixture():
    times = (3., 4.4, 4.8, 7.4, 12.8, 20.4, 23., 27.3, 34.1, 37., 46.1)
    scored = [
        ScoredFrame(t, .3, .3, 0., {"lift": .8 if t in {4.8, 34.1} else .1})
        for t in times
    ]
    segments = [
        Segment(3., 12.8, "change", .8, (0, 1, 2, 3, 4)),
        Segment(20.4, 37., "change", .8, (5, 6, 7, 8, 9)),
        Segment(46.1, 46.1, "stable", 0., (10,)),
    ]
    plan = EvidencePlan(
        "review_six", first_time=4.4, last_time=43.1, postroll_time=46.1,
        motion_times=(7.44, 23.1, 37.1),
    )
    return scored, segments, plan


def test_motion_phases_survive_a_six_cell_change_selection():
    scored, segments, plan = _fixture()
    chosen = choose_cells(scored, segments, plan=plan, cells=6, max_cells=9)
    assert 7.4 in chosen, "走向镜头下缘的阶段不能被4.4/4.8秒两格挤掉"
    assert {4.4, 4.8, 34.1, 46.1} <= set(chosen), "快速门变化和首尾仍须保留"
    assert {23., 37.} <= set(chosen)
    assert len(chosen) == 9
    assert set(chosen) <= {item.timestamp for item in scored}


def test_no_motion_keeps_both_independent_zone_phases():
    scored, segments, plan = _fixture()
    chosen = choose_cells(
        scored, segments, plan=replace(plan, motion_times=()), cells=6, max_cells=9
    )
    assert chosen == (3., 4.4, 4.8, 12.8, 20.4, 27.3, 34.1, 43.1, 46.1)


def test_already_represented_motion_does_not_grow_the_grid():
    scored, segments, plan = _fixture()
    chosen = choose_cells(
        scored, segments, plan=replace(plan, motion_times=(4.81, 34.12)),
        cells=6, max_cells=9,
    )
    without_motion = choose_cells(
        scored, segments, plan=replace(plan, motion_times=()), cells=6, max_cells=9
    )
    assert chosen == without_motion


@pytest.mark.parametrize("motion", [(), (4.81, 34.12)])
def test_unconfigured_regions_keep_the_original_six(motion):
    scored, segments, plan = _fixture()
    scored = [
        replace(frame, sudden=frame.best_zone, cumulative=.3, zones={})
        for frame in scored
    ]
    chosen = choose_cells(
        scored, segments, plan=replace(plan, motion_times=motion),
        cells=6, max_cells=9,
    )
    assert chosen == (3., 4.4, 4.8, 20.4, 34.1, 46.1)


@pytest.mark.parametrize("motion", [(), (1.1, 4.1, 7.1)])
def test_no_new_motion_keeps_partial_change_rows_unchanged(motion):
    scored = [ScoredFrame(float(i), float(i % 3), 0., 0., {}) for i in range(10)]
    segments = [
        Segment(float(i), float(i + 1), "change", 1., (i, i + 1))
        for i in (0, 3, 6)
    ]
    plan = EvidencePlan(
        "review_six", first_time=0., last_time=8., postroll_time=9.,
        motion_times=motion,
    )
    chosen = choose_cells(scored, segments, plan=plan, cells=6, max_cells=9)
    assert chosen == (0., 1., 3., 6., 7., 9.)


def test_sparse_candidates_keep_the_original_six_and_door_pairs():
    times = (0., 2., 3., 8., 12., 13., 27., 30.)
    scored = [
        ScoredFrame(t, .3, .3, 0., {"lift": .8 if t in {3., 13.} else .1})
        for t in times
    ]
    segments = [
        Segment(2., 8., "change", .8, (1, 2, 3)),
        Segment(12., 27., "change", .8, (4, 5, 6)),
    ]
    plan = EvidencePlan(
        "review_six", first_time=0., last_time=27., postroll_time=30.,
        motion_times=(8., 27.),
    )
    chosen = choose_cells(scored, segments, plan=plan, cells=6, max_cells=9)
    assert chosen == (0., 2., 3., 12., 13., 30.)


def test_expansion_uses_remaining_candidates_when_anchors_are_not_preview_times():
    times = (2., 3., 8., 12., 13., 27., 30.)
    scored = [
        ScoredFrame(t, .3, .3, 0., {"lift": .8 if t in {3., 13.} else .1})
        for t in times
    ]
    segments = [
        Segment(2., 8., "change", .8, (0, 1, 2)),
        Segment(12., 27., "change", .8, (3, 4, 5)),
    ]
    plan = EvidencePlan(
        "review_six", first_time=1.1, last_time=27.1, postroll_time=30.1,
        motion_times=(8.,),
    )
    chosen = choose_cells(scored, segments, plan=plan, cells=6, max_cells=9)
    assert len(chosen) == 9
    assert {2., 3., 8., 12., 13.} <= set(chosen)
