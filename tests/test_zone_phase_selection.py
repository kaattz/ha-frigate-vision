"""Different focus regions must survive one shared whole-frame change run."""

from custom_components.frigate_vision.changes import ScoredFrame, Segment
from custom_components.frigate_vision.media import EvidencePlan, choose_cells


def _three_regions(*, simultaneous=False):
    scored = []
    for index in range(13):
        zones = {
            "elevator": 1.0 if 1 <= index <= 9 else 0.0,
            "home_door": .4 if index == (2 if simultaneous else 4) else 0.0,
            "door": .5 if index == (2 if simultaneous else 7) else 0.0,
            "quiet": 0.0,
        }
        scored.append(ScoredFrame(float(index), .3, .3, 0., zones))
    segments = [
        Segment(0., 0., "stable", 0., (0,)),
        Segment(1., 9., "change", 1., tuple(range(1, 10))),
        Segment(10., 12., "stable", 0., (10, 11, 12)),
    ]
    return scored, segments


def test_three_region_peaks_survive_one_global_segment_with_motion():
    scored, segments = _three_regions()
    plan = EvidencePlan(
        "review_six", first_time=0., last_time=11., postroll_time=12.,
        motion_times=(6., 9., 10.),
    )
    chosen = choose_cells(scored, segments, plan=plan, cells=6, max_cells=9)
    assert {1., 4., 7.} <= set(chosen), "电梯长变化不能吞掉两扇门的独立变化"
    assert {6., 9., 10.} <= set(chosen), "区域补充仍保留人物移动阶段"
    assert len(chosen) == 9


def test_simultaneous_region_changes_share_a_cell():
    scored, segments = _three_regions(simultaneous=True)
    plan = EvidencePlan("review_six", first_time=0., last_time=11., postroll_time=12.)
    chosen = choose_cells(scored, segments, plan=plan, cells=6, max_cells=9)
    assert 2. in chosen
    assert len(chosen) == len(set(chosen))
    assert len(chosen) in {6, 9}


def test_independent_region_cycles_are_available_for_selection():
    from custom_components.frigate_vision.changes import segment_focus_zones

    scored, _ = _three_regions()
    scored[8] = ScoredFrame(8., .3, .3, 0., {
        **scored[8].zones, "home_door": .35,
    })
    grouped = segment_focus_zones(scored, change_threshold=.25)
    assert len([s for s in grouped["home_door"] if s.kind == "change"]) == 2
    assert not any(s.kind == "change" for s in grouped["quiet"])


def test_repair_cannot_cross_a_region_change_inside_a_global_run():
    from custom_components.frigate_vision.changes import refine_focus_segments
    from custom_components.frigate_vision.media import within_segment

    scored, segments = _three_regions()
    refined = refine_focus_segments(scored, segments, change_threshold=.25)
    assert within_segment(4.2, 4., refined)
    assert not within_segment(4.8, 4., refined)
    assert not within_segment(6.8, 4., refined)
    assert within_segment(5.1, 4.8, refined), "采样间的锚点也须有明确归属"


def test_no_regions_keep_the_original_repair_segments():
    from custom_components.frigate_vision.changes import refine_focus_segments

    scored = [ScoredFrame(float(i), .3, .3, 0., {}) for i in range(4)]
    segments = [Segment(0., 3., "change", .3, (0, 1, 2, 3))]
    assert refine_focus_segments(scored, segments, change_threshold=.25) == segments


def test_outer_anchors_cannot_bypass_known_region_boundaries():
    from dataclasses import replace

    from custom_components.frigate_vision.changes import refine_focus_segments
    from custom_components.frigate_vision.media import within_segment

    scored, segments = _three_regions()
    scored = [replace(frame, timestamp=frame.timestamp + 100.) for frame in scored]
    segments = [
        replace(segment, start=segment.start + 100., end=segment.end + 100.)
        for segment in segments
    ]
    refined = refine_focus_segments(scored, segments, change_threshold=.25)
    assert not within_segment(104., 99.2, refined)
    assert not within_segment(107., 113., refined)
    assert within_segment(99.4, 99.2, refined)
    assert within_segment(114., 113., refined)
