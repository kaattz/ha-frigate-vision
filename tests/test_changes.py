"""Tests for the block-based change analysis.

The load-bearing test here is
`test_a_local_change_beats_a_whole_frame_average`: it is the reason the module
exists. A door opening on the left of a still corridor moves a small fraction of
the frame, and a whole-frame average cannot separate that from sensor noise --
so the suite has to fail if the analysis ever degrades to an average.
"""

from __future__ import annotations

import numpy as np
import pytest

from custom_components.frigate_vision.changes import (
    BLOCK_COLUMNS,
    BLOCK_ROWS,
    BlockSpec,
    frame_score,
    global_change,
    residual,
    segment_frames,
)


def _flat(value: int, *, height: int = 90, width: int = 160) -> np.ndarray:
    return np.full((height, width), value, dtype=np.uint8)


def _with_patch(
    value: int,
    patch: tuple[int, int, int, int],
    patch_value: int,
    *,
    height: int = 90,
    width: int = 160,
) -> np.ndarray:
    frame = _flat(value, height=height, width=width)
    top, left, bottom, right = patch
    frame[top:bottom, left:right] = patch_value
    return frame


class TestResidualCompensatesForGlobalLight:
    """A uniformly brighter frame must not read as a changed one."""

    def test_a_uniform_brightness_shift_leaves_no_residual(self) -> None:
        """The whole point of subtracting the median.

        Without it, a camera's own auto-exposure would score every frame as
        changed and the segment structure would be noise.
        """
        before = _flat(100)
        after = _flat(140)
        assert residual(before, after).max() == 0

    def test_a_local_change_survives_the_compensation(self) -> None:
        """A door moving is local, so the median does not follow it."""
        before = _flat(100)
        after = _with_patch(100, (0, 0, 90, 40), 200)
        assert residual(before, after).max() > 0

    def test_a_dark_change_does_not_wrap_around(self) -> None:
        """Unsigned subtraction would report a dark pixel as a bright change.

        `uint8` arithmetic wraps: 10 - 30 is 236, a huge positive difference that
        looks like the brightest possible change. The `int16` cast is what stops
        a shadow from reading as a light.
        """
        before = _flat(30)
        after = _with_patch(30, (0, 0, 90, 40), 10)
        difference = residual(before, after)
        assert difference.max() == 20


class TestGlobalChangeIsRecordedNotDiscarded:
    """Compensation removes the uniform part; the metadata must keep it."""

    def test_a_brightness_swing_is_reported(self) -> None:
        assert global_change(_flat(100), _flat(140)) == pytest.approx(40.0)

    def test_a_local_change_is_not_reported_as_global(self) -> None:
        before = _flat(100)
        after = _with_patch(100, (0, 0, 90, 40), 200)
        # Three quarters of the frame is untouched, so the median stays put.
        assert global_change(before, after) == pytest.approx(0.0)


class TestBlocksLocaliseChange:
    """The frame score must come from the busiest blocks, not the average."""

    def test_a_local_change_beats_a_whole_frame_average(self) -> None:
        """THE load-bearing test.

        A door-sized change covers two of the 48 blocks. The whole-frame mean over
        a 90x160 frame is `600 / 14400 = 4.2%` of pixels -- small enough to sit
        inside a threshold chosen for real camera noise. Taking the mean of the
        busiest blocks reports the proportion *inside the changed blocks*, which
        is near 1.0 for the two that changed.

        The score is the mean of the top THREE blocks, so with only two changed
        blocks the third contributes zero and the score lands at 2/3. That is the
        intended behaviour rather than a shortcoming: requiring exactly the number
        of changed blocks would need the count known in advance, and 2/3 is still
        an order of magnitude above the frame-wide mean.

        Asserted as a ratio against the frame-wide mean so the test fails if the
        implementation ever degrades to averaging the whole grid.
        """
        before = _flat(100)
        after = _with_patch(100, (0, 0, 15, 40), 220)

        score = frame_score(before, after)
        whole_frame_mean = float((residual(before, after) > 12.0).mean())
        assert score > 0.6, f"the changed blocks are almost fully changed: {score}"
        assert whole_frame_mean < 0.05, (
            f"and the whole-frame mean is tiny: {whole_frame_mean}"
        )
        assert score > 10 * whole_frame_mean, (
            "分块必须显著优于全图平均，否则门口那一小块变化会被稀释掉"
        )

    def test_the_score_ignores_the_quiet_blocks(self) -> None:
        """The score reflects the changed blocks, not the size of the frame.

        An average cannot have this property: the same event would score lower on
        a larger frame simply because there is more still wall to average over.
        The comparison scales the changed region with the frame so both describe
        the same two blocks.
        """
        # 90x160 -> blocks are 20x15; a 2-block region is rows 0-14, cols 0-39.
        small_before = _flat(100, height=90, width=160)
        small_after = small_before.copy()
        small_after[0:15, 0:40] = 220

        # 180x320 -> blocks are 40x30; the same two blocks are rows 0-29, cols 0-79.
        large_before = _flat(100, height=180, width=320)
        large_after = large_before.copy()
        large_after[0:30, 0:80] = 220

        small_score = frame_score(small_before, small_after)
        large_score = frame_score(large_before, large_after)
        assert small_score == pytest.approx(large_score), (
            "同样两个块发生变化时必须得到同样的分数，与整帧大小无关"
        )

        # The contrast with an average is the SIZE of the number, not its trend:
        # both frames have the same proportion of changed pixels, so the
        # whole-frame mean equals the fraction of the grid that moved (2/48),
        # while the block score reports the proportion changed *inside* the
        # blocks that moved. That gap is what lets one threshold work.
        whole_frame_mean = float((residual(small_before, small_after) > 12.0).mean())
        assert whole_frame_mean == pytest.approx(2 / 48, abs=1e-3)
        assert small_score > 10 * whole_frame_mean

    def test_an_unchanged_frame_scores_zero(self) -> None:
        assert frame_score(_flat(100), _flat(100)) == 0.0

    def test_a_uniform_brightness_shift_scores_zero(self) -> None:
        assert frame_score(_flat(100), _flat(160)) == 0.0

    def test_a_spread_out_change_scores_lower_than_a_concentrated_one(self) -> None:
        """Concentration is the signal: the same pixel count over more blocks
        must score lower, or the measure would not prefer a local event."""
        before = _flat(100)
        concentrated = _with_patch(100, (0, 0, 15, 20), 220)
        spread = _flat(100)
        # Same 300-pixel area, but 10 pixels in each of 30 blocks.
        for column in range(BLOCK_COLUMNS):
            for row in range(BLOCK_ROWS):
                top = row * 15
                left = column * 20
                spread[top : top + 2, left : left + 5] = 220
        assert frame_score(before, concentrated) > frame_score(before, spread)


class TestTheGridToleratesAnOddSize:
    """A frame whose size is not divisible by the grid is analysed, not rejected."""

    def test_a_frame_one_pixel_off_still_analyses(self) -> None:
        before = _flat(100, height=91, width=161)
        after = _with_patch(100, (0, 0, 15, 40), 220, height=91, width=161)
        assert frame_score(before, after) > 0.5

    def test_a_frame_smaller_than_the_grid_is_refused(self) -> None:
        """Too small to hold one block per cell is a real error, not a silent 0."""
        tiny = _flat(100, height=3, width=3)
        with pytest.raises(ValueError, match="frame_too_small_for_grid"):
            frame_score(tiny, tiny)

    def test_a_custom_grid_is_honoured(self) -> None:
        """A coarser grid gives each block more pixels, so the change is more
        concentrated and the score rises for the same event."""
        spec = BlockSpec(columns=4, rows=4)
        before = _flat(100)
        after = _with_patch(100, (0, 0, 22, 40), 220)
        coarse = frame_score(before, after, spec=spec)
        fine = frame_score(before, after)
        assert coarse > 0
        assert coarse != fine, "网格尺寸必须真的改变结果"


class TestSegmentsAreLocalNotGlobal:
    """Adjacent-only grouping, because two similar frames can be far apart."""

    def test_a_single_change_run_forms_one_segment(self) -> None:
        times = (0.0, 1.0, 2.0, 3.0, 4.0)
        scores = (0.0, 0.9, 0.9, 0.0)
        segments = segment_frames(times, scores, change_threshold=0.5)
        assert [s.kind for s in segments] == ["stable", "change", "stable"]

    def test_closed_opened_closed_is_not_collapsed(self) -> None:
        """The door case: two similar frames must stay in separate segments.

        A global dedup pass would see frames 0 and 4 as nearly identical and merge
        them, losing the opening between them -- which is the whole event.
        """
        times = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
        # still -> change -> still -> change -> still
        scores = (0.0, 0.8, 0.1, 0.9, 0.1)
        segments = segment_frames(times, scores, change_threshold=0.5)
        kinds = [s.kind for s in segments]
        assert kinds.count("change") == 2, (
            "两次开门必须是两个变化段，不能因为首尾画面相似就被合并"
        )
        change_segments = [s for s in segments if s.kind == "change"]
        assert change_segments[0].end < change_segments[1].start

    def test_every_frame_belongs_to_exactly_one_segment(self) -> None:
        """Coverage and disjointness, or a frame would be usable twice or lost."""
        times = (0.0, 1.0, 2.0, 3.0, 4.0)
        scores = (0.1, 0.9, 0.1, 0.9)
        segments = segment_frames(times, scores, change_threshold=0.5)
        seen = [index for segment in segments for index in segment.indices]
        assert sorted(seen) == [0, 1, 2, 3, 4]

    def test_an_all_stable_run_is_one_segment(self) -> None:
        segments = segment_frames((0.0, 1.0, 2.0), (0.0, 0.0), change_threshold=0.5)
        assert len(segments) == 1
        assert segments[0].kind == "stable"

    def test_an_all_change_run_is_one_segment(self) -> None:
        segments = segment_frames((0.0, 1.0, 2.0), (0.9, 0.9), change_threshold=0.5)
        assert len(segments) == 1
        assert segments[0].kind == "change"

    def test_a_single_frame_is_a_segment(self) -> None:
        segments = segment_frames((5.0,), (), change_threshold=0.5)
        assert len(segments) == 1
        assert segments[0].indices == (0,)

    def test_no_frames_is_no_segments(self) -> None:
        assert segment_frames((), (), change_threshold=0.5) == []

    def test_mismatched_score_count_is_refused(self) -> None:
        """A score per adjacent pair: n frames need n-1 scores."""
        with pytest.raises(ValueError, match="scores_must_describe_adjacent_pairs"):
            segment_frames((0.0, 1.0, 2.0), (0.5,), change_threshold=0.5)

    def test_segment_bounds_follow_the_frame_times(self) -> None:
        times = (10.0, 11.5, 13.0, 20.0)
        scores = (0.0, 0.9, 0.0)
        segments = segment_frames(times, scores, change_threshold=0.5)
        assert segments[0].start == 10.0
        assert segments[-1].end == 20.0
        # A one-frame segment has zero span, which is legitimate: it is a single
        # instant, not a range. Every longer segment must be a real interval.
        for segment in segments:
            if len(segment.indices) > 1:
                assert segment.span > 0, (
                    f"多帧段必须是真正的时间区间：{segment}"
                )


class TestAnalyzeSequenceScoresAgainstTwoReferences:
    """Both comparisons must be present, and the anchor must actually reset.

    Comparing only against the previous frame misses a slow change; comparing only
    against the run's first frame reports every later frame as changed because the
    scene has drifted. The reset is what keeps the second reference meaningful, and
    without it a long still period after one movement keeps scoring high against a
    moment that no longer matters.
    """

    def test_a_slow_accumulating_change_is_caught(self) -> None:
        """Four small steps, none big enough to notice pairwise.

        This is the door-opening-over-several-preview-frames case. Each step moves
        6 grey levels, under PIXEL_THRESHOLD (12), so `sudden` is exactly 0 for
        every step while the total is unmistakable. Asserting that explicitly is
        what makes this test fail if the analysis ever stops comparing against a
        segment anchor -- without that assertion the fixture would pass on the
        pairwise score alone and the property would go untested.
        """
        from custom_components.frigate_vision.changes import analyze_sequence

        base = _flat(100)
        frames = [(float(index), base.copy()) for index in range(5)]
        for index in range(1, 5):
            frames[index][1][0:15, 0:40] = 100 + 6 * index

        scored = analyze_sequence(frames)
        assert all(item.sudden == 0.0 for item in scored), (
            "每一步本身都必须低于阈值，否则这条测试测的不是累积比较"
        )
        assert scored[3].cumulative > 0.5, (
            "累积比较必须发现门已经明显打开，否则慢变化会被漏掉"
        )
        assert scored[3].score > 0.5, (
            "分数必须取两者较大值，否则慢变化仍然进不了变化段"
        )

    def test_the_anchor_resets_after_a_change_segment(self) -> None:
        """Otherwise a later still frame keeps scoring against the old state."""
        from custom_components.frigate_vision.changes import analyze_sequence

        base = _flat(100)
        moved = base.copy()
        moved[0:15, 0:40] = 220
        # frame0 still, frame1 moved, frames 2-4 still at the moved state.
        frames = [(0.0, base), (1.0, moved), (2.0, moved), (3.0, moved), (4.0, moved)]

        without_reset = analyze_sequence(frames)
        with_reset = analyze_sequence(frames, anchor_reset_threshold=0.5)

        # Without the reset, the anchor is still frame0, so every later frame
        # reports the full difference.
        assert without_reset[4].cumulative > 0.5
        # With it, frame1 became the anchor, so the still frames report nothing.
        assert with_reset[4].cumulative == 0.0, (
            "变化段结束后锚点必须前移，否则静止画面会一直报「有变化」"
        )

    def test_the_first_frame_has_no_comparison(self) -> None:
        """A single frame cannot differ from a predecessor that does not exist."""
        from custom_components.frigate_vision.changes import analyze_sequence

        scored = analyze_sequence([(5.0, _flat(100))])
        assert len(scored) == 1
        assert scored[0].sudden == 0.0
        assert scored[0].cumulative == 0.0
        assert scored[0].score == 0.0

    def test_no_frames_scores_nothing(self) -> None:
        from custom_components.frigate_vision.changes import analyze_sequence

        assert analyze_sequence([]) == []

    def test_zone_scores_are_reported_per_zone(self) -> None:
        from custom_components.frigate_vision.changes import analyze_sequence

        base = _flat(100)
        changed = base.copy()
        # `_flat` is 160x90, and the home-door polygon occupies x 29..84,
        # y 11..89 at that size -- so this patch sits inside the zone.
        changed[40:70, 40:75] = 220
        zones = {"home_door": _DOOR_POLYGON, "elevator": _LIFT_POLYGON}

        scored = analyze_sequence([(0.0, base), (1.0, changed)], zones=zones)
        assert set(scored[1].zones) == {"home_door", "elevator"}
        assert scored[1].zones["home_door"] > 0.0
        assert scored[1].best_zone == scored[1].zones["home_door"]

    def test_without_zones_every_zone_score_is_zero(self) -> None:
        """Zero regression: an unconfigured entry must behave as before."""
        from custom_components.frigate_vision.changes import analyze_sequence

        changed = _flat(100)
        changed[0:15, 0:40] = 220
        scored = analyze_sequence([(0.0, _flat(100)), (1.0, changed)])
        assert scored[1].zones == {}
        assert scored[1].best_zone == 0.0

    def test_the_uniform_shift_is_recorded_not_compensated_away(self) -> None:
        """An exposure swing must be visible in the metadata."""
        from custom_components.frigate_vision.changes import analyze_sequence

        scored = analyze_sequence([(0.0, _flat(100)), (1.0, _flat(140))])
        assert scored[1].global_shift == pytest.approx(40.0)
        assert scored[1].score == 0.0, "但均匀变化不应被当成场景变化"


class TestSegmentScoredFrames:
    """The adapter applies one threshold to the combined score."""

    def test_it_groups_by_the_combined_score(self) -> None:
        from custom_components.frigate_vision.changes import (
            analyze_sequence,
            segment_scored_frames,
        )

        base = _flat(100)
        moved = base.copy()
        moved[0:15, 0:40] = 220
        frames = [(0.0, base), (1.0, moved), (2.0, moved)]
        scored = analyze_sequence(frames)
        segments = segment_scored_frames(scored, change_threshold=0.5)
        assert len(segments) >= 1
        assert segments[0].start == 0.0
        assert segments[-1].end == 2.0

    def test_a_flat_sequence_is_one_stable_segment(self) -> None:
        from custom_components.frigate_vision.changes import (
            analyze_sequence,
            segment_scored_frames,
        )

        scored = analyze_sequence([(float(i), _flat(100)) for i in range(4)])
        segments = segment_scored_frames(scored, change_threshold=0.5)
        assert [s.kind for s in segments] == ["stable"]


class TestFocusZoneCanStartAChangeSegment:
    ZONE = (
        (25 / 160, 20 / 90),
        (35 / 160, 20 / 90),
        (35 / 160, 30 / 90),
        (25 / 160, 30 / 90),
    )

    def test_small_zone_change_crosses_the_threshold(self) -> None:
        from custom_components.frigate_vision.changes import (
            analyze_sequence,
            segment_scored_frames,
        )

        before = _flat(80)
        opened = _with_patch(80, (20, 25, 30, 35), 160)
        frames = [(0.0, before), (1.0, before), (2.0, opened)]
        control = analyze_sequence(frames, anchor_reset_threshold=0.5)
        focused = analyze_sequence(
            frames, zones={"door": self.ZONE}, anchor_reset_threshold=0.5
        )

        assert focused[-1].sudden < 0.25
        assert focused[-1].cumulative < 0.25
        assert focused[-1].best_zone == pytest.approx(1.0)
        assert all(
            segment.kind == "stable"
            for segment in segment_scored_frames(control, change_threshold=0.25)
        )
        assert any(
            segment.kind == "change"
            for segment in segment_scored_frames(focused, change_threshold=0.25)
        )

    def test_open_hold_close_produces_two_separate_changes(self) -> None:
        from custom_components.frigate_vision.changes import (
            analyze_sequence,
            segment_scored_frames,
        )

        closed = _flat(80)
        opened = _with_patch(80, (20, 25, 30, 35), 160)
        frames = list(
            enumerate([closed, closed, opened, opened, opened, closed, closed])
        )
        scored = analyze_sequence(
            frames, zones={"door": self.ZONE}, anchor_reset_threshold=0.5
        )

        assert scored[3].best_zone == 0.0
        assert scored[5].best_zone == pytest.approx(1.0)
        assert [
            segment.kind
            for segment in segment_scored_frames(scored, change_threshold=0.25)
        ] == ["stable", "change", "stable", "change", "stable"]

    def test_zone_reset_does_not_erase_other_accumulating_changes(self) -> None:
        from custom_components.frigate_vision.changes import analyze_sequence

        frames = []
        for index in range(5):
            frame = _flat(80)
            frame[0:15, 0:40] = 80 + index * 6
            frame[20:30, 25:35] = 160 if index % 2 else 80
            frames.append((float(index), frame))
        scored = analyze_sequence(
            frames, zones={"door": self.ZONE}, anchor_reset_threshold=0.5
        )

        assert scored[3].cumulative > 0.5
        assert scored[2].best_zone == pytest.approx(1.0)

    def test_uniform_brightness_does_not_trigger_the_zone(self) -> None:
        from custom_components.frigate_vision.changes import (
            analyze_sequence,
            segment_scored_frames,
        )

        scored = analyze_sequence(
            [(0.0, _flat(80)), (1.0, _flat(120)), (2.0, _flat(160))],
            zones={"door": self.ZONE},
            anchor_reset_threshold=0.5,
        )
        assert all(frame.best_zone == 0.0 for frame in scored)
        assert [
            segment.kind
            for segment in segment_scored_frames(scored, change_threshold=0.25)
        ] == ["stable"]

    def test_preview_zone_settles_between_detection_and_reset_thresholds(self) -> None:
        from io import BytesIO

        from PIL import Image

        from custom_components.frigate_vision.media import _analyse_preview_frames

        closed = _flat(80)
        opened = _with_patch(80, (20, 25, 23, 35), 160)
        frames = []
        for index, frame in enumerate([closed, closed, opened, opened, closed]):
            buffer = BytesIO()
            Image.fromarray(frame).convert("RGB").save(buffer, "WEBP", lossless=True)
            frames.append((float(index), buffer.getvalue()))

        result = _analyse_preview_frames(frames, {"door": self.ZONE}, 0.25, 0.5)
        assert result is not None
        scored, segments = result
        assert 0.25 <= scored[2].best_zone < 0.5
        assert scored[3].best_zone == 0.0
        assert scored[4].best_zone >= 0.25
        assert [segment.kind for segment in segments] == [
            "stable", "change", "stable", "change"
        ]


_DOOR_POLYGON = [
    (0.181, 0.175),
    (0.407, 0.125),
    (0.442, 0.677),
    (0.456, 0.738),
    (0.533, 0.981),
    (0.286, 1.0),
]
_LIFT_POLYGON = [
    (0.411, 0.083),
    (0.68, 0.048),
    (0.662, 0.553),
    (0.713, 0.768),
    (0.516, 0.909),
    (0.471, 0.766),
    (0.445, 0.672),
]
