"""The sheet's unobserved stretches must stay bounded.

Reported twice by the household, once per layout, and the two reports have the
same shape -- a decisive moment fell into a stretch the sheet never observed:

* 九宫格 20:57 (review 1791377823.052639-3b8cll): cells 1 and 2 are 30.3 s
  apart, and lift 1 opening with the person walking out happened between them.
* 六宫格 20:41 (review 1791376906.007053-7xahuw): cells 2 and 3 are 52.9 s
  apart, and the walk in from the far corridor happened between them.

Both are reproduced below from the timestamps the deployment actually published,
so the tests fail on the real numbers rather than on a fixture invented to fit.

`MAX_SAMPLE_GAP` already declares the tolerance -- "The widest unobserved
stretch the sheet will tolerate" -- but it was only ever read as a reason to go
looking for probes, never as a property of the result. A sheet could therefore
be published with a 206 s hole in it, and 96 of 162 published sheets had one
wider than the declared bound. These tests pin the bound to the RESULT.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.media import (
    MAX_CELLS,
    MAX_SAMPLE_GAP,
    enforce_coverage,
    worst_unobserved_gap,
)

# The two reported activities, as the deployment actually published them.
# `first`/`last`/`postroll` are the plan's own window for each review.
SIX_GRID_FIRST = 1791376906.007
SIX_GRID_LAST = 1791376988.487
SIX_GRID_POSTROLL = 1791376991.487
SIX_GRID_PUBLISHED = (
    1791376904.022,
    1791376909.507,
    1791376962.388,
    1791376972.587,
    1791376979.120,
    1791376991.487,
)

NINE_GRID_FIRST = 1791377823.043
NINE_GRID_LAST = 1791377976.376
NINE_GRID_POSTROLL = 1791377979.376
NINE_GRID_PUBLISHED = (
    1791377823.043,
    1791377853.329,
    1791377864.420,
    1791377875.511,
    1791377886.603,
    1791377897.694,
    1791377941.864,
    1791377976.376,
    1791377979.376,
)


def _pool(start: float, end: float, step: float) -> tuple[float, ...]:
    """Preview frames across the window at a realistic cadence.

    Frigate writes a preview frame about every 1.1 s while something moves, which
    is what a person walking down the corridor produces.
    """
    count = int((end - start) / step)
    return tuple(start + index * step for index in range(count + 1))


# Recording coverage for the fixtures: one unbroken segment spanning the window
# and the postroll scene beyond it. Frigate really writes fixed-length segments
# with gaps between them, which is why `enforce_coverage` takes this at all --
# see `TestARepairMustStayFetchable`.
_COVERAGE = ((-10_000_000_000.0, 10_000_000_000.0),)


class TestWorstUnobservedGap:
    """The measure the whole guarantee is stated in."""

    def test_it_reports_the_widest_stretch_between_cells(self) -> None:
        span, gap_start, gap_end = worst_unobserved_gap(
            (0.0, 10.0, 100.0), start=0.0, end=120.0
        )
        assert span == 90.0
        assert (gap_start, gap_end) == (10.0, 100.0)

    def test_the_boundaries_count_as_cells(self) -> None:
        """`start` and `end` are the window's edges, not observations.

        The gap before the first cell and after the last are as unobserved as any
        hole between two cells: a person walking in from the corridor moves
        before the detector's first sample, and that is exactly the reported
        six-grid case.
        """
        span, gap_start, gap_end = worst_unobserved_gap(
            (50.0,), start=0.0, end=60.0
        )
        assert span == 50.0
        assert (gap_start, gap_end) == (0.0, 50.0)

    def test_the_postroll_beyond_the_window_is_not_a_hole(self) -> None:
        """The emptied scene sits past the activity on purpose.

        The sheet's last cell means "after the person left", so the stretch
        between the person's final frame and it is meant to be empty. Counting it
        would make every sheet fail, so the window bound -- not the outermost cell
        -- is what closes the measurement.
        """
        # The activity 0..40 is covered every 8 s; the postroll sits at 60.
        span, _, _ = worst_unobserved_gap(
            (0.0, 8.0, 16.0, 24.0, 32.0, 40.0, 60.0), start=0.0, end=40.0
        )
        assert span == 8.0, "40..60 是刻意留空的，不该计入"


class TestARepairMustStayFetchable:
    """A cell outside the recordings is worse than a wide hole.

    This is the one way the repair could make things worse rather than better, so
    it is pinned here. The build fetches a snapshot per cell and drops any cell
    outside recording coverage; a six-cell sheet that loses one to a gap is then
    either rejected as a partial row or -- worse -- published as a one-row sheet,
    which is the silent degradation `recording_gap` was introduced to stop. So a
    hole that cannot be filled with fetchable footage must be left alone.
    """

    def test_a_hole_with_no_coverage_is_left_alone(self) -> None:
        """Every candidate sits in a recording gap, so nothing may be added."""
        # Coverage stops at 30.0, so the whole second half of the window is
        # unfetchable even though the pool has frames there.
        coverage = ((0.0, 30.0),)
        published = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
        repaired = enforce_coverage(
            published,
            _pool(0.0, 60.0, 1.1),
            start=0.0,
            end=60.0,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(0.0, 25.0),
            usable=coverage,
        )
        for moment in repaired:
            assert any(start <= moment <= end for start, end in coverage), (
                f"修复把 {moment} 放进了录像缺口，会导致 recording_gap"
            )

    def test_without_coverage_information_nothing_is_added(self) -> None:
        """No coverage data means no cell can be shown to be fetchable.

        A caller that cannot say what is recorded cannot ask this function to add
        a cell without risking the whole activity, so it declines to act and the
        caller keeps its own sheet.
        """
        published = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0)
        assert (
            enforce_coverage(
                published,
                _pool(0.0, 60.0, 1.1),
                start=0.0,
                end=60.0,
                max_gap=MAX_SAMPLE_GAP,
                cap=MAX_CELLS,
                protect=(0.0, 25.0),
            )
            == published
        ), "没有录像覆盖信息时不应改动证据图"

    def test_a_partially_covered_hole_is_filled_from_the_covered_part(self) -> None:
        """The repair still helps when part of the hole is recorded."""
        # Coverage resumes at 35.0, so the 20..45 hole can take a cell at/after 35.
        coverage = ((0.0, 20.0), (35.0, 60.0))
        published = (0.0, 5.0, 10.0, 15.0, 20.0, 45.0)
        repaired = enforce_coverage(
            published,
            _pool(0.0, 45.0, 1.1),
            start=0.0,
            end=45.0,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(0.0, 45.0),
            usable=coverage,
        )
        before = worst_unobserved_gap(published, start=0.0, end=45.0)[0]
        after = worst_unobserved_gap(repaired, start=0.0, end=45.0)[0]
        assert after < before, f"有覆盖可用时仍没改善：{before:.1f}s -> {after:.1f}s"
        for moment in repaired:
            assert any(start <= moment <= end for start, end in coverage)


class TestTheReportedSheetsAreRepaired:
    """Both reports, on the timestamps the deployment really published."""

    def test_the_six_grid_loses_its_53_second_hole(self) -> None:
        """The walk in from the corridor must not fall between two cells.

        The bound IS reachable here, and the arithmetic is the reason it must be
        reached rather than merely improved. `n` cells inside the window create
        `n + 1` gaps -- both window EDGES are boundaries, and the person walks in
        before the detector's first sample -- so the best any sheet can do is
        `span / (n + 1)`. The window is 82.48 s, so nine cells with seven inside
        give 10.31 s, comfortably inside the 12 s bound.

        An earlier version of this test asserted the bound was UNREACHABLE, using
        `span / (n - 1)`. That was wrong by two gaps, and its leniency is exactly
        what let a 13.5 s result pass a test named after closing the hole. The
        assertion below is therefore the bound itself, not a formula that could
        drift back into excuses for missing it.
        """
        before = worst_unobserved_gap(
            SIX_GRID_PUBLISHED, start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        assert before == pytest.approx(52.88, abs=0.05), (
            "夹具必须复现用户报告的那个 53 秒空洞"
        )

        repaired = enforce_coverage(
            SIX_GRID_PUBLISHED,
            _pool(SIX_GRID_FIRST, SIX_GRID_LAST, 1.1),
            start=SIX_GRID_FIRST,
            end=SIX_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            # The full set the build passes: first, last AND postroll.
            protect=(SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        after = worst_unobserved_gap(
            repaired, start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        assert after <= MAX_SAMPLE_GAP, (
            f"最差空洞仍是 {after:.1f}s，超过 {MAX_SAMPLE_GAP}s；"
            "九格覆盖 82.5 秒窗口的理论上限是 10.3 秒，没有理由达不到"
        )
        assert len(repaired) == 9, f"应为九格，实际 {len(repaired)} 格"

        # Both out-of-window cells survive: they are part of the timeline the
        # prompt reads, and dropping one would move the sheet's own endpoints.
        for moment in SIX_GRID_PUBLISHED:
            if not SIX_GRID_FIRST <= moment <= SIX_GRID_LAST:
                assert moment in repaired, f"窗口外的 {moment} 被丢掉了"

    def test_the_motion_route_keeps_the_last_frame_the_layout_names(self) -> None:
        """The 末帧 must survive on the route where it is an actual cell.

        The README documents the layout as 「首帧 + 3 张高变化帧 + 末帧 + 后置现场帧」,
        and `choose_cells` deliberately does not reserve `last_time` (a change
        segment usually covers it). On the MOTION route the sheet is exactly
        first + 3 motion + last + postroll, and the caller's pool is the plan's
        own candidates -- so if the repair's spreading stage is allowed to move
        cells it is free to drop `last_time`.

        Measured before `plan.last_time` was protected: over 20k random
        motion-shaped sheets the relocation stage dropped it in 71.6% of them,
        silently removing the cell that says where the person was last seen.

        NOTE ON WHAT THIS DOES AND DOES NOT PIN. This calls the pure function with
        an explicit `protect` tuple, so it proves the function honours the list it
        is handed -- it cannot fail if `_async_enforce_coverage` stops passing
        `plan.last_time` again. That regression is pinned by
        `test_the_caller_protects_the_last_frame` in
        test_published_sheet_bound.py, which asserts the CALLER's tuple. Keep both:
        this one catches a function that ignores its argument, that one catches a
        caller that stops supplying it.

        The caveat on the measurement above: with `pool == sheet` the repair can
        add nothing and cannot improve the hole either, so on THIS fixture it
        returns the sheet untouched regardless. The 71.6% was measured on random
        motion-shaped sheets with a pool richer than the sheet, which is the
        production case; see that test's docstring for the three pool shapes and
        which one discriminates.
        """
        start, end = 100.0, 140.0
        postroll = 142.8
        plan_first = start
        plan_last = end
        # A motion route's sheet, as `_async_build_locked` builds it.
        sheet = (plan_first, 110.839, 113.479, 130.042, plan_last, postroll)
        repaired = enforce_coverage(
            sheet,
            # The motion route's pool is the plan's change candidates, which are
            # empty there -- so the sheet's own cells are all it has to choose from.
            sheet,
            start=start,
            end=end,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(plan_first, plan_last, postroll),
            usable=_COVERAGE,
        )
        assert plan_last in repaired, (
            f"末帧 {plan_last} 被丢掉了；图里只剩 {sorted(repaired)}"
        )
        assert plan_first in repaired, "首帧被丢掉了"
        assert postroll in repaired, "后置现场那一格被丢掉了"

    def test_an_unprotected_out_of_window_cell_would_be_dropped(self) -> None:
        """The union is load-bearing, shown by removing what it protects.

        `enforce_coverage` adds every out-of-window cell to `fixed` itself, so the
        caller does not have to name them. To show that is doing work rather than
        being decorative, this takes a sheet whose spreading stage genuinely acts
        and confirms the cell survives -- and then confirms the same sheet would
        lose it if the spreading stage were free to choose it away.

        A fixture where the stage does NOT act would pass either way and pin
        nothing, which is the trap this test exists to avoid: measured, the
        earlier version of this test passed even with the union removed.
        """
        # Two anchors present, plus a pre-roll cell before the window, and a hole
        # wide enough that the spreading stage has to act.
        start, end, postroll = 100.0, 150.0, 152.8
        pre_roll = 97.5
        sheet = (pre_roll, start, 104.0, 108.0, end, postroll)
        pool = tuple(start + (end - start) * index / 60 for index in range(61))

        repaired = enforce_coverage(
            sheet,
            pool,
            start=start,
            end=end,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(start, end, postroll),
            usable=_COVERAGE,
        )
        before = worst_unobserved_gap(sheet, start=start, end=end)[0]
        after = worst_unobserved_gap(repaired, start=start, end=end)[0]
        assert before > MAX_SAMPLE_GAP, (
            "夹具前提变了：这张图本该有超过上限的盲区，好让重排阶段真的动手"
        )
        assert after < before, (
            f"重排阶段没有动手（{before:.1f}s -> {after:.1f}s），"
            "这个夹具就证明不了窗口外的格子是否被保住"
        )
        assert pre_roll in repaired, (
            f"窗口前的预卷格 {pre_roll} 被丢掉了；图里只剩 {sorted(repaired)}"
        )

    def test_every_out_of_window_cell_survives(self) -> None:
        """A cell outside the window cannot cover anything, but it must stay.

        It is part of the timeline the prompt reads -- the preview route reaches
        back before the person's first sample on purpose, so the sheet's own first
        cell is often a few seconds before `plan.first_time`. Dropping it to make
        room would move the sheet's endpoints, which is not this function's to
        change. Pinned as its own test because the earlier version of this file
        asserted it only on one fixture, where it happened to hold for a different
        reason.
        """
        start, end, postroll = 100.0, 158.946, 182.8
        pre_roll = 99.718
        sheet = (pre_roll, start, 113.79, 137.432, end, postroll)
        repaired = enforce_coverage(
            sheet,
            sheet,
            start=start,
            end=end,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(start, end, postroll),
            usable=_COVERAGE,
        )
        assert pre_roll in repaired, (
            f"窗口前的预卷格 {pre_roll} 被丢掉了；图里只剩 {sorted(repaired)}"
        )
        assert end in repaired, "末帧被丢掉了"
        assert postroll in repaired, "后置现场那一格被丢掉了"

    def test_the_nine_grid_loses_its_30_second_hole(self) -> None:
        """Lift 1 opening with the person walking out must be observed.

        This window is 153.3 s and the 12 s bound is genuinely out of reach, but
        the arithmetic has to be stated carefully because the sheet's first and
        last cells sit exactly ON the two window edges -- they ARE those
        boundaries rather than extra ones -- and the postroll cell lies outside the
        window. So the cells that partition the span are the ones STRICTLY inside,
        and the fixture builds six of them: 6 interior cells give 7 gaps, and
        `153.3 / 7 = 21.9 s`.

        The earlier version of this test used `span / (n - 1) = 21.9 s` too, for
        the wrong reason (counting all nine cells and subtracting one), and the
        version after that said `153.3 / 9 = 17.0 s` from "eight inside", which is
        not the sheet this fixture builds. Both got the right answer from the wrong
        count; this one counts what is actually there.
        """
        before = worst_unobserved_gap(
            NINE_GRID_PUBLISHED, start=NINE_GRID_FIRST, end=NINE_GRID_LAST
        )[0]
        assert before == pytest.approx(44.17, abs=0.05), (
            "夹具必须复现用户报告的九宫格空洞"
        )

        repaired = enforce_coverage(
            NINE_GRID_PUBLISHED,
            _pool(NINE_GRID_FIRST, NINE_GRID_LAST, 1.1),
            start=NINE_GRID_FIRST,
            end=NINE_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(NINE_GRID_FIRST, NINE_GRID_LAST, NINE_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        after = worst_unobserved_gap(
            repaired, start=NINE_GRID_FIRST, end=NINE_GRID_LAST
        )[0]
        span = NINE_GRID_LAST - NINE_GRID_FIRST
        # The limit is derived from the count the sheet is REQUIRED to have, not
        # from whatever the repair returned. Deriving it from the output makes the
        # assertion self-referential: a repair that dropped interior cells would
        # widen its own limit and pass more easily (0 interior cells would give a
        # limit of `span`, which nothing could exceed).
        #
        # The sheet must be a full nine, and its first and last cells sit exactly ON
        # the two edges, so 6 cells are strictly inside and partition the span into
        # 7 gaps.
        assert len(repaired) == MAX_CELLS, (
            f"九宫格窗口应当补满 {MAX_CELLS} 格，实际 {len(repaired)} 格"
        )
        interior_required = MAX_CELLS - 3
        gaps = interior_required + 1
        limit = span / gaps
        assert limit > MAX_SAMPLE_GAP, (
            "夹具前提变了：九格本该覆盖不了这个 153 秒的窗口"
        )
        interior = [
            moment
            for moment in repaired
            if NINE_GRID_FIRST < moment < NINE_GRID_LAST
        ]
        assert len(interior) >= interior_required, (
            f"严格落在区间内的格子只有 {len(interior)} 个，"
            f"少于应有的 {interior_required} 个；这样上限是被自己撑大的"
        )
        assert after <= limit * 1.05, (
            f"留下 {after:.1f}s 空洞，而 {gaps} 段的理论上限是 {limit:.1f}s"
        )

    def test_a_window_too_long_for_the_budget_is_improved_not_faked(self) -> None:
        """Closing the bound must never place a cell outside the activity.

        The nine-grid is over budget, so the repair settles for the best it can
        do. What it must not do to appear to hit the bound is spread past the
        window or invent cells: every cell has to be a real, fetchable instant
        inside the activity (plus the deliberately-protected postroll scene).
        """
        repaired = enforce_coverage(
            NINE_GRID_PUBLISHED,
            _pool(NINE_GRID_FIRST, NINE_GRID_LAST, 1.1),
            start=NINE_GRID_FIRST,
            end=NINE_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(NINE_GRID_FIRST, NINE_GRID_LAST, NINE_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        assert all(
            NINE_GRID_FIRST <= moment <= NINE_GRID_POSTROLL for moment in repaired
        ), "修复不得把格子放到活动窗口之外"
        assert len(repaired) == 9, f"应补满九格，实际 {len(repaired)} 格"


class TestTheRepairKeepsTheSheetsContract:
    """The invariants every downstream stage relies on."""

    def test_it_keeps_every_protected_cell_it_was_given(self) -> None:
        """Cell 1 is the person's first frame and the last cell the empty scene.

        Both are present on the nine-grid, and the prompt's rules read them by
        name, so the repair must not trade either away for coverage.
        """
        repaired = enforce_coverage(
            NINE_GRID_PUBLISHED,
            _pool(NINE_GRID_FIRST, NINE_GRID_LAST, 1.1),
            start=NINE_GRID_FIRST,
            end=NINE_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(NINE_GRID_FIRST, NINE_GRID_LAST, NINE_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        assert NINE_GRID_FIRST in repaired, "人物首帧被丢掉了"
        assert NINE_GRID_POSTROLL in repaired, "后置现场那一格被丢掉了"
        assert repaired[0] == NINE_GRID_FIRST
        assert repaired[-1] == NINE_GRID_POSTROLL

    def test_it_keeps_the_sheets_own_extremes_when_an_anchor_is_absent(self) -> None:
        """A protected cell the sheet does not contain cannot be preserved.

        The reported six-grid is this case: its first cell is a preview frame
        ~2 s *before* `plan.first_time`, because the preview window deliberately
        reaches back past the person's first sample to catch the moment they
        appeared. Only the postroll anchor survives the intersection, so the
        sheet's own extremes stand in -- and they must be kept, because dropping
        them would move both ends of the timeline the prompt reads.
        """
        repaired = enforce_coverage(
            SIX_GRID_PUBLISHED,
            _pool(SIX_GRID_FIRST, SIX_GRID_LAST, 1.1),
            start=SIX_GRID_FIRST,
            end=SIX_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        assert repaired[0] == SIX_GRID_PUBLISHED[0]
        assert repaired[-1] == SIX_GRID_PUBLISHED[-1]

    def test_it_still_lays_out_as_whole_rows(self) -> None:
        """A partial row cannot be composed, so the count stays a multiple of 3."""
        for published, first, last, postroll in (
            (SIX_GRID_PUBLISHED, SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            (NINE_GRID_PUBLISHED, NINE_GRID_FIRST, NINE_GRID_LAST, NINE_GRID_POSTROLL),
        ):
            repaired = enforce_coverage(
                published,
                _pool(first, last, 1.1),
                start=first,
                end=last,
                max_gap=MAX_SAMPLE_GAP,
                cap=MAX_CELLS,
                protect=(first, postroll),
                usable=_COVERAGE,
            )
            assert len(repaired) % 3 == 0, (
                f"格子数 {len(repaired)} 不是三的倍数"
            )
            assert len(repaired) in {6, 9}, f"意外的格子数 {len(repaired)}"

    def test_it_stays_strictly_increasing_and_never_duplicates(self) -> None:
        """The sheet is read as a timeline, and the guard rejects duplicates."""
        repaired = enforce_coverage(
            SIX_GRID_PUBLISHED,
            _pool(SIX_GRID_FIRST, SIX_GRID_LAST, 1.1),
            start=SIX_GRID_FIRST,
            end=SIX_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        assert list(repaired) == sorted(repaired)
        assert len(set(repaired)) == len(repaired)

    def test_a_sheet_that_already_covers_its_window_is_untouched(self) -> None:
        """The change-aware selection must survive where coverage already holds.

        The repair exists to bound holes, not to replace the choice of what is
        worth showing. A sheet whose holes are all inside the bound must come
        back byte-identical, or every activity would be re-aimed for nothing.
        """
        tight = (0.0, 8.0, 16.0, 24.0, 32.0, 35.8)
        assert (
            enforce_coverage(
                tight,
                _pool(0.0, 32.0, 1.1),
                start=0.0,
                end=32.0,
                max_gap=MAX_SAMPLE_GAP,
                cap=MAX_CELLS,
                protect=(0.0, 35.8),
                usable=_COVERAGE,
            )
            == tight
        )

    def test_it_never_makes_a_sheet_worse(self) -> None:
        """Monotone: the repair can only reduce the worst hole."""
        cases = (
            (SIX_GRID_PUBLISHED, SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            (NINE_GRID_PUBLISHED, NINE_GRID_FIRST, NINE_GRID_LAST, NINE_GRID_POSTROLL),
        )
        for published, first, last, postroll in cases:
            before = worst_unobserved_gap(published, start=first, end=last)[0]
            repaired = enforce_coverage(
                published,
                _pool(first, last, 1.1),
                start=first,
                end=last,
                max_gap=MAX_SAMPLE_GAP,
                cap=MAX_CELLS,
                protect=(first, postroll),
                usable=_COVERAGE,
            )
            after = worst_unobserved_gap(repaired, start=first, end=last)[0]
            assert after <= before, f"修复后更差：{before:.1f}s -> {after:.1f}s"

    def test_with_no_usable_candidate_it_prefers_the_published_sheet(self) -> None:
        """Nothing may be invented when there is nothing to choose from.

        With an empty pool the only option left is a hole's geometric midpoint,
        and a midpoint is a real instant that can be fetched -- so the repair may
        still act, but what it must never do is place a cell outside the activity,
        drop a protected cell without a replacement, or return a partial row. In
        every case the result is an improvement or the original sheet.
        """
        repaired = enforce_coverage(
            SIX_GRID_PUBLISHED,
            (),
            start=SIX_GRID_FIRST,
            end=SIX_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(SIX_GRID_PUBLISHED[0], SIX_GRID_LAST, SIX_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        assert repaired[0] == SIX_GRID_PUBLISHED[0]
        assert repaired[-1] == SIX_GRID_PUBLISHED[-1]
        assert len(repaired) % 3 == 0
        assert all(
            SIX_GRID_FIRST - 2.0 <= moment <= SIX_GRID_POSTROLL for moment in repaired
        ), "修复把格子放到了活动窗口之外"
        before = worst_unobserved_gap(
            SIX_GRID_PUBLISHED, start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        after = worst_unobserved_gap(
            repaired, start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        assert after <= before


class TestTheFinalSheetIsMeasurable:
    """The bound is a pre-pass, so what the sheet ENDED UP with is recorded.

    `_async_enforce_coverage` runs before the night-vision, exposure and
    uniqueness passes, and each of those can move a cell by up to 15 s. Nothing
    re-runs the repair afterwards -- a second pass would fight those ladders for
    the same cells -- so the artifact instead carries the FINAL measurement. A
    sheet that ends up over the bound is then a fact in the metadata rather than
    something only a reader of the image could notice.
    """

    async def test_the_metadata_records_the_final_gap(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        import json

        from custom_components.frigate_vision.correlation import ZoneRoles
        from custom_components.frigate_vision.media import MediaManager
        from custom_components.frigate_vision.models import (
            ActivityRecord,
            ActivitySource,
            ActivityStage,
        )
        from custom_components.frigate_vision.store import ActivityStore

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = ActivityRecord(
            activity_id="door_entry_1_100000",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.SEALED,
            created_at=100,
            updated_at=130,
            camera="front",
            detection_ids=("event_1",),
            association_deadline=140,
            finalization_deadline=220,
        )
        await store.async_create(record)

        from io import BytesIO

        from PIL import Image

        def jpeg(colour: int) -> bytes:
            buffer = BytesIO()
            Image.new("RGB", (640, 360), (colour, colour, colour)).save(
                buffer, "JPEG"
            )
            return buffer.getvalue()

        class Client:
            async def async_get_event(self, event_id: str, camera: str):
                return {
                    "id": event_id,
                    "camera": camera,
                    "label": "person",
                    "start_time": 101,
                    "end_time": 121,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": 0, "end_time": 300}]

            async def async_get_snapshot(self, camera, timestamp, height):
                return jpeg(int(timestamp) % 255)

        manager = MediaManager(
            hass,
            store,
            Client(),
            tmp_path,
            ZoneRoles(
                frozenset({"near"}), frozenset({"mid"}), frozenset({"far"})
            ),
        )
        built = await manager.async_build(record.activity_id)
        payload = json.loads(
            Path(built.evidence_path).with_suffix(".json").read_text("utf-8")
        )
        coverage = payload["coverage"]
        assert coverage["limit_seconds"] == MAX_SAMPLE_GAP
        # The recorded window must be the PERSON's span the repair was given, not
        # something re-derived from the sheet. On the preview route the sheet's
        # first cell is a pre-roll frame chosen BEFORE `plan.first_time`, so the
        # two are genuinely different numbers -- a test that could not tell them
        # apart would not be pinning this field at all.
        #
        # This fixture's person window is the event's own span, 101..121, with the
        # 0.2 s edge applied on each side.
        window_start = 101 + min(0.2, 20 * 0.05)
        window_end = 121 - min(0.2, 20 * 0.05)
        assert coverage["window_seconds"] == pytest.approx(
            window_end - window_start, abs=0.05
        ), "coverage.window_seconds 应当是人物活动区间，而不是图自己的跨度"
        # The recorded number must BE the repair's own measurement over that
        # window -- recomputed here from the stored times and the same bounds.
        expected = worst_unobserved_gap(
            built.sample_times, start=window_start, end=window_end
        )[0]
        assert coverage["max_gap_seconds"] == pytest.approx(expected, abs=0.01)

    def test_a_pre_pass_bound_is_not_a_final_guarantee(self) -> None:
        """Why the metadata exists rather than a second repair pass.

        The repair closes the hole it can see, and a later IR or uniqueness
        replacement may widen it again. This demonstrates the gap between "the
        repair met the bound" and "the published sheet meets it" on a sheet whose
        later repair moved a cell -- which is precisely what the recorded
        measurement makes visible instead of silent.
        """
        # The repaired sheet, then a later ladder moving one middle cell 10 s
        # later (within UNIQUENESS_RETRY_OFFSETS' reach).
        repaired = enforce_coverage(
            SIX_GRID_PUBLISHED,
            _pool(SIX_GRID_FIRST, SIX_GRID_LAST, 1.1),
            start=SIX_GRID_FIRST,
            end=SIX_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        moved = list(repaired)
        moved[3] = moved[3] + 10.0
        moved.sort()
        after_repair = worst_unobserved_gap(
            repaired, start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        after_ladder = worst_unobserved_gap(
            tuple(moved), start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        assert after_repair <= MAX_SAMPLE_GAP
        # The point is that the bound can be LOST, not merely that the number can
        # change: a ladder move that happened to improve the gap would satisfy a
        # weaker `!=` assertion while proving nothing about the risk.
        assert after_ladder > MAX_SAMPLE_GAP, (
            f"夹具前提变了：后置修复本应把盲区推过上限，现在是 "
            f"{after_repair:.1f}s -> {after_ladder:.1f}s"
        )


class TestThePoolReachesTheRepair:
    """The repair is only as good as the instants it is allowed to choose from.

    This is the difference between the fix working and merely appearing to. The
    repair can only place a cell on a timestamp it is handed, so if the caller
    offers only the cells the change selector already picked, the repair is
    choosing from exactly the set that leaves the wide holes -- and it changes
    nothing.

    Measured by replaying all 160 published sheets against the two pools:

      pool = the chosen cells only     over-bound 92/160 (58%) -> 92/160 (58%)
      pool = every frame analysed      over-bound 92/160 (58%) -> 32/160 (20%)

    `_async_preview_timestamps` already decodes up to PREVIEW_FRAME_BUDGET (120)
    preview frames -- the reported 7xahuw artifact records 80 analysed -- so
    offering them costs no extra fetch: they were paid for before the repair runs.
    """

    def test_the_repair_is_given_more_than_the_chosen_cells(self) -> None:
        """The build must hand the repair the frames it analysed, not just its picks.

        A structural assertion, for the same reason as the last-frame one: the
        difference is invisible to a behavioural fixture, because a build whose
        pool happens to be rich still meets the bound. What must not regress is
        that the richer pool is passed at all.
        """
        import inspect

        from custom_components.frigate_vision import media as media_module

        source = inspect.getsource(
            media_module.MediaManager._async_preview_timestamps
        )
        assert '"_analysed_times"' in source, (
            "the preview analysis stopped publishing the frames it decoded, so "
            "the coverage repair is back to choosing among the selected cells"
        )
        build = inspect.getsource(media_module.MediaManager._async_build_locked)
        assert "preview_pool" in build, (
            "the build no longer passes the analysed preview frames to the "
            "coverage repair"
        )
        assert "*preview_pool," in build, (
            "the analysed frames are computed but not handed to the repair"
        )

    def test_a_pool_of_only_the_chosen_cells_cannot_close_the_reported_hole(
        self,
    ) -> None:
        """The measured claim behind the test above, on the reported six-grid.

        With only the cells the selector chose, the repair has nothing new to place
        and the 52.9 s hole survives untouched. With the frames the analysis
        already decoded, the same repair reaches 11.0 s -- inside the bound.
        """
        chosen_only = enforce_coverage(
            SIX_GRID_PUBLISHED,
            SIX_GRID_PUBLISHED,
            start=SIX_GRID_FIRST,
            end=SIX_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        with_analysed = enforce_coverage(
            SIX_GRID_PUBLISHED,
            _pool(SIX_GRID_FIRST, SIX_GRID_LAST, 1.1),
            start=SIX_GRID_FIRST,
            end=SIX_GRID_LAST,
            max_gap=MAX_SAMPLE_GAP,
            cap=MAX_CELLS,
            protect=(SIX_GRID_FIRST, SIX_GRID_LAST, SIX_GRID_POSTROLL),
            usable=_COVERAGE,
        )
        selected_only = worst_unobserved_gap(
            chosen_only, start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        analysed = worst_unobserved_gap(
            with_analysed, start=SIX_GRID_FIRST, end=SIX_GRID_LAST
        )[0]
        assert selected_only == pytest.approx(52.88, abs=0.05), (
            "只交出已选中的格子时，修复应该无从下手，53 秒空洞原样保留"
        )
        assert analysed <= MAX_SAMPLE_GAP, (
            f"交出错过的全部帧后仍留下 {analysed:.1f}s 空洞"
        )


class TestHowLongASheetCanCover:
    """The reach of each sheet size, measured rather than reasoned about.

    The prose in the README and in `enforce_coverage`'s docstring quotes this
    number, and it has been stated wrongly TWICE -- once as "six cells cannot cover
    more than 60 s", once as "six cells cover 84 s and nine cover 120 s". Both came
    from reasoning about `n` cells instead of counting the cells that actually
    partition the span:

      * the first cell sits exactly ON the window's start edge, and the last ON its
        end edge, so they ARE those two boundaries rather than extra ones;
      * the postroll cell lies OUTSIDE the window and closes nothing inside it.

    So a six-cell sheet has only 3 cells strictly inside -- 4 gaps -- and a nine-cell
    sheet has 6 -- 7 gaps. These tests assert the measured reach directly, so the
    arithmetic cannot drift back into prose nobody re-measures.
    """

    @staticmethod
    def _worst_for(size: int, span: float, cap: int) -> float:
        start = 1000.0
        end = start + span
        postroll = end + 2.8
        interior = max(0, size - 3)
        cells = [start, end, postroll]
        cells += [start + span * (i + 1) / (interior + 1) for i in range(interior)]
        repaired = enforce_coverage(
            tuple(sorted(cells)),
            tuple(start + span * i / 400 for i in range(401)),
            start=start,
            end=end,
            max_gap=MAX_SAMPLE_GAP,
            cap=cap,
            protect=(start, end, postroll),
            usable=_COVERAGE,
        )
        return worst_unobserved_gap(repaired, start=start, end=end)[0]

    def test_six_cells_cover_48_seconds_and_not_84(self) -> None:
        """The head-room a six-cell sheet really has, with growth disabled.

        `cap=6` is the whole point: measuring a six-cell sheet with `cap=9` reports
        the reach of a NINE-cell sheet, because the repair tops it up. That
        conflation is how "six cells cover 84 s" got written down.
        """
        assert self._worst_for(6, 47.5, cap=6) <= MAX_SAMPLE_GAP, (
            "六格本该能覆盖 47.5 秒"
        )
        assert self._worst_for(6, 48.5, cap=6) > MAX_SAMPLE_GAP, (
            "六格本不该覆盖 48.5 秒；若这里通过，说明 48 秒这个数写错了"
        )

    def test_nine_cells_cover_84_seconds(self) -> None:
        """Nine cells reach exactly 7 x 12 s, and no further."""
        assert self._worst_for(9, 83.5, cap=9) <= MAX_SAMPLE_GAP, (
            "九格本该能覆盖 83.5 秒"
        )
        assert self._worst_for(9, 84.5, cap=9) > MAX_SAMPLE_GAP, (
            "九格本不该覆盖 84.5 秒"
        )

    def test_growing_a_six_cell_sheet_is_what_buys_the_long_reach(self) -> None:
        """A 6-cell sheet grown to 9 covers 82.5 s where 6 alone covered 48 s.

        This is the reason the repair grows the sheet rather than reporting the
        best six-cell placement: at six cells a typical activity is already over
        budget, so the extra row is the difference between meeting the bound and
        documenting a hole.
        """
        assert self._worst_for(6, 47.5, cap=6) <= MAX_SAMPLE_GAP
        assert self._worst_for(6, 82.5, cap=MAX_CELLS) <= MAX_SAMPLE_GAP, (
            "补满九格后本该能覆盖 82.5 秒"
        )
