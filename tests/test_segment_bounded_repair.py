"""Tests for the repair constraint that keeps a frame in its own moment.

The defect these pin is the one the household noticed first: a frame chosen
because it shows the door opening gets replaced by a frame five seconds later,
because the repair's only test was "is this frame no longer infrared". The
replacement passes every existing check -- it is real, unique, colour, inside the
recording -- while the cell no longer depicts the event it was selected for.

`INFRARED_RETRY_OFFSETS` walks up to five seconds, so the ladder itself provides
no protection: it accepts the first offset that clears the test.
"""

from __future__ import annotations

from io import BytesIO

import pytest
from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.changes import Segment
from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import (
    MediaError,
    MediaManager,
    assert_cells_describe_the_person,
    frame_is_infrared,
    segment_bounds_for,
    within_segment,
)


def _jpeg(value: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (64, 36), (value, value, value)).save(buffer, "JPEG")
    return buffer.getvalue()


def _ir_jpeg(value: int = 160) -> bytes:
    """A frame with the camera's night-vision cast (R-G well below -60)."""
    buffer = BytesIO()
    Image.new("RGB", (64, 36), (5, value, 105)).save(buffer, "JPEG")
    return buffer.getvalue()


# A segment structure with two events far apart: an opening door (100-103) and a
# person crouching much later (110-113). This is the shape the defect needs -- a
# repair that steps out of the first must land in the second.
SEGMENTS = [
    Segment(start=100.0, end=103.0, kind="change", score=0.9, indices=(0, 1, 2)),
    Segment(start=103.0, end=110.0, kind="stable", score=0.0, indices=(3, 4)),
    Segment(start=110.0, end=113.0, kind="change", score=0.8, indices=(5, 6, 7)),
]


class TestSegmentBounds:
    """The lookup that decides what "the same moment" means."""

    def test_a_moment_inside_a_segment_finds_its_bounds(self) -> None:
        assert segment_bounds_for(SEGMENTS, 101.5) == (100.0, 103.0)

    def test_a_moment_in_the_stable_stretch_finds_that_one(self) -> None:
        assert segment_bounds_for(SEGMENTS, 106.0) == (103.0, 110.0)

    def test_a_moment_outside_every_segment_finds_nothing(self) -> None:
        assert segment_bounds_for(SEGMENTS, 99.0) is None

    def test_no_segments_means_no_constraint(self) -> None:
        """With no structure to compare against, the old rules are all there is.

        Returning False here would block every repair on a plan built without the
        preview analysis -- a silent regression for every deployment that does not
        use it.
        """
        assert within_segment(105.0, 101.0, ()) is True

    def test_a_candidate_inside_the_same_segment_is_accepted(self) -> None:
        assert within_segment(102.5, 101.0, SEGMENTS) is True

    def test_a_candidate_in_a_later_segment_is_refused(self) -> None:
        """THE defect: the door frame must not become the crouching frame."""
        assert within_segment(111.0, 101.0, SEGMENTS) is False

    def test_a_candidate_in_the_stable_stretch_is_refused(self) -> None:
        """A different moment is a different moment, changed or not."""
        assert within_segment(106.0, 101.0, SEGMENTS) is False

    def test_a_moment_with_no_segment_is_unconstrained(self) -> None:
        """A frame outside every segment cannot be pinned to one."""
        assert within_segment(200.0, 99.0, SEGMENTS) is True


class TestTheRepairStaysInItsMoment:
    """End to end through `_async_retry_frames`."""

    async def _manager(self, hass: HomeAssistant, tmp_path, client) -> MediaManager:
        from custom_components.frigate_vision.store import ActivityStore

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        return MediaManager(
            hass,
            store,
            client,
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )

    async def test_an_ir_frame_is_not_replaced_across_a_segment(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """The reported symptom, driven through the real repair pass.

        The frame at 101.0 is infrared, inside a change segment that ends at 103.
        The ladder reaches 106, 110 and beyond -- instants in LATER segments -- and
        every one of them is refused. That refusal has to happen before the fetch
        (a candidate in another moment is not worth a request) and it has to be
        recorded, or the metadata looks like the ladder simply found nothing.

        The neighbours are far apart on purpose. The neighbour window is set by the
        caller's own frame spacing, and a tight one would leave every
        cross-segment candidate outside the window entirely -- the constraint
        would never be reached and the test would pass without observing anything.
        An earlier version of this test used 100.5/102.5 as neighbours, which is
        exactly that mistake.
        """
        manager = await self._manager(hass, tmp_path, None)

        class Client:
            def __init__(self) -> None:
                self.requested: list[float] = []

            async def async_get_snapshot(self, camera, timestamp, height):
                self.requested.append(timestamp)
                # Colour far away, infrared near, so a rejection cannot be
                # confused with "nothing acceptable was ever fetched".
                return _jpeg(120) if timestamp >= 108.0 else _ir_jpeg()

        client = Client()
        manager._client = client
        manager._segments = list(SEGMENTS)

        ir_path = tmp_path / "ir.jpg"
        ir_path.write_bytes(_ir_jpeg())
        neighbour = tmp_path / "ok.jpg"
        neighbour.write_bytes(_jpeg(40))
        # A wide neighbour window, so the ladder really can reach past 103.
        ordered = [(100.0, neighbour), (101.0, ir_path), (112.0, neighbour)]

        result, replacements = await manager._async_retry_frames(
            "front",
            tmp_path,
            ordered,
            [{"start_time": 90.0, "end_time": 130.0}],
            offsets=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 9.0, 10.0),
            tag="ir",
            needed=lambda _i, path, _r: frame_is_infrared(path),
            acceptable=lambda path, _i, _r: not frame_is_infrared(path),
            segments=manager._segments,
        )

        assert replacements == [], (
            "跨段的彩色候选必须被拒绝，即使它们本身完全可用"
        )
        assert 100.0 <= result[1][0] <= 103.0, (
            f"帧必须留在同一段内，实际 {result[1][0]}"
        )
        rejected = manager._last_cross_segment_rejected
        assert rejected, "拒绝必须被记录，否则元数据看起来像「阶梯没找到」"
        assert all(item["code"] == "cross_segment_rejected" for item in rejected)
        assert all(not (100.0 <= item["to"] <= 103.0) for item in rejected), (
            "被记录的必须全部是跨段候选"
        )
        # And no distant instant was ever fetched: the refusal happens before the
        # HTTP request, so a cross-segment candidate costs nothing.
        assert not any(moment >= 108.0 for moment in client.requested), (
            "跨段候选必须在取图之前就被拒绝，不该产生请求"
        )

    async def test_a_colour_frame_inside_the_segment_is_accepted(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """The constraint must not block the repair it is meant to allow."""
        manager = await self._manager(hass, tmp_path, None)

        class Client:
            async def async_get_snapshot(self, camera, timestamp, height):
                # Colour from 101.5 onward, still inside the same segment.
                return _jpeg(120) if timestamp >= 101.5 else _ir_jpeg()

        manager._client = Client()
        manager._segments = list(SEGMENTS)

        ir_path = tmp_path / "ir2.jpg"
        ir_path.write_bytes(_ir_jpeg())
        neighbour = tmp_path / "ok2.jpg"
        neighbour.write_bytes(_jpeg(40))
        ordered = [(100.5, neighbour), (101.0, ir_path), (102.5, neighbour)]

        result, replacements = await manager._async_retry_frames(
            "front",
            tmp_path,
            ordered,
            [{"start_time": 90.0, "end_time": 130.0}],
            offsets=(0.5, 1.0, 2.0),
            tag="ir",
            needed=lambda _i, path, _r: frame_is_infrared(path),
            acceptable=lambda path, _i, _r: not frame_is_infrared(path),
            segments=manager._segments,
        )

        assert replacements, "同段内更好的候选必须被接受，否则约束就成了禁用修复"
        assert result[1][0] > 101.0
        assert result[1][0] <= 103.0, "替换必须留在同一段内"

    async def test_without_segments_the_old_behaviour_is_unchanged(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Zero regression: a plan with no preview analysis repairs as before.

        This is the case for every existing deployment and every existing test
        double, so the constraint must be inert rather than restrictive when there
        is no segment structure. The fixture leaves real room on both sides -- the
        neighbours are far apart -- so the ladder is limited only by the recording,
        which is what it was before.
        """
        manager = await self._manager(hass, tmp_path, None)

        class Client:
            async def async_get_snapshot(self, camera, timestamp, height):
                return _jpeg(120) if timestamp > 140.0 else _ir_jpeg()

        manager._client = Client()
        manager._segments = []

        ir_path = tmp_path / "ir3.jpg"
        ir_path.write_bytes(_ir_jpeg())
        neighbour = tmp_path / "ok3.jpg"
        neighbour.write_bytes(_jpeg(40))
        # 150s of room ahead, so an unconstrained ladder can reach past 140.
        ordered = [(100.5, neighbour), (101.0, ir_path), (200.0, neighbour)]

        result, replacements = await manager._async_retry_frames(
            "front",
            tmp_path,
            ordered,
            [{"start_time": 90.0, "end_time": 260.0}],
            offsets=tuple(30.0 + index for index in range(15)),
            tag="ir",
            needed=lambda _i, path, _r: frame_is_infrared(path),
            acceptable=lambda path, _i, _r: not frame_is_infrared(path),
            segments=manager._segments,
        )

        assert replacements, "没有段落信息时，原有的阶梯行为必须保持不变"
        assert result[1][0] > 130.0, (
            f"无约束时阶梯应能走到远处，实际 {result[1][0]}"
        )


class TestSegmentsDoNotLeakBetweenBuilds:
    """A manager serves many activities; a stale structure would corrupt one."""

    async def _manager(self, hass: HomeAssistant, tmp_path) -> MediaManager:
        from custom_components.frigate_vision.store import ActivityStore

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        return MediaManager(
            hass,
            store,
            None,
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )

    async def test_a_build_starts_without_a_segment_structure(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        from custom_components.frigate_vision.media import MediaError

        manager = await self._manager(hass, tmp_path)
        manager._segments = list(SEGMENTS)
        manager._last_cross_segment_rejected = [{"stale": True}]

        # Any build resets both, whatever else it does -- including failing.
        with pytest.raises(MediaError):
            await manager.async_build("does_not_exist")

        assert manager._segments == [], "上一次活动的段落结构不能留给下一次"
        assert manager._last_cross_segment_rejected == []


class TestASelectionMustBeAbleToAnswerTheQuestion:
    """A sheet with no person, or no departure, cannot support the scene rules.

    The prompt tells the model what the ends of the sheet are and reads arrival
    and departure from them. A selection missing either property is still a valid
    JPEG of six real frames, so nothing downstream complains -- the model is
    simply asked a question its evidence cannot answer and answers anyway. These
    errors make that visible instead.
    """

    def test_a_normal_selection_passes(self) -> None:
        assert_cells_describe_the_person(
            (100.0, 105.0, 110.0, 115.0, 118.0, 120.8),
            person_start=100.0,
            person_end=118.0,
            departure=120.8,
        )

    def test_a_sheet_with_no_person_is_refused(self) -> None:
        """Nothing inside the window means nothing to judge."""
        with pytest.raises(MediaError, match="no_person_frame"):
            assert_cells_describe_the_person(
                (90.0, 92.0, 94.0, 96.0, 98.0, 120.8),
                person_start=100.0,
                person_end=118.0,
                departure=120.8,
            )

    def test_a_sheet_with_no_departure_is_refused(self) -> None:
        """All cells before the person left: "still here" is indistinguishable."""
        with pytest.raises(MediaError, match="no_departure_frame"):
            assert_cells_describe_the_person(
                (100.0, 102.0, 104.0, 106.0, 108.0, 110.0),
                person_start=100.0,
                person_end=118.0,
                departure=120.8,
            )

    def test_an_empty_sheet_is_refused(self) -> None:
        with pytest.raises(MediaError, match="no_person_frame"):
            assert_cells_describe_the_person(
                (), person_start=100.0, person_end=118.0, departure=120.8
            )

    def test_a_cell_at_the_departure_instant_counts(self) -> None:
        """The bound is inclusive: the frame AT departure shows the emptied scene."""
        assert_cells_describe_the_person(
            (105.0, 110.0, 115.0, 118.0),
            person_start=100.0,
            person_end=118.0,
            departure=118.0,
        )

    def test_the_window_bounds_are_inclusive(self) -> None:
        """A cell exactly at either edge still counts as showing the person."""
        assert_cells_describe_the_person(
            (100.0, 118.0),
            person_start=100.0,
            person_end=118.0,
            departure=118.0,
        )


class TestConcurrentBuildsCannotShareSegmentState:
    """Two activities must not see each other's segment structure.

    _segments decides which repairs are allowed, so an activity that reads
    another's structure would refuse a legitimate replacement and say nothing
    about why. This is reachable in production: _schedule_media gives every
    activity its own background task (media_tasks[activity_id] = task), so two
    builds really do overlap, and the lock sync_build used to take was keyed by
    activity id -- it serialized repeats of ONE activity and left different
    activities free to interleave.
    """

    async def test_the_manager_serialises_whole_builds(self, hass, tmp_path) -> None:
        """A second build must not start while the first is mid-flight.

        Driven with two activities whose event lookups block on a shared event, so
        the interleaving is forced rather than hoped for.
        """
        import asyncio

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

        started: list[str] = []
        inside = asyncio.Event()
        release = asyncio.Event()

        class Client:
            async def async_get_event(self, event_id, camera):
                started.append(event_id)
                if len(started) == 1:
                    inside.set()
                    await release.wait()
                return {
                    "id": event_id, "camera": camera, "label": "person",
                    "start_time": 1000.0, "end_time": 1010.0,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                return _jpeg(80)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _jpeg(120)

        manager = MediaManager(
            hass, store, Client(), tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
            preview_candidates=False,
        )

        async def build(name: str):
            record = ActivityRecord(
                activity_id=name,
                entry_id="entry_1",
                source=ActivitySource.STANDALONE_REVIEW,
                stage=ActivityStage.SEALED,
                created_at=990.0,
                updated_at=1020.0,
                camera="front",
                review_ids=(f"review_{name}",),
                detection_ids=(f"event_{name}",),
                finalization_deadline=1020.0,
            )
            await store.async_create(record)
            return await manager.async_build(name)

        first = asyncio.create_task(build("activity_a"))
        await inside.wait()
        second = asyncio.create_task(build("activity_b"))
        # Give the second task every chance to interleave.
        for _ in range(20):
            await asyncio.sleep(0)
        # Only ONE event lookup may have happened: the second build must be
        # parked on the manager's lock, not inside _async_build_locked.
        assert started == ["event_activity_a"], (
            f"第二个构建在第一个完成前就进入了构建体：{started}"
        )
        release.set()
        await asyncio.gather(first, second, return_exceptions=True)


class TestTheDepartureBoundIsNotShortCircuited:
    """The departure test must compare against departure, not the window end.

    plan_evidence sets departure = person_end + 2.8, so an earlier clause
    alue >= person_end was always true whenever the departure bound was -- the
    check could never fire. The fixtures below separate the two bounds so the
    difference is observable: same selection, three different departures, and the
    verdict must change.
    """

    def test_a_selection_before_departure_is_refused(self) -> None:
        with pytest.raises(MediaError, match="no_departure_frame"):
            assert_cells_describe_the_person(
                (105.0, 110.0, 115.0, 118.0),
                person_start=100.0,
                person_end=118.0,
                departure=120.8,
            )

    def test_a_selection_at_departure_is_accepted(self) -> None:
        assert_cells_describe_the_person(
            (105.0, 110.0, 115.0, 120.8),
            person_start=100.0,
            person_end=118.0,
            departure=120.8,
        )

    def test_the_departure_bound_actually_changes_the_verdict(self) -> None:
        """THE regression for the dead clause.

        With person_end=118 and a latest cell of 118, the old
        >= person_end clause accepted unconditionally. A departure of 120.8 must
        reject it.
        """
        cells = (105.0, 110.0, 115.0, 118.0)
        with pytest.raises(MediaError, match="no_departure_frame"):
            assert_cells_describe_the_person(
                cells, person_start=100.0, person_end=118.0, departure=120.8
            )
        # The same cells are fine once the departure is the last cell itself.
        assert_cells_describe_the_person(
            cells, person_start=100.0, person_end=118.0, departure=118.0
        )
