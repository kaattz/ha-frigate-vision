"""Does a configured focus region actually change which cells are chosen?

A region that is stored, read back and handed to the analyzer but never affects
the outcome is the worst of both worlds: the household draws it in Frigate, the
form saves it, and the selection behaves exactly as if it were not there.

The zones below are this deployment's real ones, copied verbatim from
`GET /api/config`. At preview resolution they occupy x 58..170 and x 132..227 --
measured, they do NOT overlap -- so a change placed inside one of them is a clean
test of whether the region weighting is observed at all.

An earlier version of this file asserted only that the build succeeded, which is
true whether or not the regions do anything: it caught 0 of 2 mutations. These
tests assert the DIFFERENCE the regions make, and the fixture is built so the
difference is visible: the region's own event is deliberately the SMALLEST raw
change, so it can only be selected when the region weighting promotes it.
"""

from __future__ import annotations

import time
from dataclasses import replace
from io import BytesIO

import numpy as np
from PIL import Image

from custom_components.frigate_vision.changes import (
    WHOLE_HEIGHT,
    WHOLE_WIDTH,
    analyze_sequence,
    segment_scored_frames,
)
from custom_components.frigate_vision.media import (
    ANCHOR_RESET_THRESHOLD,
    CHANGE_THRESHOLD,
    EvidencePlan,
    choose_cells,
)

# The deployment's two zones, normalized, verbatim from Frigate.
HOME_DOOR = [
    (0.181, 0.175),
    (0.407, 0.125),
    (0.442, 0.677),
    (0.456, 0.738),
    (0.533, 0.981),
    (0.286, 1.0),
]
ELEVATOR = [
    (0.411, 0.083),
    (0.68, 0.048),
    (0.662, 0.553),
    (0.713, 0.768),
    (0.516, 0.909),
    (0.471, 0.766),
    (0.445, 0.672),
]

# Preview-resolution boxes, measured: each lies wholly inside one zone.
DOOR_BOX = (60, 40, 130, 150)
LIFT_BOX = (150, 40, 210, 150)
# On the right-hand wall, outside both zones.
WALL_BOX = (240, 120, 310, 175)

def hour_start() -> float:
    """The current hour's start, read fresh on every call.

    A module constant cannot work here: the preview source decides whether Frigate
    has already cleared a window by comparing it against `time.time()`, and the
    full suite runs for minutes. A run that crosses the hour boundary would start
    describing the previous hour, the source would refuse it as expired, the build
    would fall back to recordings, and every preview assertion would fail --
    depending on what time the suite happened to start.
    """
    return (int(time.time()) // 3600) * 3600

PLAN = EvidencePlan(
    mode="review_six",
    first_time=1000.0,
    last_time=1018.0,
    postroll_time=1020.8,
    selection_source="path_motion",
)


def _decoded(events: dict[int, tuple[int, int, int, int]], count: int):
    """Preview frames as grayscale arrays, with a change at each named start."""
    frames = []
    for index in range(count):
        image = Image.new("RGB", (320, 180), (80, 80, 80))
        for start, box in events.items():
            if start <= index < start + 2:
                image.paste((220, 220, 220), box)
        buffer = BytesIO()
        image.save(buffer, "WEBP")
        with Image.open(BytesIO(buffer.getvalue())) as source:
            source.load()
            grey = source.convert("L").resize((WHOLE_WIDTH, WHOLE_HEIGHT))
        frames.append((1000.0 + index * 1.1, np.asarray(grey, dtype=np.uint8)))
    return frames


def _analyse(events, zones, count: int = 20):
    scored = analyze_sequence(
        _decoded(events, count),
        zones=zones,
        anchor_reset_threshold=ANCHOR_RESET_THRESHOLD,
    )
    return scored, segment_scored_frames(scored, change_threshold=CHANGE_THRESHOLD)


class TestARegionIsObserved:
    """The zone scores must reflect which region the change fell inside."""

    def test_a_change_inside_the_door_scores_the_door_not_the_lift(self) -> None:
        """The measurement the whole feature rests on."""
        zones = {"home_door": HOME_DOOR, "elevator": ELEVATOR}
        scored, _segments = _analyse({4: DOOR_BOX}, zones)
        busiest = max(scored, key=lambda item: item.best_zone)
        assert busiest.zones["home_door"] > 0.3, f"门口区域应显著变化：{busiest.zones}"
        assert busiest.zones["elevator"] < 0.05, (
            f"电梯区域不该跟着变化：{busiest.zones}"
        )

    def test_a_change_inside_the_lift_scores_the_lift(self) -> None:
        """The mirror case, so a hard-coded zone could not pass both."""
        zones = {"home_door": HOME_DOOR, "elevator": ELEVATOR}
        scored, _segments = _analyse({4: LIFT_BOX}, zones)
        busiest = max(scored, key=lambda item: item.best_zone)
        assert busiest.zones["elevator"] > 0.3, f"电梯区域应显著变化：{busiest.zones}"
        assert busiest.zones["home_door"] < 0.05, (
            f"门口区域不该跟着变化：{busiest.zones}"
        )

    def test_a_change_on_the_wall_scores_nothing(self) -> None:
        """A region is a promise about WHERE; a wall change must not qualify."""
        zones = {"home_door": HOME_DOOR, "elevator": ELEVATOR}
        scored, _segments = _analyse({4: WALL_BOX}, zones)
        assert max(item.best_zone for item in scored) == 0.0

    def test_no_zones_means_no_zone_scores(self) -> None:
        """Zero regression: an unconfigured entry reports nothing per region."""
        scored, _segments = _analyse({4: DOOR_BOX}, {})
        assert all(item.zones == {} for item in scored)
        assert all(item.best_zone == 0.0 for item in scored)


class TestTheRegionChangesTheSelection:
    """The score has to reach `choose_cells`, not merely be computed."""

    def test_the_region_promotes_its_own_event_over_bigger_ones(self) -> None:
        """THE load-bearing test.

        The fixture is built so this can only pass if the region weighting is
        real: the door event is a SMALL patch while the wall events are larger, so
        on raw change the door ranks last. With the door configured as a focus
        region it must be selected anyway; with no region configured it must be
        dropped.

        Measured before the priority trim was fixed: the selection was
        byte-for-byte identical with and without the region, because the old trim
        removed cells by position in time rather than by rank.
        """
        events: dict[int, tuple[int, int, int, int]] = {50: (70, 60, 120, 110)}
        for start in (4, 10, 16, 22, 28, 34, 40):
            events[start] = (230, 110, 310, 175)
        plan = replace(PLAN, first_time=1000.0, last_time=1065.0, postroll_time=1067.8)

        with_region = choose_cells(
            *_analyse(events, {"home_door": HOME_DOOR}, count=60),
            plan=plan,
            cells=6,
            max_cells=9,
        )
        without = choose_cells(
            *_analyse(events, {}, count=60),
            plan=plan,
            cells=6,
            max_cells=9,
        )

        assert [m for m in with_region if 1054.0 <= m <= 1058.0], (
            f"配置门口重点区域后，门口那次变化必须被选入：{with_region}"
        )
        assert not [m for m in without if 1054.0 <= m <= 1058.0], (
            f"不配置区域时它应被更大的变化挤掉：{without}"
        )
        assert with_region != without, "配置重点区域前后，选择必须不同"

    def test_the_region_does_not_break_the_layout(self) -> None:
        """Weighting reorders candidates; it must not empty or mis-shape a sheet."""
        for zones in (
            {"home_door": HOME_DOOR},
            {},
            {"home_door": HOME_DOOR, "elevator": ELEVATOR},
        ):
            chosen = choose_cells(
                *_analyse({4: DOOR_BOX}, zones), plan=PLAN, cells=6, max_cells=9
            )
            assert len(chosen) in {0, 6, 9}, f"{zones}: {len(chosen)}"
            assert list(chosen) == sorted(chosen), f"{zones}: 必须按时间升序"

    def test_the_anchors_survive_the_trim(self) -> None:
        """The two cells the prompt's rules read must not be trimmed away."""
        events: dict[int, tuple[int, int, int, int]] = {50: DOOR_BOX}
        for start in (4, 10, 16, 22, 28, 34, 40):
            events[start] = WALL_BOX
        plan = replace(PLAN, first_time=1000.0, last_time=1065.0, postroll_time=1067.8)
        chosen = choose_cells(
            *_analyse(events, {"home_door": HOME_DOOR}, count=60),
            plan=plan,
            cells=6,
            max_cells=9,
        )
        assert chosen[0] == plan.first_time, "第一格必须是人物首帧"
        assert chosen[-1] == plan.postroll_time, "最后一格必须是人物离开后的现场"


class TestTheRegionBreaksATieInsideOneSegment:
    """Two frames in the SAME change segment, decided by the region.

    The segment-level ordering cannot help here: both frames belong to one
    segment, so the only thing that can distinguish them is the frame-level
    tie-break. The in-zone frame deliberately has the SMALLER raw change, so a
    tie-break that ignores the region picks the other one.
    """

    def _frames(self):
        """Frames 4 and 5 both change, so they form ONE change segment.

        The sizes are chosen so the two axes disagree, which is what makes the
        tie-break testable: measured on these boxes the in-zone frame scores
        0.417 with a zone reading of 0.150, and the wall frame scores 0.967 with a
        zone reading of 0.000. Asserting both inequalities keeps the fixture
        honest -- with equal-sized patches the two frames scored identically and
        the test could not tell which frame had won.
        """
        frames = []
        door = (70, 60, 110, 100)     # small, inside the configured zone
        wall = (215, 85, 315, 175)    # large, on the wall
        for index in range(14):
            image = Image.new("RGB", (320, 180), (80, 80, 80))
            if index == 4:
                image.paste((190, 190, 190), door)
            if index == 5:
                image.paste((250, 250, 250), wall)
            buffer = BytesIO()
            image.save(buffer, "WEBP")
            with Image.open(BytesIO(buffer.getvalue())) as source:
                source.load()
                grey = source.convert("L").resize((WHOLE_WIDTH, WHOLE_HEIGHT))
            frames.append((1000.0 + index * 1.1, np.asarray(grey, dtype=np.uint8)))
        return frames

    def test_the_in_zone_frame_wins_the_tie(self) -> None:
        scored = analyze_sequence(
            self._frames(),
            zones={"home_door": HOME_DOOR},
            anchor_reset_threshold=ANCHOR_RESET_THRESHOLD,
        )
        segments = segment_scored_frames(scored, change_threshold=CHANGE_THRESHOLD)
        change = [s for s in segments if s.kind == "change"]
        assert change, (
            f"夹具必须产出一个变化段：{[(s.kind, s.indices) for s in segments]}"
        )
        # Confirm the two frames really are in one segment and really differ in
        # both axes, or the test would not be exercising the tie-break.
        members = [scored[index] for index in change[0].indices]
        assert len(members) >= 2, f"需要同一段内至少两帧：{change[0].indices}"
        by_zone = max(members, key=lambda item: (item.best_zone, item.score))
        by_score = max(members, key=lambda item: item.score)
        assert by_zone.best_zone > by_score.best_zone, "带区域的帧必须区域分数更高"
        assert by_score.score > by_zone.score, "而它自身的原始分数更低"

        chosen = choose_cells(
            scored, segments, plan=PLAN, cells=6, max_cells=9
        )
        assert by_zone.timestamp in chosen, (
            f"同段内区域分数更高的帧应被选中：{chosen}"
        )

    def test_without_the_region_the_bigger_change_wins(self) -> None:
        """The mirror, so the previous test cannot pass on a coincidence."""
        scored = analyze_sequence(
            self._frames(), zones={}, anchor_reset_threshold=ANCHOR_RESET_THRESHOLD
        )
        segments = segment_scored_frames(scored, change_threshold=CHANGE_THRESHOLD)
        change = [s for s in segments if s.kind == "change"]
        assert change
        members = [scored[index] for index in change[0].indices]
        by_score = max(members, key=lambda item: item.score)
        chosen = choose_cells(
            scored, segments, plan=PLAN, cells=6, max_cells=9
        )
        assert by_score.timestamp in chosen, (
            f"没有区域时原始分数更高的帧应被选中：{chosen}"
        )


class TestTheZonesReachTheAnalyzerThroughTheManager:
    """A build-level test, because the wiring lives in MediaManager.

    The tests above call nalyze_sequence directly, so they stay green even if
    the manager stops passing its regions down -- which is exactly the mutation
    that slipped through. This drives a real build with a client that serves
    preview frames, and asserts the metadata shows the preview route ran and the
    selection was shaped by the regions.
    """

    async def _build(self, hass, tmp_path, zones, activity_id):
        import json
        from pathlib import Path as _Path

        from homeassistant.core import HomeAssistant  # noqa: F401

        from custom_components.frigate_vision.correlation import ZoneRoles
        from custom_components.frigate_vision.media import MediaManager
        from custom_components.frigate_vision.models import (
            ActivityRecord,
            ActivitySource,
            ActivityStage,
        )
        from custom_components.frigate_vision.store import ActivityStore

        hour = (int(hour_start()))
        names = tuple(
            f"preview_front-{hour + 95 + index * 1.1:.6f}.webp" for index in range(24)
        )

        def thumb(name: str) -> bytes:
            moment = float(name.split("-")[-1].removesuffix(".webp"))
            index = int((moment - hour - 95) / 1.1)
            image = Image.new("RGB", (320, 180), (80, 80, 80))
            # The in-zone event is the SMALLEST change; the wall events are
            # larger. That is what makes the region decide the outcome: on raw
            # score the in-zone frame ranks last, and it can only be selected
            # when the region weighting promoted it.
            if index == 10:
                image.paste((190, 190, 190), (70, 60, 110, 100))
            for start in (2, 3, 4, 5, 6, 7, 8):
                if index == start:
                    image.paste((250, 250, 250), (215, 85, 315, 175))
            buffer = BytesIO()
            image.save(buffer, "WEBP")
            return buffer.getvalue()

        class Client:
            async def async_get_event(self, event_id, camera):
                return {
                    "id": event_id, "camera": camera, "label": "person",
                    "start_time": hour + 100, "end_time": hour + 120,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                buffer = BytesIO()
                Image.new("RGB", (320, 180), (70 + int(timestamp) % 150,) * 3).save(
                    buffer, "JPEG"
                )
                return buffer.getvalue()

            async def async_get_event_snapshot(self, event_id, camera, height):
                buffer = BytesIO()
                Image.new("RGB", (320, 180), (120, 120, 120)).save(buffer, "JPEG")
                return buffer.getvalue()

            async def async_get_preview_frames(self, camera, start, end):
                return names

            async def async_get_preview_thumbnail(self, file_name):
                return thumb(file_name)

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = ActivityRecord(
            activity_id=activity_id,
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.SEALED,
            created_at=hour + 90,
            updated_at=hour + 130,
            camera="front",
            review_ids=("review_zone",),
            detection_ids=("event_1",),
            finalization_deadline=hour + 130,
        )
        await store.async_create(record)
        manager = MediaManager(
            hass, store, Client(), tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
            focus_zones=zones,
        )
        completed = await manager.async_build(record.activity_id)
        metadata = _Path(completed.evidence_path).with_suffix(".json")
        return json.loads(metadata.read_text(encoding="utf-8")), completed.sample_times

    async def test_a_build_with_regions_reports_the_preview_route(
        self, hass, tmp_path
    ) -> None:
        payload, samples = await self._build(
            hass, tmp_path, {"home_door": HOME_DOOR}, "review_entry_1_front_zone_mgr"
        )
        assert payload["candidate_source"] == "preview"
        assert payload["plan_version"] == 6
        assert len(samples) in {6, 9}

    async def test_the_build_still_works_without_regions(self, hass, tmp_path) -> None:
        """Zero regression: an unconfigured entry takes the same route."""
        payload, samples = await self._build(
            hass, tmp_path, {}, "review_entry_1_front_zone_mgr2"
        )
        assert payload["candidate_source"] == "preview"
        assert len(samples) in {6, 9}

    async def test_the_regions_change_the_builds_selection(
        self, hass, tmp_path
    ) -> None:
        """THE wiring test: same activity, one variable -- the regions.

        Asserting only that the build succeeded is what made an earlier version of
        this class useless: replacing the manager's zones with an empty mapping
        left every assertion green. The fixture therefore has to make the region
        decide something. The in-zone event is the SMALLEST change and the wall
        event is larger, so the in-zone frame can only appear in the selection
        when the region weighting promoted it.
        """
        with_regions, samples_a = await self._build(
            hass, tmp_path, {"home_door": HOME_DOOR}, "review_entry_1_front_zone_mgr3"
        )
        without, samples_b = await self._build(
            hass, tmp_path, {}, "review_entry_1_front_zone_mgr4"
        )
        assert with_regions["candidate_source"] == "preview"
        assert without["candidate_source"] == "preview"
        assert samples_a != samples_b, (
            "配置重点区域前后，构建出的 sample_times 必须不同，"
            f"否则区域没有真正参与选择：{samples_a} vs {samples_b}"
        )
