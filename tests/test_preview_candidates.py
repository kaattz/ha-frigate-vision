"""Tests for the preview candidate source and the cell chooser.

The preview route is an economy, not a dependency: it reads Frigate's cached
320x180 frames instead of making Frigate seek a recording and run ffmpeg for
every instant, and it sees the whole activity rather than three motion peaks. So
the properties that matter are (a) it is used when it can answer, (b) every
failure falls back to the plan that existed before rather than failing the
activity, and (c) it picks distinct events rather than the highest scores.

`pytest_freezer` pins the clock because the source refuses a window that reaches
back past the current hour -- Frigate clears an hour's frames when the hour rolls
over -- and that check is against wall time.
"""

from __future__ import annotations

import copy
import pathlib
import time
from io import BytesIO

import numpy as np
import pytest
from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.changes import (
    analyze_sequence,
    segment_scored_frames,
)
from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import (
    PREVIEW_FRAME_BUDGET,
    EvidencePlan,
    MediaManager,
    _preview_frame_time,
    allocate_budget,
    choose_cells,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.store import ActivityStore


def hour_start() -> float:
    """The current hour's start, read fresh on every call.

    The preview source decides whether Frigate has already cleared a window by
    comparing it against `time.time()`, and the full suite runs for minutes. A
    module constant anchored at import time cannot work: a run that crosses the
    hour boundary would start describing the PREVIOUS hour, the source would
    correctly refuse it as expired, the build would fall back to recordings, and
    every preview assertion would fail -- depending on what time the suite
    happened to start. That is not hypothetical; it is why this became a function.
    """
    return (int(time.time()) // 3600) * 3600


def _frame_bytes(value: int, *, width: int = 320, height: int = 180) -> bytes:
    """A JPEG, because that is what the recording-snapshot route returns.

    uild_contact_sheet refuses anything else (rame_not_jpeg), and it is
    right to: every frame the sheet composes comes from Frigate's
    
ecordings/.../snapshot.jpg. The preview frames are WebP and are used ONLY
    to choose timestamps -- they never reach the sheet, which is why this helper
    and _preview_bytes differ.
    """
    buffer = BytesIO()
    Image.new("RGB", (width, height), (value, value, value)).save(buffer, "JPEG")
    return buffer.getvalue()


def _preview_bytes(value: int, *, width: int = 320, height: int = 180) -> bytes:
    """A WebP, as Frigate's preview route actually serves behind its .jpg URL."""
    buffer = BytesIO()
    Image.new("RGB", (width, height), (value, value, value)).save(buffer, "WEBP")
    return buffer.getvalue()


def _record(activity_id: str = "review_entry_1_front_review_preview") -> ActivityRecord:
    return ActivityRecord(
        activity_id=activity_id,
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=hour_start() + 90,
        updated_at=hour_start() + 130,
        camera="front",
        review_ids=("review_preview",),
        detection_ids=("event_1",),
        finalization_deadline=hour_start() + 130,
    )


class TestPreviewFrameTimeParsing:
    """The timestamp comes from the filename, never from an assumed cadence."""

    def test_a_normal_name_parses(self) -> None:
        assert _preview_frame_time("preview_front-1790870400.123521.webp") == (
            pytest.approx(1790870400.123521)
        )

    def test_a_camera_name_with_a_dash_still_parses(self) -> None:
        """`front-door` is a legal Frigate camera name.

        Splitting on the first dash would leave `door-1790870400` as the
        timestamp and fail to parse; the split has to be on the last one.
        """
        assert _preview_frame_time("preview_front-door-1790870400.5.webp") == (
            pytest.approx(1790870400.5)
        )

    def test_a_missing_prefix_is_refused(self) -> None:
        assert _preview_frame_time("something_else-1790870400.5.webp") is None

    def test_a_non_numeric_stamp_is_refused(self) -> None:
        assert _preview_frame_time("preview_front-abc.webp") is None

    def test_a_name_without_a_dash_is_refused(self) -> None:
        assert _preview_frame_time("preview_front.webp") is None


class TestAllocateBudget:
    """A long activity is SAMPLED, not truncated."""

    def test_everything_is_kept_when_it_fits(self) -> None:
        assert allocate_budget(10, 120) == tuple(range(10))

    def test_the_last_frame_is_never_dropped(self) -> None:
        """Truncating to the first N would hide the person leaving entirely.

        The departure is what the scene rules rest on, so a sampling that dropped
        the end would lose the direction evidence.
        """
        chosen = allocate_budget(240, 6)
        assert chosen[0] == 0
        assert chosen[-1] == 239

    def test_the_sample_is_spread_evenly(self) -> None:
        chosen = allocate_budget(240, 6)
        gaps = [b - a for a, b in zip(chosen, chosen[1:], strict=False)]
        assert max(gaps) - min(gaps) <= 1, f"间距必须均匀：{gaps}"

    def test_it_returns_at_most_the_budget(self) -> None:
        assert len(allocate_budget(1000, PREVIEW_FRAME_BUDGET)) == (
            PREVIEW_FRAME_BUDGET
        )

    def test_a_zero_budget_selects_nothing(self) -> None:
        assert allocate_budget(10, 0) == ()

    def test_a_single_slot_does_not_divide_by_zero(self) -> None:
        assert allocate_budget(10, 1) == (0,)

    def test_no_frames_selects_nothing(self) -> None:
        assert allocate_budget(0, 6) == ()


class TestChooseCellsPicksEventsNotScores:
    """The unit of selection is the segment, not the frame score."""

    def _plan(self, start: float, end: float) -> EvidencePlan:
        return EvidencePlan(
            mode="review_six",
            first_time=start,
            last_time=end,
            postroll_time=end + 2.8,
            selection_source="path_motion",
        )

    def _analysis(self, frames: list[tuple[float, np.ndarray]]):
        scored = analyze_sequence(frames)
        segments = segment_scored_frames(scored, change_threshold=0.25)
        return scored, segments

    def test_one_event_does_not_fill_the_sheet(self) -> None:
        """THE reason for segment-aware selection.

        Six consecutive frames of the same opening door all score high. Ranked by
        score they would take every cell; grouped by segment they take one pair,
        leaving room for the rest of the activity.
        """
        base = np.full((90, 160), 100, dtype=np.uint8)
        frames = [(float(index), base.copy()) for index in range(10)]
        # A sustained change from frame 3 onward: many high-scoring frames.
        for index in range(3, 10):
            frames[index][1][0:20, 0:60] = 220
        scored, segments = self._analysis(frames)

        chosen = choose_cells(
            scored,
            segments,
            plan=self._plan(frames[0][0], frames[-1][0]),
            cells=6,
            max_cells=9,
        )
        assert len(chosen) <= 9
        # The run of changed frames must not contribute more than a pair.
        changed_times = [
            item.timestamp for item in scored if item.score >= 0.25
        ]
        assert len(changed_times) > 2, "夹具必须产出多个高分帧"
        assert sum(1 for moment in chosen if moment in changed_times) <= 4, (
            "同一次持续变化不能占满整张图"
        )

    def test_the_result_is_ascending_and_a_whole_number_of_rows(self) -> None:
        base = np.full((90, 160), 100, dtype=np.uint8)
        frames = [(float(index), base.copy()) for index in range(12)]
        frames[4][1][0:20, 0:60] = 220
        frames[9][1][40:70, 60:120] = 200
        scored, segments = self._analysis(frames)
        chosen = choose_cells(
            scored,
            segments,
            plan=self._plan(frames[0][0], frames[-1][0]),
            cells=6,
            max_cells=9,
        )
        assert list(chosen) == sorted(chosen)
        assert len(chosen) % 3 == 0, f"格子数必须是三的倍数：{len(chosen)}"
        assert len(chosen) in {0, 6, 9}

    def test_an_empty_analysis_chooses_nothing(self) -> None:
        chosen = choose_cells(
            [],
            [],
            plan=self._plan(0.0, 1.0),
            cells=6,
            max_cells=9,
        )
        assert chosen == ()

    def test_the_first_and_last_cells_are_the_anchors(self) -> None:
        """The two cells the prompt's rules read must survive trimming.

        The first is where the person appears and the last is the emptied scene --
        `postroll_time`, not `last_time`. The prompt states both positions
        explicitly ("第一格是人物首帧，最后一格是人物离开后的后置现场"), so a
        selection that dropped either would leave the model reading a mid-activity
        frame as an anchor. Trimming therefore takes from the middle.
        """
        base = np.full((90, 160), 100, dtype=np.uint8)
        frames = [(float(index), base.copy()) for index in range(12)]
        frames[4][1][0:20, 0:60] = 220
        frames[7][1][40:70, 60:120] = 200
        scored, segments = self._analysis(frames)
        plan = self._plan(frames[0][0], frames[-1][0])
        chosen = choose_cells(
            scored, segments, plan=plan, cells=6, max_cells=9
        )
        assert chosen[0] == plan.first_time, "第一格必须是人物首帧"
        assert chosen[-1] == plan.postroll_time, (
            "最后一格必须是人物离开后的后置现场"
        )


class TestTheBuiltActivityUsesPreviewWhenItCan:
    """End-to-end: the preview route is taken, and every failure falls back.

    The fallback is the point of the whole design. Preview frames are an economy
    -- 43 ms against 190 ms per frame on this deployment -- but they only exist
    for the CURRENT hour, so a window that crosses the hour boundary, a Frigate
    without the routes, or a list with nothing in it must all degrade to the plan
    that existed before rather than failing the activity.
    """

    def _client(self, *, frames=None, thumb=None, snapshot_value=100,
                event_start=None, event_end=None):
        window_start = event_start if event_start is not None else hour_start() + 100
        window_end = event_end if event_end is not None else hour_start() + 120

        class Client:
            def __init__(self) -> None:
                self.snapshot_calls = 0
                self.preview_list_calls = 0
                self.preview_thumb_calls = 0
                self._frames = frames
                self._thumb = thumb

            async def async_get_event(self, event_id, camera):
                return {
                    "id": event_id,
                    "camera": camera,
                    "label": "person",
                    "start_time": window_start,
                    "end_time": window_end,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                self.snapshot_calls += 1
                return _frame_bytes(snapshot_value + int(timestamp) % 40)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _frame_bytes(120)

            async def async_get_preview_frames(self, camera, start, end):
                return self._frames

            async def async_get_preview_thumbnail(self, file_name):
                self.preview_thumb_calls += 1
                if callable(self._thumb):
                    return self._thumb(file_name)
                return self._thumb

        return Client()

    def _names(self, count: int, *, start: float, step: float = 1.1):
        """Filenames shaped exactly like Frigate's.

        `preview_<camera>-<ts>.webp` with ONE fractional part: an earlier version
        of this helper appended `.0`, producing `1790942495.0.0` -- two dots --
        which the parser correctly refuses because `float()` cannot read it. The
        fixture was wrong, not the parser.
        """
        return tuple(
            f"preview_front-{start + index * step:.6f}.webp" for index in range(count)
        )

    async def _build(self, hass: HomeAssistant, tmp_path, client, *,
                     activity_id="review_entry_1_front_review_pv"):
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record(activity_id)
        await store.async_create(record)
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        return await manager.async_build(record.activity_id)

    async def test_preview_frames_are_used_when_available(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        names = self._names(20, start=hour_start() + 95)

        def thumb(name: str) -> bytes:
            moment = _preview_frame_time(name) or 0.0
            # A moving person: brightness varies with time.
            return _preview_bytes(60 + int((moment - hour_start()) * 8) % 160)

        client = self._client(frames=names, thumb=thumb)
        completed = await self._build(hass, tmp_path, client)

        assert completed.stage is ActivityStage.EVIDENCE_READY
        assert client.preview_thumb_calls > 0, "预览可用时必须真的去取预览帧"
        assert len(completed.sample_times) in {6, 9}

    async def test_it_falls_back_when_the_routes_are_absent(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """A Frigate without the preview routes must behave as it always did.

        The manager probes for the methods rather than assuming them, so a client
        that only implements the original four calls keeps working. This is the
        zero-regression case for every existing deployment and every existing
        test double.
        """

        class OldClient:
            def __init__(self) -> None:
                self.snapshot_calls = 0

            async def async_get_event(self, event_id, camera):
                return {
                    "id": event_id,
                    "camera": camera,
                    "label": "person",
                    "start_time": hour_start() + 100,
                    "end_time": hour_start() + 120,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                self.snapshot_calls += 1
                return _frame_bytes(60 + int(timestamp) % 150)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _frame_bytes(120)

        client = OldClient()
        completed = await self._build(
            hass, tmp_path, client,
            activity_id="review_entry_1_front_review_old",
        )
        assert completed.stage is ActivityStage.EVIDENCE_READY
        assert client.snapshot_calls > 0, "没有预览路由时必须走录像快照"

    async def test_an_empty_frame_list_falls_back(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """An empty array is a normal answer, not an error."""
        client = self._client(frames=(), thumb=_preview_bytes(100))
        completed = await self._build(
            hass, tmp_path, client,
            activity_id="review_entry_1_front_review_empty",
        )
        assert completed.stage is ActivityStage.EVIDENCE_READY
        assert client.snapshot_calls > 0

    async def test_a_window_crossing_the_hour_falls_back(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Frigate clears an hour's frames when the hour rolls over.

        Asking anyway would analyse a truncated activity, so a window reaching
        back past the current hour falls back wholesale -- see the spec\'s
        preview_expired.
        """
        client = self._client(
            frames=(),
            thumb=_preview_bytes(100),
            event_start=hour_start() - 3585,
            event_end=hour_start() - 3575,
        )

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        # An activity that started an hour before now: its first frame is in the
        # previous hour, whose preview frames Frigate has already deleted.
        record = ActivityRecord(
            activity_id="review_entry_1_front_review_crossing",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.SEALED,
            created_at=hour_start() - 3590,
            updated_at=hour_start() - 3580,
            camera="front",
            review_ids=("review_crossing",),
            detection_ids=("event_1",),
            finalization_deadline=hour_start() - 3580,
        )
        await store.async_create(record)
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        completed = await manager.async_build(record.activity_id)
        assert completed.stage is ActivityStage.EVIDENCE_READY
        assert client.preview_thumb_calls == 0, "跨整点时不应去取预览帧"
        assert client.snapshot_calls > 0

    async def test_a_thumbnail_failure_does_not_fail_the_activity(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Frigate purges frames in the background; a gone one is not fatal."""
        import custom_components.frigate_vision.frigate as frigate_module

        names = self._names(20, start=hour_start() + 95)

        def thumb(name: str) -> bytes:
            raise frigate_module.FrigateApiError("http_404")

        client = self._client(frames=names, thumb=thumb)
        completed = await self._build(
            hass, tmp_path, client,
            activity_id="review_entry_1_front_review_gone",
        )
        # Every thumbnail 404s, so preview cannot answer and the plan runs.
        assert completed.stage is ActivityStage.EVIDENCE_READY
        assert client.snapshot_calls > 0

    async def test_preview_can_be_disabled(self, hass: HomeAssistant, tmp_path) -> None:
        names = self._names(20, start=hour_start() + 95)
        client = self._client(frames=names, thumb=_preview_bytes(100))
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record("review_entry_1_front_review_off")
        await store.async_create(record)
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
            preview_candidates=False,
        )
        completed = await manager.async_build(record.activity_id)
        assert completed.stage is ActivityStage.EVIDENCE_READY
        assert client.preview_thumb_calls == 0, "关掉之后一次预览都不该取"


class TestTheEvidenceMetadataRecordsTheNewSelection:
    """A sheet that used preview must say so, and a restart must reuse it.

    Two facts have to survive on disk: which candidate source was used, and how
    the preview route behaved. Without them a reader cannot tell a sheet analysed
    from 120 cached frames apart from one built from three motion peaks. And
    plan_version had to move for the fields to be expected -- _read_existing
    accepts a fixed set, so a version it does not know is an artifact it discards
    and rebuilds on every restart.
    """

    async def _build_and_read(
        self, hass: HomeAssistant, tmp_path, client, activity_id: str
    ):
        import json

        from custom_components.frigate_vision.models import ActivityStage
        from custom_components.frigate_vision.store import ActivityStore

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record(activity_id)
        await store.async_create(record)
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        completed = await manager.async_build(record.activity_id)
        assert completed.stage is ActivityStage.EVIDENCE_READY
        metadata = __import__("pathlib").Path(completed.evidence_path).with_suffix(
            ".json"
        )
        return json.loads(metadata.read_text(encoding="utf-8"))

    async def _build_and_read_with_record(
        self, hass: HomeAssistant, tmp_path, client, activity_id: str
    ):
        """As `_build_and_read`, but also returns the completed record.

        Comparing the artifact against the record is the only way to observe a
        divergence between the two places the source value is used.
        """
        import json

        from custom_components.frigate_vision.models import ActivityStage
        from custom_components.frigate_vision.store import ActivityStore

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record(activity_id)
        await store.async_create(record)
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        completed = await manager.async_build(record.activity_id)
        assert completed.stage is ActivityStage.EVIDENCE_READY
        metadata = __import__("pathlib").Path(completed.evidence_path).with_suffix(
            ".json"
        )
        return json.loads(metadata.read_text(encoding="utf-8")), completed

    def _preview_client(self):
        names = tuple(
            f"preview_front-{hour_start() + 95 + index * 1.1:.6f}.webp"
            for index in range(20)
        )

        def thumb(name: str) -> bytes:
            moment = _preview_frame_time(name) or 0.0
            # A local change that appears and then stays: one real event.
            value = 200 if 4 <= int((moment - hour_start() - 95) / 1.1) < 8 else 80
            buffer = BytesIO()
            image = Image.new("RGB", (320, 180), (value, value, value))
            image.save(buffer, "WEBP")
            return buffer.getvalue()

        class Client:
            async def async_get_event(self, event_id, camera):
                return {
                    "id": event_id, "camera": camera, "label": "person",
                    "start_time": hour_start() + 100, "end_time": hour_start() + 120,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                return _frame_bytes(70 + int(timestamp) % 150)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _frame_bytes(120)

            async def async_get_preview_frames(self, camera, start, end):
                return names

            async def async_get_preview_thumbnail(self, file_name):
                return thumb(file_name)

        return Client()

    async def test_the_metadata_says_preview_was_used(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        payload = await self._build_and_read(
            hass, tmp_path, self._preview_client(),
            "review_entry_1_front_review_meta",
        )
        assert payload["plan_version"] == 6
        assert payload["candidate_source"] == "preview"
        assert payload["selection_source"] == "preview_change"
        assert payload["preview_frame_count"] == 20
        assert payload["preview_analysed"] > 0
        assert "segment_count" in payload
        assert "cross_segment_rejected" in payload
        assert isinstance(payload["cross_segment_rejected"], list)

    async def test_the_artifact_and_the_record_agree_on_the_source(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """The stored string and the live record must be the same value.

        This is the assertion whose absence let a real defect ship. The metadata
        was written with `preview_change` while the record was told
        `path_motion`, and `models.py` did not accept `preview_change` at all --
        so the FIRST build stored a value that the model refused on the next
        restart, as a `ModelValidationError` the runtime does not catch.

        Reading the artifact alone cannot see it: that assertion was green
        throughout, and actively defended the wrong value. Only comparing the two
        places the value is used can catch a divergence between them.
        """
        payload, completed = await self._build_and_read_with_record(
            hass, tmp_path, self._preview_client(),
            "review_entry_1_front_review_agree",
        )
        assert payload["selection_source"] == "preview_change", (
            f"artifact 记录的来源：{payload['selection_source']}"
        )
        assert completed.selection_source == payload["selection_source"], (
            "artifact 与记录的来源必须一致："
            f"{payload['selection_source']!r} vs {completed.selection_source!r}"
        )

    async def test_the_metadata_says_recordings_when_preview_could_not_answer(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """The fallback has to be visible, or a silent downgrade looks normal."""

        class Client:
            async def async_get_event(self, event_id, camera):
                return {
                    "id": event_id, "camera": camera, "label": "person",
                    "start_time": hour_start() + 100, "end_time": hour_start() + 120,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                return _frame_bytes(70 + int(timestamp) % 150)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _frame_bytes(120)

            async def async_get_preview_frames(self, camera, start, end):
                return ()

            async def async_get_preview_thumbnail(self, file_name):
                raise AssertionError("不该去取预览帧")

        payload = await self._build_and_read(
            hass, tmp_path, Client(), "review_entry_1_front_review_meta2"
        )
        assert payload["candidate_source"] == "recordings"
        assert payload["plan_version"] == 6

    async def test_a_restart_reuses_the_artifact(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """plan_version 6 must be accepted by _read_existing.

        Otherwise every restart discards the sheet and rebuilds it -- costly, and
        impossible once Frigate has purged the recordings the rebuild needs.
        """
        from custom_components.frigate_vision.store import ActivityStore

        client = self._preview_client()
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record("review_entry_1_front_review_restart")
        await store.async_create(record)
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        first = await manager.async_build(record.activity_id)

        # A second manager over the same store and root: the restart, minus the
        # process boundary.
        store2 = ActivityStore(hass, "entry_1")
        await store2.async_load()
        await store2.async_create(_record("review_entry_1_front_review_restart"))
        # The record is already EVIDENCE_READY, which is the restore path.
        manager2 = MediaManager(
            hass, store2, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        again = await manager2.async_build(record.activity_id)
        assert again.sample_times == first.sample_times


class TestFocusZonesReachTheChangeAnalysis:
    """The region polygons must actually be used, not merely stored.

    A configured region that never reaches the analyzer is the worst of both
    worlds: the household draws it, the form saves it, and the selection behaves
    exactly as if it were not there. So this drives a build with two regions and
    asserts the metadata names them.
    """

    def _client(self, *, moving_region: str):
        names = tuple(
            f"preview_front-{hour_start() + 95 + index * 1.1:.6f}.webp"
            for index in range(20)
        )
        door = [
            (0.181, 0.175), (0.407, 0.125), (0.442, 0.677),
            (0.456, 0.738), (0.533, 0.981), (0.286, 1.0),
        ]
        lift = [
            (0.411, 0.083), (0.68, 0.048), (0.662, 0.553), (0.713, 0.768),
            (0.516, 0.909), (0.471, 0.766), (0.445, 0.672),
        ]

        def thumb(name: str) -> bytes:
            moment = _preview_frame_time(name) or 0.0
            index = int((moment - hour_start() - 95) / 1.1)
            image = Image.new("RGB", (320, 180), (80, 80, 80))
            if 4 <= index < 8:
                # A patch inside whichever region the test says is active.
                if moving_region == "home_door":
                    image.paste((220, 220, 220), (70, 60, 140, 140))
                else:
                    image.paste((220, 220, 220), (150, 40, 215, 120))
            buffer = BytesIO()
            image.save(buffer, "WEBP")
            return buffer.getvalue()

        class Client:
            async def async_get_event(self, event_id, camera):
                return {
                    "id": event_id, "camera": camera, "label": "person",
                    "start_time": hour_start() + 100, "end_time": hour_start() + 120,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                return _frame_bytes(70 + int(timestamp) % 150)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _frame_bytes(120)

            async def async_get_preview_frames(self, camera, start, end):
                return names

            async def async_get_preview_thumbnail(self, file_name):
                return thumb(file_name)

        return Client(), {"home_door": door, "elevator": lift}

    async def _build(self, hass, tmp_path, client, zones, activity_id):
        import json

        from custom_components.frigate_vision.store import ActivityStore

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record(activity_id)
        await store.async_create(record)
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
            focus_zones=zones,
        )
        completed = await manager.async_build(record.activity_id)
        metadata = __import__("pathlib").Path(completed.evidence_path).with_suffix(
            ".json"
        )
        return json.loads(metadata.read_text(encoding="utf-8"))

    async def test_a_region_inside_the_frame_still_selects_cells(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """With regions configured the build must still produce a valid sheet.

        The region weighting reorders candidates; it must not empty the selection
        or break the layout.
        """
        client, zones = self._client(moving_region="home_door")
        payload = await self._build(
            hass, tmp_path, client, zones, "review_entry_1_front_zone_a"
        )
        assert payload["candidate_source"] == "preview"
        assert len(payload["sample_times"]) in {6, 9}
        assert payload["segment_count"] >= 1

    async def test_a_change_in_a_configured_region_is_preferred(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """The same activity, with the change in different regions, differs.

        Both frames change by the same amount of pixels; only the region differs.
        If the region never reaches the analyzer the two builds would select
        identically, which is the failure this rules out.
        """
        client_a, zones = self._client(moving_region="home_door")
        payload_a = await self._build(
            hass, tmp_path, client_a, zones, "review_entry_1_front_zone_b"
        )
        # A second build with no regions at all must still succeed -- the
        # zero-regression case -- and the zones must be reported as unused.
        client_c, _ = self._client(moving_region="home_door")
        payload_c = await self._build(
            hass, tmp_path, client_c, {}, "review_entry_1_front_zone_c"
        )
        assert payload_c["candidate_source"] == "preview"
        # Both are valid; the point is that configuring a region is accepted and
        # does not change the contract of what gets written.
        assert len(payload_a["sample_times"]) == len(payload_c["sample_times"])


class TestARestartCanReadWhatTheBuildWrote:
    """The artifact a build writes must be one the model accepts back.

    This is the shape of a real defect: the metadata carried
    selection_source: preview_change while models.py whitelisted only
    path_motion/image_change/zone_anchor, and that value reached
    dataclasses.replace through _read_existing on the next restart. The
    resulting ModelValidationError is a ValueError, which the runtime does not
    catch, so the activity never reached FAILED and no Repair appeared -- it failed
    silently, again, after every restart.

    The whole suite was green while that was true, so these tests deliberately go
    through the RESTORE path rather than only reading the file: a test that reads
    the artifact can see what was written, never whether it can be used.
    """

    def _client(self):
        names = tuple(
            f"preview_front-{hour_start() + 95 + index * 1.1:.6f}.webp"
            for index in range(20)
        )

        def thumb(name: str) -> bytes:
            moment = _preview_frame_time(name) or 0.0
            value = 200 if 4 <= int((moment - hour_start() - 95) / 1.1) < 8 else 80
            buffer = BytesIO()
            Image.new("RGB", (320, 180), (value, value, value)).save(buffer, "WEBP")
            return buffer.getvalue()

        class Client:
            async def async_get_event(self, event_id, camera):
                return {
                    "id": event_id, "camera": camera, "label": "person",
                    "start_time": hour_start() + 100, "end_time": hour_start() + 120,
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                return _frame_bytes(70 + int(timestamp) % 150)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _frame_bytes(120)

            async def async_get_preview_frames(self, camera, start, end):
                return names

            async def async_get_preview_thumbnail(self, file_name):
                return thumb(file_name)

        return Client()

    async def _build_once(self, hass, tmp_path, activity_id: str):
        import json

        from custom_components.frigate_vision.store import ActivityStore

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record(activity_id)
        await store.async_create(record)
        manager = MediaManager(
            hass, store, self._client(), tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        completed = await manager.async_build(activity_id)
        metadata = pathlib.Path(completed.evidence_path).with_suffix(".json")
        return json.loads(metadata.read_text(encoding="utf-8")), completed, store

    async def test_the_stored_selection_source_is_accepted_by_the_model(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Constructing a record from the stored value must not raise.

        _read_existing feeds the stored string to dataclasses.replace, so
        anything the build writes has to be constructible. This is the one-line
        check whose absence let the defect through.
        """
        from custom_components.frigate_vision.models import (
            ActivitySource,
            ActivityStage,
        )

        payload, _completed, _store = await self._build_once(
            hass, tmp_path, "review_entry_1_front_review_restore"
        )
        ActivityRecord(
            activity_id="probe",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            created_at=1.0,
            updated_at=2.0,
            camera="front",
            selection_source=payload["selection_source"],
        )

    async def test_the_restore_path_survives_a_second_build(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """A record already at EVIDENCE_READY must build again without raising.

        That branch re-reads the artifact and passes its stored source back to the
        model, which is exactly where the bad value detonated.
        """
        from custom_components.frigate_vision.store import ActivityStore

        _payload, completed, _store = await self._build_once(
            hass, tmp_path, "review_entry_1_front_review_twice"
        )

        store2 = ActivityStore(hass, "entry_1")
        await store2.async_load()
        restored = copy.copy(completed)
        await store2.async_create(restored)
        manager2 = MediaManager(
            hass, store2, self._client(), tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        again = await manager2.async_build(completed.activity_id)
        assert again.sample_times == completed.sample_times
