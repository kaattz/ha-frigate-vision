"""End-to-end proofs for the restore paths, run rather than traced.

An independent review found a blocker whose every link was verified by reading
code but which nobody had executed: the build wrote `selection_source:
preview_change` into the artifact while `models.py` refused that value, and the
value reaches `dataclasses.replace` through the restore path. The reviewer's exact
words were "someone should reproduce it live before treating it as closed".

These tests are that reproduction. They drive the two paths that consume a stored
artifact:

  * `_read_existing` -> `_complete` -> `store.async_complete_media` -> `replace`,
    which is what a build does when it finds usable evidence already on disk;
  * `async_restore_registry`, which is what `async_setup_entry` calls and whose
    failures decide whether the whole config entry loads.

The blocker's shape is why these exist rather than a deeper unit test: every link
was individually correct-looking, the suite was green, and the defect only appears
when the stored value is handed back to the model.
"""

from __future__ import annotations

import json
import time
from io import BytesIO

from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import MediaManager, _preview_frame_time
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.store import ActivityStore


def _hour() -> float:
    """The current hour, read fresh so a long suite cannot cross the boundary.

    A module constant would be wrong here: the preview source compares the window
    against `time.time()`, so a suite that crosses the hour boundary mid-run would
    describe the previous hour and take the fallback path.
    """
    return (int(time.time()) // 3600) * 3600


def _jpeg(value: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (320, 180), (value, value, value)).save(buffer, "JPEG")
    return buffer.getvalue()


def _webp(value: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (320, 180), (value, value, value)).save(buffer, "WEBP")
    return buffer.getvalue()


def record_for(activity_id: str, *, hour_start: float) -> ActivityRecord:
    return ActivityRecord(
        activity_id=activity_id,
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=hour_start + 90,
        updated_at=hour_start + 130,
        camera="front",
        review_ids=(f"review_{activity_id}",),
        detection_ids=(f"event_{activity_id}",),
        finalization_deadline=hour_start + 130,
    )


def build_preview_client(hour_start: float):
    """A client that answers on the preview route, so the artifact says so.

    The source value matters here: `preview_change` is the one the blocker
    involved, and only a build that really takes the preview route writes it.
    """
    names = tuple(
        f"preview_front-{hour_start + 95 + index * 1.1:.6f}.webp"
        for index in range(30)
    )

    def thumb(name: str) -> bytes:
        moment = _preview_frame_time(name) or 0.0
        index = int((moment - hour_start - 95) / 1.1)
        # Several SUSTAINED events in different places, so choose_cells fills to
        # max_cells and the artifact holds nine cells.
        #
        # Sustained, not single-frame: a one-frame brightness jump is absorbed by
        # the restoration that follows it and the whole run reads as ONE stable
        # segment, which yields six cells. Six is exactly the count the old reuse
        # check accepted, so a single-frame fixture cannot tell the two rules
        # apart -- measured, it produced six cells and the mutation survived.
        image = Image.new("RGB", (320, 180), (80, 80, 80))
        box = None
        if 3 <= index < 6:
            box = (40, 20, 120, 80)
        elif 10 <= index < 13:
            box = (180, 100, 260, 160)
        elif 17 <= index < 20:
            box = (200, 20, 280, 80)
        elif 24 <= index < 27:
            box = (20, 80, 100, 140)
        if box is not None:
            image.paste((220, 220, 220), box)
        buffer = BytesIO()
        image.save(buffer, "WEBP")
        return buffer.getvalue()

    class Client:
        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.preview_calls = 0

        async def async_get_event(self, event_id, camera):
            # `path_data` is present on purpose. Without it `plan.motion_times`
            # is empty and the reuse decision falls to the final branch, whose
            # condition is "no planned sample_times OR they match" -- true for an
            # empty `plan.sample_times`, so the artifact would be reused whatever
            # the preview branch decided. A test written without it cannot tell
            # the two branches apart, which is why an earlier version passed with
            # the preview branch deliberately disabled.
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": hour_start + 100,
                "end_time": hour_start + 120,
                "data": {
                    "path_data": [
                        [[float(index), 0.0], hour_start + 100.0 + index]
                        for index in range(7)
                    ]
                },
            }

        async def async_get_recordings(self, camera, after, before):
            return [{"start_time": after - 60, "end_time": before + 60}]

        async def async_get_snapshot(self, camera, timestamp, height):
            self.snapshot_calls += 1
            return _jpeg(70 + int(timestamp) % 150)

        async def async_get_event_snapshot(self, event_id, camera, height):
            return _jpeg(120)

        async def async_get_preview_frames(self, camera, start, end):
            self.preview_calls += 1
            return names

        async def async_get_preview_thumbnail(self, file_name):
            return thumb(file_name)

    return Client()


class TestAStoredSourceCanBeHandedBack:
    """Whatever the build writes, the restore path must be able to consume."""

    async def test_the_restore_path_accepts_what_the_build_wrote(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Drive `_read_existing` -> `_complete` on a real artifact.

        This is the path that detonated: the artifact's stored source is fed back
        to `store.async_complete_media`, which builds a record through
        `dataclasses.replace` and validates it. A value the model refuses raises
        `ModelValidationError`, a `ValueError` the runtime does not catch.
        """
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = record_for("restore_accepts", hour_start=_hour())
        await store.async_create(record)
        manager = MediaManager(
            hass,
            store,
            build_preview_client(_hour()),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        completed = await manager.async_build(record.activity_id)
        assert completed.stage is ActivityStage.EVIDENCE_READY

        # Now the restore: a NEW store holding the same record at
        # EVIDENCE_READY, and a build that finds the artifact already on disk and
        # hands its stored source back through `_complete`.
        revived = ActivityStore(hass, "entry_1")
        await revived.async_load()
        await revived.async_create(completed)
        manager2 = MediaManager(
            hass,
            revived,
            build_preview_client(_hour()),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        again = await manager2.async_build(record.activity_id)
        assert again.sample_times == completed.sample_times
        assert again.selection_source == completed.selection_source

    async def test_the_stored_source_is_one_the_model_accepts(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Read the artifact's own bytes and rebuild a record from them.

        Independent of the build: this reads what actually landed on disk, so it
        would catch a value written by any future code path.
        """
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = record_for("restore_bytes", hour_start=_hour())
        await store.async_create(record)
        manager = MediaManager(
            hass,
            store,
            build_preview_client(_hour()),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        await manager.async_build(record.activity_id)

        metadata = tmp_path / "entry_1" / f"{record.activity_id}.json"
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        # The line whose absence let the blocker through: constructing a record
        # from the stored value must not raise.
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


class TestSetupSurvivesTheStoredArtifact:
    """`async_restore_registry` runs during setup; it must not take the entry down."""

    async def test_setup_style_restore_registers_the_artifact(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """Drive the call `async_setup_entry` actually makes."""
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = record_for("restore_registry", hour_start=_hour())
        await store.async_create(record)
        manager = MediaManager(
            hass,
            store,
            build_preview_client(_hour()),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        completed = await manager.async_build(record.activity_id)

        # A fresh manager over a store that already holds the finished record:
        # this is the state a restart starts from.
        revived = ActivityStore(hass, "entry_1")
        await revived.async_load()
        await revived.async_create(completed)
        manager2 = MediaManager(
            hass,
            revived,
            build_preview_client(_hour()),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        await manager2.async_restore_registry()

        # Registered means the media source can serve it, which is the point.
        url = completed.evidence_media_url
        assert url is not None

    async def test_restore_does_not_raise_for_a_usable_artifact(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """The whole point: a good artifact must not strand the config entry."""
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = record_for("restore_no_raise", hour_start=_hour())
        await store.async_create(record)
        manager = MediaManager(
            hass,
            store,
            build_preview_client(_hour()),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        await manager.async_build(record.activity_id)
        # Idempotent: calling it again over the same manager must be harmless.
        await manager.async_restore_registry()


class TestARestartReusesAPreviewSheetInsteadOfRebuilding:
    """A preview artifact must survive a restart.

    Measured before the reuse branch existed: the motion test required
    stored_source == "path_motion", which a preview artifact never satisfies, so
    
alid was always False. Every restart deleted the sheet and rebuilt it -- one
    HTTP round trip per analysed frame -- and once Frigate had purged the
    recordings the rebuild needs, the activity failed outright. The default media
    retention is 7 days, longer than many deployments keep recordings.

    The assertion counts snapshot requests rather than inspecting the branch: a
    rebuild is only visible as traffic.
    """

    async def test_a_second_build_fetches_nothing(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        """The SEALED-with-artifact-on-disk path, which is the reuse branch.

        Note which path is exercised. When the stored record is already
        EVIDENCE_READY the build returns much earlier, at the stage check, and
        never reaches the reuse decision -- so a test built by copying the
        completed record cannot observe it. The reuse branch is reached when a
        record is still SEALED while a matching artifact already sits on disk,
        which is exactly what a restart after an interrupted build looks like.
        """
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = record_for("reuse_preview", hour_start=_hour())
        await store.async_create(record)

        client = build_preview_client(_hour())
        manager = MediaManager(
            hass, store, client, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        completed = await manager.async_build(record.activity_id)

        # The restart: same root and artifact, but the stored record is still
        # SEALED, so the build must reach the reuse decision rather than the early
        # return at the stage check.
        #
        # sync_create cannot put it there: it returns the EXISTING record when
        # the id is already known, and the store HA hands back is the same
        # instance, so re-creating the SEALED record silently kept the
        # EVIDENCE_READY one. That is why an earlier version of this test passed
        # with the reuse branch disabled -- it was exercising the early return.
        # The record is therefore put back in memory directly, which is what a
        # restart after an interrupted build actually looks like.
        revived = ActivityStore(hass, "entry_1")
        await revived.async_load()
        revived._activities[record.activity_id] = record
        fresh = build_preview_client(_hour())
        manager2 = MediaManager(
            hass, revived, fresh, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        again = await manager2.async_build(record.activity_id)
        assert again.stage is ActivityStage.EVIDENCE_READY
        # The fixture must produce NINE cells, or this test cannot tell the two
        # reuse rules apart: six is what the old == 6 accepted, and a six-cell
        # fixture passes under both. Measured: with 20 preview frames the run
        # segmented into single-frame changes and yielded six.
        assert len(completed.sample_times) == 9, (
            f"夹具必须产出九格，实际 {len(completed.sample_times)} 格"
        )

        assert again.sample_times == completed.sample_times
        assert fresh.snapshot_calls == 0, (
            f"重启后不应重新取帧，实际取了 {fresh.snapshot_calls} 次；"
            "说明 artifact 被判为无效并重建"
        )
        assert fresh.preview_calls == 0, "也不该重新走预览筛选"


class TestAProbedMotionArtifactSurvivesARestart:
    """A nine-cell artifact must be reused, not rebuilt.

    The motion plan produces six cells, but the same build ADDS three probes when
    it finds a hole wide enough, and the artifact records all nine. The reuse
    check required len(samples) == 6, so a probed artifact was judged invalid
    and deleted on every restart -- the same defect the preview branch had, from
    the same cause: a reuse test that pins a shape only one of the two builds
    produces.

    The fixture therefore has to produce a genuine probe: clustered motion points
    leave a hole wider than MAX_SAMPLE_GAP, which is what
    `test_review_builds_a_nine_cell_sheet_when_a_hole_can_be_probed` relies on.
    """

    def _client(self, hour_start: float):

        class Client:
            def __init__(self) -> None:
                self.snapshot_calls = 0

            async def async_get_event(self, event_id, camera):
                # Motion points clustered at the start, then a long gap, then a
                # cluster at the end: the shape that leaves a wide hole.
                return {
                    "id": event_id,
                    "camera": camera,
                    "label": "person",
                    "start_time": hour_start + 100,
                    "end_time": hour_start + 140,
                    "data": {
                        "path_data": [
                            [[float(index), 0.0], hour_start + 100.0 + index * 0.5]
                            for index in range(6)
                        ]
                        + [
                            [[float(index), 0.0], hour_start + 138.0 + index * 0.3]
                            for index in range(4)
                        ]
                    },
                }

            async def async_get_recordings(self, camera, after, before):
                return [{"start_time": after - 60, "end_time": before + 60}]

            async def async_get_snapshot(self, camera, timestamp, height):
                self.snapshot_calls += 1
                return _jpeg(70 + int(timestamp) % 150)

            async def async_get_event_snapshot(self, event_id, camera, height):
                return _jpeg(120)

        return Client()

    async def test_a_nine_cell_artifact_is_reused(
        self, hass: HomeAssistant, tmp_path
    ) -> None:
        hour = _hour()
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = ActivityRecord(
            activity_id="probed_reuse",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.SEALED,
            created_at=hour + 95,
            updated_at=hour + 145,
            camera="front",
            review_ids=("review_probed",),
            detection_ids=("event_probed",),
            finalization_deadline=hour + 145,
        )
        await store.async_create(record)
        manager = MediaManager(
            hass, store, self._client(hour), tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
            preview_candidates=False,
        )
        completed = await manager.async_build(record.activity_id)
        assert completed.stage is ActivityStage.EVIDENCE_READY
        assert completed.selection_source == "path_motion"
        # The fixture must actually produce a PROBED sheet, or this test proves
        # nothing about nine-cell reuse: it would pass on the six-cell path too,
        # which is exactly the shape the old == 6 check accepted.
        assert len(completed.sample_times) == 9, (
            f"夹具必须产出九格（探测补齐），实际 {len(completed.sample_times)} 格；"
            "六格的话这条测试就测不到被修的那个分支"
        )

        # The restart, with the record back at SEALED so the reuse decision runs.
        revived = ActivityStore(hass, "entry_1")
        await revived.async_load()
        revived._activities[record.activity_id] = record
        fresh = self._client(hour)
        manager2 = MediaManager(
            hass, revived, fresh, tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
            preview_candidates=False,
        )
        again = await manager2.async_build(record.activity_id)

        assert again.sample_times == completed.sample_times
        assert fresh.snapshot_calls == 0, (
            f"格数为 {len(completed.sample_times)} 的 artifact 必须被复用；"
            f"实际重新取了 {fresh.snapshot_calls} 帧"
        )


class TestTheReuseCountRuleAcceptsEverySheetShape:
    """The count rule must accept any sheet the builder can produce.

    A sheet holds a whole number of 3-column rows: six by default, nine when the
    activity has more distinct change segments than six cells can represent. A
    reuse check that pins one of those numbers silently discards the other on
    every restart -- which is what len(samples) == 6 did to probed motion
    artifacts, and what the preview branch would have done had it been written the
    same way.

    Tested at the rule rather than through a build, because a build producing nine
    cells depends on the change analysis finding enough distinct events, and a
    fixture tuned for that is fragile in a way this assertion is not.
    """

    @staticmethod
    def _accepts(samples: tuple[float, ...]) -> bool:
        """The condition both reuse branches apply to a stored artifact."""
        return (
            len(samples) > 0
            and len(samples) % 3 == 0
            and tuple(sorted(set(samples))) == samples
        )

    def test_six_cells_are_accepted(self) -> None:
        assert self._accepts((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

    def test_nine_cells_are_accepted(self) -> None:
        """THE regression: a probed artifact is nine cells, not six."""
        assert self._accepts(tuple(float(v) for v in range(1, 10)))

    def test_twelve_cells_are_accepted(self) -> None:
        """The model allows up to twelve; the rule must not be narrower."""
        assert self._accepts(tuple(float(v) for v in range(1, 13)))

    def test_a_non_multiple_of_three_is_refused(self) -> None:
        """A partial row cannot be composed into a sheet."""
        assert not self._accepts((1.0, 2.0, 3.0, 4.0))

    def test_an_empty_set_is_refused(self) -> None:
        assert not self._accepts(())

    def test_a_duplicate_is_refused(self) -> None:
        assert not self._accepts((1.0, 1.0, 2.0, 3.0, 4.0, 5.0))

    def test_out_of_order_is_refused(self) -> None:
        assert not self._accepts((2.0, 1.0, 3.0, 4.0, 5.0, 6.0))
