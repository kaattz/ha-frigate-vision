"""The preview route must not lose an activity the recordings route can build.

Measured twice on the live deployment, on the SAME activity, inside its own hour:

    14:52:03  review, 22.9 s, 2 detections
      OLD (recordings only)    OK    recordings/path_motion  9 cells
      NEW (preview first)      FAIL  MediaError: duplicate_evidence_frame

    18:48:07  review, 36.9 s, 3 detections
      OLD (recordings only)    OK    recordings/path_motion  6 cells
      NEW (preview first)      FAIL  MediaError: duplicate_evidence_frame

The collision itself is now named rather than inferred. The guard refused cells 0
and 1 with `changed=0.00000`: not one of the 2304 signature pixels differs by more
than the noise level. The two instants are near `first_time`, where the person has
not yet moved, so the corridor is still and the snapshots are the same picture. The
repair passes cannot fix it -- every offset they try lands on another frame of the
same still moment.

So the preview route, which the code calls "an economy, not a dependency", becomes
a dependency at exactly the wrong time: it turns a buildable activity into a failed
one. These tests pin the requirement that it must not.

They are written against `MediaManager` rather than a helper, because the defect is
in the ORDER of the build -- preview selection is attempted, and a later failure has
no way back -- not in any single function.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import MediaError, MediaManager
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.store import ActivityStore


def _hour() -> int:
    """The current hour, read fresh so a long suite cannot cross the boundary."""
    import time

    return int(time.time() // 3600) * 3600


def _jpeg(colour: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (640, 360), (colour, colour, colour)).save(buffer, "JPEG")
    return buffer.getvalue()


def _record(activity_id: str) -> ActivityRecord:
    hour = _hour()
    return ActivityRecord(
        activity_id=activity_id,
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        created_at=hour + 90,
        updated_at=hour + 130,
        camera="front",
        review_ids=(f"review_{activity_id}",),
        detection_ids=("event_1",),
        finalization_deadline=hour + 130,
    )


def _preview_client(hour: int, *, snapshots_collide: bool):
    """A client whose recordings snapshots collide at the first two cells.

    Two independent controls, because the live failure needs both:

    * the preview THUMBNAILS must differ, or `_async_preview_timestamps` returns
      None by design (no change means nothing to select) and the recordings route
      runs on its own -- a fallback that already works;
    * the recordings SNAPSHOTS at cells 0 and 1 must be identical, which is what
      the guard refused on 14:52:03 (`changed=0.00000` over the whole signature).

    With `snapshots_collide=False` the snapshots are distinct, so a preview build
    succeeds -- the zero-regression case.
    """
    names = tuple(
        f"preview_front-{hour + 95 + index * 0.5:.6f}.webp" for index in range(24)
    )

    def thumb(index: int) -> bytes:
        # A large whole-frame step, so the change analysis has something to find.
        # The first few frames hold still and then it moves, which mirrors a person
        # appearing: the moments before that are the ones that collided.
        value = 40 if index < 4 else (200 if index < 10 else 90)
        buffer = BytesIO()
        Image.new("RGB", (320, 180), (value, value, value)).save(buffer, "WEBP")
        return buffer.getvalue()

    class Client:
        def __init__(self) -> None:
            self.snapshot_calls = 0
            self.preview_thumb_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": hour + 100,
                "end_time": hour + 120,
            }

        async def async_get_recordings(
            self, camera: str, after: float, before: float
        ):
            return [{"start_time": after - 60, "end_time": before + 60}]

        async def async_get_snapshot(
            self, camera: str, timestamp: float, height: int
        ):
            self.snapshot_calls += 1
            if snapshots_collide:
                # Everything in the first second is ONE picture -- the still
                # corridor before the person moved. Cells chosen inside it are
                # therefore duplicates, which is the live failure.
                key = 0 if timestamp < hour + 101 else int(timestamp * 10) % 180 + 1
            else:
                key = int(round(timestamp * 10)) % 200
            return _jpeg(20 + key)

        async def async_get_event_snapshot(
            self, event_id: str, camera: str, height: int
        ):
            return _jpeg(90)

        async def async_get_preview_frames(
            self, camera: str, start: float, end: float
        ):
            return names

        async def async_get_preview_thumbnail(self, file_name: str) -> bytes:
            self.preview_thumb_calls += 1
            moment = float(file_name.rsplit("-", 1)[-1].removesuffix(".webp"))
            return thumb(int(round((moment - hour - 95) / 0.5)))

    return Client()


async def _build(hass: HomeAssistant, tmp_path: Path, activity_id: str, client):
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _record(activity_id)
    await store.async_create(record)
    manager = MediaManager(
        hass,
        store,
        client,
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    return manager, record


class TestAFailedPreviewBuildFallsBackToRecordings:
    """A preview failure must not cost an activity the recordings route can build."""

    async def test_a_preview_collision_still_produces_a_sheet(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """THE regression: the activity the live deployment lost twice.

        The preview selection collides -- every thumbnail is the same picture -- and
        the guard refuses it. The build must then take the recordings route, which
        this client can satisfy, rather than raising.
        """
        client = _preview_client(_hour(), snapshots_collide=True)
        manager, record = await _build(
            hass, tmp_path, "review_entry_1_front_fallback", client
        )
        built = await manager.async_build(record.activity_id)
        assert built.stage is ActivityStage.EVIDENCE_READY
        assert built.evidence_path is not None
        exists = await hass.async_add_executor_job(
            lambda: Path(built.evidence_path).is_file()
        )
        assert exists

    async def test_the_fallback_is_recorded_as_a_fallback(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """The artifact must say which route actually built it.

        A sheet that silently came from the recordings route would read as a
        preview success, and the next person debugging the preview path would be
        looking at evidence that never exercised it.
        """
        client = _preview_client(_hour(), snapshots_collide=True)
        manager, record = await _build(
            hass, tmp_path, "review_entry_1_front_fallback_meta", client
        )
        built = await manager.async_build(record.activity_id)
        metadata = Path(built.evidence_path).with_suffix(".json")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        assert payload["candidate_source"] == "recordings", (
            "预览失败后应当记录为录像路径，否则回退会被误读为预览成功"
        )
        assert payload["selection_source"] != "preview_change"

    async def test_the_fallback_does_not_reuse_the_preview_selection(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """The recordings route must plan its own frames, not retry the preview's.

        Reusing the preview timestamps would fetch the same colliding instants and
        fail again -- the retry would be theatre. The recordings plan is derived
        from `path_motion`, so its frames differ.
        """
        client = _preview_client(_hour(), snapshots_collide=True)
        manager, record = await _build(
            hass, tmp_path, "review_entry_1_front_fallback_fresh", client
        )
        built = await manager.async_build(record.activity_id)
        metadata = Path(built.evidence_path).with_suffix(".json")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        # A preview-plan artifact carries the preview bookkeeping; a recordings
        # plan does not.
        assert payload.get("preview_frame_count") is None, (
            "回退后不应带上预览路径的统计字段"
        )

    async def test_a_successful_preview_build_is_unchanged(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """Zero regression: when preview works, it is still what runs.

        The fallback must be reachable ONLY after a failure. If it ran eagerly, the
        whole point of the preview route -- cheaper frames, and seeing the entire
        activity rather than the three motion peaks -- would be lost silently.
        """
        client = _preview_client(_hour(), snapshots_collide=False)
        manager, record = await _build(
            hass, tmp_path, "review_entry_1_front_preview_ok", client
        )
        built = await manager.async_build(record.activity_id)
        metadata = Path(built.evidence_path).with_suffix(".json")
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        assert payload["candidate_source"] == "preview"
        assert payload["selection_source"] == "preview_change"


class TestTheFallbackIsNotAMask:
    """A fallback must not hide a broken recordings route either."""

    async def test_when_both_routes_fail_the_error_is_raised(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """If neither route can build, the activity still fails -- loudly.

        The fallback exists to rescue an activity, not to swallow errors. A build
        that cannot be completed by either route must still raise, or a genuinely
        broken camera would produce silent nothing instead of a failure the owner
        can see on `sensor.*_last_error`.
        """
        hour = _hour()

        class Broken:
            async def async_get_event(self, event_id: str, camera: str):
                return {
                    "id": event_id,
                    "camera": camera,
                    "label": "person",
                    "start_time": hour + 100,
                    "end_time": hour + 120,
                }

            async def async_get_recordings(
                self, camera: str, after: float, before: float
            ):
                # No coverage at all, so neither route can proceed.
                return []

            async def async_get_snapshot(
                self, camera: str, timestamp: float, height: int
            ):
                raise MediaError("frigate_unavailable")

            async def async_get_event_snapshot(
                self, event_id: str, camera: str, height: int
            ):
                raise MediaError("frigate_unavailable")

            async def async_get_preview_frames(
                self, camera: str, start: float, end: float
            ):
                return ()

            async def async_get_preview_thumbnail(self, file_name: str) -> bytes:
                raise MediaError("frigate_unavailable")

        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record("review_entry_1_front_both_broken")
        await store.async_create(record)
        manager = MediaManager(
            hass,
            store,
            Broken(),
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        with pytest.raises(MediaError):
            await manager.async_build(record.activity_id)


class TestOnlyTheMeasuredFailureIsRetried:
    """The retry is scoped to the one failure with evidence behind it.

    Retrying every `MediaError` would mean guessing. The fallback exists because the
    preview SELECTION can pick two instants that fetch one picture -- measured twice,
    by the guard naming cells 0 and 1 with `changed=0.00000`. No other media error has
    been shown to be cured by changing the plan: a recording gap is still a gap. A
    retry for those would spend a second full build to reach the same failure, and
    would make the activity take twice as long to report it.
    """

    async def test_a_recording_gap_is_not_retried(
        self, hass: HomeAssistant, tmp_path: Path
    ) -> None:
        """COUNTED, not asserted by exception type alone.

        The point is that the build runs ONCE. A retry shows up as a second
        recordings lookup, which an exception-type assertion cannot see.
        """
        hour = _hour()

        class Gapped:
            def __init__(self) -> None:
                self.recording_calls = 0

            async def async_get_event(self, event_id: str, camera: str):
                return {
                    "id": event_id,
                    "camera": camera,
                    "label": "person",
                    "start_time": hour + 100,
                    "end_time": hour + 120,
                }

            async def async_get_recordings(
                self, camera: str, after: float, before: float
            ):
                self.recording_calls += 1
                # Never covers the window, so `recording_gap` is raised.
                return []

            async def async_get_snapshot(
                self, camera: str, timestamp: float, height: int
            ):
                raise MediaError("frigate_unavailable")

            async def async_get_event_snapshot(
                self, event_id: str, camera: str, height: int
            ):
                raise MediaError("frigate_unavailable")

            async def async_get_preview_frames(
                self, camera: str, start: float, end: float
            ):
                return ()

            async def async_get_preview_thumbnail(self, file_name: str) -> bytes:
                raise MediaError("frigate_unavailable")

        client = Gapped()
        store = ActivityStore(hass, "entry_1")
        await store.async_load()
        record = _record("review_entry_1_front_gap_not_retried")
        await store.async_create(record)
        manager = MediaManager(
            hass,
            store,
            client,
            tmp_path,
            ZoneRoles(frozenset(), frozenset(), frozenset()),
        )
        with pytest.raises(MediaError, match="recording_gap"):
            await manager.async_build(record.activity_id)
        assert client.recording_calls == 1, (
            f"recording_gap 不该重试；实际查询了 {client.recording_calls} 次录像，"
            "说明重试范围超过了已测量的那一种失败"
        )
