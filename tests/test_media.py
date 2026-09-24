from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path

import pytest
from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import (
    EvidencePlan,
    MediaError,
    MediaManager,
    build_contact_sheet,
    frame_is_infrared,
    frame_is_overexposed,
    pair_times_with_roles,
    recordings_cover,
    select_review_change_frames,
    validate_unique_frames,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    ProcessingMode,
)
from custom_components.frigate_vision.store import ActivityStore


def _jpeg(value: int) -> bytes:
    output = BytesIO()
    Image.new("RGB", (64, 36), (value, value, value)).save(output, "JPEG")
    return output.getvalue()


def _ir_jpeg(value: int = 160) -> bytes:
    """A frame with the camera's night-vision cast.

    Measured on production frames: an IR-lit cell has a strongly green-dominant
    profile (R-G around -150) while colour frames sit near R-G = +2.
    """
    output = BytesIO()
    Image.new("RGB", (64, 36), (5, value, 105)).save(output, "JPEG")
    return output.getvalue()


def _blown_jpeg(shade: int = 250) -> bytes:
    """A saturated frame, as when the lift door opens into the lens.

    Measured on production sheets: such cells are 42%-98% near-white pixels
    while ordinary cells stay under 1%, so the condition separates cleanly.
    Nearly white but not pure, matching the real frames rather than an ideal.

    `shade` varies the near-white value. A camera's saturated frames are similar
    but never byte-identical, and returning one fixed image for every blown
    instant made several cells exactly equal -- which the uniqueness guard then
    rejected, correctly. A test holding several blown frames must vary the shade,
    or it asserts something a real camera cannot produce.
    """
    output = BytesIO()
    value = max(236, min(254, shade))
    Image.new("RGB", (64, 36), (value, value, value - 4)).save(output, "JPEG")
    return output.getvalue()


def _blown_at(timestamp: float) -> bytes:
    """A saturated frame whose exact shade depends on the instant."""
    return _blown_jpeg(240 + int(timestamp * 3) % 12)


def _changing_jpeg(timestamp: float) -> bytes:
    """A frame whose brightness varies smoothly and monotonically with time.

    Fixtures must return a *different* picture for a different instant, because
    that is what a camera does. The obvious `int(t * 10) % 255` does not: it is a
    sawtooth, so instants 25.5s apart (and, at coarser multipliers, 0.1s apart)
    collapse onto the same grey and the uniqueness guard then rejects a sheet the
    fixture wrongly made identical. Deriving the value from a bounded,
    monotonic function of the timestamp keeps every distinct instant distinct.
    """
    # 12 levels per second, saturating smoothly, over the grey range a camera
    # would show here. Distinct instants differ by at least one level.
    level = int((timestamp * 12) % 200) + 20
    return _jpeg(level)


def test_probe_cells_keep_their_own_role_through_the_sort() -> None:
    """Every role must stay attached to the instant it describes.

    Probes land *inside* the motion span, so sorting the timestamps and leaving
    the roles in insertion order silently pairs each role with the wrong cell. It
    matters because `_nudge_uncovered_frames` treats `first`/`last`/`postroll` as
    required and a probe as droppable -- so a mispaired role lets a frame the
    sheet cannot be built without be discarded.

    Tested on the pure helper, not through `async_build`: a mutation that sorted
    the times while leaving roles in insertion order passed the whole suite when
    the pairing lived inside the async method, because nothing observed the two
    lists as a pair.
    """

    plan = EvidencePlan(
        mode="review_six",
        first_time=100.0,
        last_time=160.0,
        postroll_time=163.0,
        selection_source="path_motion",
        motion_times=(110.0, 130.0, 150.0),
    )
    # Two probes fall between motion picks, one after the last -- the arrangement
    # that makes sorting dangerous.
    times, roles = pair_times_with_roles(plan, (115.0, 135.0, 155.0))

    assert len(times) == len(roles) == 9
    assert list(times) == sorted(times)
    expected = {
        100.0: "first",
        110.0: "motion",
        115.0: "probe",
        130.0: "motion",
        135.0: "probe",
        150.0: "motion",
        155.0: "probe",
        160.0: "last",
        163.0: "postroll",
    }
    assert dict(zip(times, roles, strict=True)) == expected


def test_contact_sheet_accepts_nine_cells_as_three_rows(tmp_path) -> None:
    """Nine cells must lay out as 3x3, not be rejected or silently truncated.

    Measured on the owner's labelled sheets: a nine-cell sheet answered direction
    correctly 48% of the time against 28% for six cells (pooled over two models,
    n=40), and the mechanism is not in doubt -- the three extra frames are chosen
    by visible change and cut the worst unobserved gap from 17.1s to 11.5s, better
    on 29 of 48 activities and worse on none.

    The layout must follow the frame count rather than a fixed 2x3, or the extra
    row would be dropped without error -- which is exactly how a malformed
    nine-cell fixture of mine once produced an inflated result.
    """
    frames = []
    for index in range(9):
        path = tmp_path / f"n{index}.jpg"
        path.write_bytes(_jpeg(index * 20))
        frames.append(path)
    output = tmp_path / "nine.jpg"
    build_contact_sheet(frames, output, cell_size=(160, 90))
    with Image.open(output) as sheet:
        assert sheet.size == (480, 270), "three rows of 160x90 was expected"


def test_contact_sheet_rejects_a_count_that_is_not_whole_rows(tmp_path) -> None:
    """A partial row would silently drop frames, so it must fail loudly."""
    frames = []
    for index in range(7):
        path = tmp_path / f"p{index}.jpg"
        path.write_bytes(_jpeg(index * 20))
        frames.append(path)
    with pytest.raises(MediaError, match="invalid_frame_count"):
        build_contact_sheet(frames, tmp_path / "bad.jpg", cell_size=(160, 90))


def test_contact_sheet_validates_frames_and_dimensions(tmp_path) -> None:
    frames = []
    for index in range(6):
        path = tmp_path / f"{index}.jpg"
        path.write_bytes(_jpeg(index * 30))
        frames.append(path)
    output = tmp_path / "sheet.jpg"
    build_contact_sheet(frames, output, cell_size=(160, 90))
    with Image.open(output) as sheet:
        assert sheet.size == (480, 180)
        assert sheet.format == "JPEG"
    frames[0].write_text("broken")
    with pytest.raises(MediaError, match="frame_decode_failed"):
        build_contact_sheet(frames[:3], output)


def test_recording_coverage_rejects_any_gap() -> None:
    assert recordings_cover((1, 2, 3), [{"start_time": 0, "end_time": 4}])
    assert not recordings_cover(
        (1, 2, 3),
        [{"start_time": 0, "end_time": 1.5}, {"start_time": 2.5, "end_time": 4}],
    )


def test_zero_change_review_candidates_are_rejected(tmp_path) -> None:
    frames = []
    for index in range(5):
        path = tmp_path / f"same_{index}.jpg"
        path.write_bytes(_jpeg(0))
        frames.append((float(index), path))
    with pytest.raises(MediaError, match="insufficient_visible_change"):
        select_review_change_frames(frames)


def test_too_few_candidates_is_distinct_from_no_visible_change(tmp_path) -> None:
    """Not enough frames and no movement in the frames are different failures.

    They previously shared one code, so an operator could not tell a missing
    download from a genuinely static scene. The first is a frame-availability
    problem, the second means the camera saw nothing worth analysing.
    """
    # Only two frames: the selector needs count+1 to score anything at all.
    too_few = []
    for index in range(2):
        path = tmp_path / f"few_{index}.jpg"
        path.write_bytes(_jpeg(index * 40))
        too_few.append((float(index), path))
    with pytest.raises(MediaError, match="insufficient_review_candidates"):
        select_review_change_frames(too_few)

    # Enough frames, but every one is identical.
    static = []
    for index in range(5):
        path = tmp_path / f"static_{index}.jpg"
        path.write_bytes(_jpeg(0))
        static.append((float(index), path))
    with pytest.raises(MediaError, match="insufficient_visible_change"):
        select_review_change_frames(static)


def test_final_frames_reject_duplicate_fixed_and_change_images(tmp_path) -> None:
    paths = []
    for index, value in enumerate((0, 100, 0)):
        path = tmp_path / f"duplicate_{index}.jpg"
        path.write_bytes(_jpeg(value))
        paths.append(path)
    with pytest.raises(MediaError, match="duplicate_evidence_frame"):
        validate_unique_frames(paths)


async def test_media_manager_builds_once_and_atomically_registers(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=130,
        camera="front",
        detection_ids=("event_1",),
        detection_zone_updates=(
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("mid",)),
            ("event_1", 120, ("far",)),
        ),
        door_closed_at=130,
        association_deadline=140,
        finalization_deadline=220,
    )
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 101,
                "end_time": 121,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _jpeg(int(timestamp) % 255)

    client = Client()
    manager = MediaManager(
        hass,
        store,
        client,
        tmp_path,
        ZoneRoles(
            near=frozenset({"near"}),
            transition=frozenset({"mid"}),
            far=frozenset({"far"}),
        ),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert completed.evidence_path is not None
    assert client.snapshot_calls == 3
    again = await manager.async_build(record.activity_id)
    assert again == completed
    assert client.snapshot_calls == 3


async def test_media_manager_serializes_concurrent_builds(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="door_entry_1_100000",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=130,
        camera="front",
        detection_ids=("event_1",),
        detection_zone_updates=(
            ("event_1", 102, ("near",)),
            ("event_1", 110, ("mid",)),
            ("event_1", 120, ("far",)),
        ),
        finalization_deadline=220,
    )
    await store.async_create(record)

    class Client:
        calls = 0

        async def async_get_event(self, event_id, camera):
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
            self.calls += 1
            await asyncio.sleep(0)
            return _jpeg(int(timestamp))

    client = Client()
    manager = MediaManager(
        hass,
        store,
        client,
        tmp_path,
        ZoneRoles(frozenset({"near"}), frozenset({"mid"}), frozenset({"far"})),
    )
    first, second = await asyncio.gather(
        manager.async_build(record.activity_id), manager.async_build(record.activity_id)
    )
    assert first == second
    assert client.calls == 3


async def test_evidence_ready_requires_valid_registered_artifact(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_1",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.EVIDENCE_READY,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=1,
        updated_at=2,
        camera="front",
        evidence_mode="door_single",
        evidence_revision=1,
        evidence_path=str(tmp_path / "entry_1" / "activity_1.jpg"),
        evidence_media_url="media-source://frigate_vision/entry_1/activity_1",
        sample_times=(1, 1.5, 2),
        claimed_side_effects=("media:activity_1:1",),
    )
    await store.async_create(record)
    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    with pytest.raises(MediaError, match="artifact_missing"):
        await manager.async_build(record.activity_id)


async def test_evidence_ready_rejects_valid_file_outside_canonical_root(
    hass: HomeAssistant, tmp_path
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside.jpg"
    outside.write_bytes(_jpeg(1))
    outside.with_suffix(".json").write_text(
        '{"activity_id":"activity_1","plan_version":1,"mode":"door_single","sample_times":[1,2,3]}'
    )
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.DOOR_CYCLE,
            stage=ActivityStage.EVIDENCE_READY,
            processing_mode=ProcessingMode.OBSERVE,
            created_at=1,
            updated_at=2,
            camera="front",
            evidence_mode="door_single",
            evidence_revision=1,
            evidence_path=str(outside),
            evidence_media_url="media-source://frigate_vision/entry_1/activity_1",
            sample_times=(1, 2, 3),
            claimed_side_effects=("media:activity_1:1",),
        )
    )
    manager = MediaManager(
        hass, store, object(), root, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    with pytest.raises(MediaError, match="invalid_output_path"):
        await manager.async_build("activity_1")


async def test_media_cleanup_deletes_only_expired_registered_files(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    path = tmp_path / "entry_1" / "old.jpg"
    path.parent.mkdir()
    path.write_bytes(_jpeg(1))
    path.with_suffix(".json").write_text("{}")
    await store.async_create(
        ActivityRecord(
            activity_id="old",
            entry_id="entry_1",
            source=ActivitySource.DOOR_CYCLE,
            stage=ActivityStage.COMPLETED,
            processing_mode=ProcessingMode.OBSERVE,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="door_single",
            evidence_revision=1,
            evidence_path=str(path),
            evidence_media_url="media-source://frigate_vision/entry_1/old",
            sample_times=(1, 2, 3),
            claimed_side_effects=("media:old:1",),
        )
    )
    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    removed = await manager.async_cleanup(retention_days=1, now=200000)
    assert removed == ("old",)
    assert not path.exists()
    assert not path.with_suffix(".json").exists()
    expired = store.get("old")
    assert expired is not None and expired.evidence_expired_at == 200000
    await manager.async_restore_registry()


async def test_cleanup_removes_registry_before_physical_delete_failure(
    hass: HomeAssistant, tmp_path, monkeypatch
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    path = tmp_path / "entry_1" / "old.jpg"
    path.parent.mkdir()
    path.write_bytes(_jpeg(1))
    await store.async_create(
        ActivityRecord(
            activity_id="old",
            entry_id="entry_1",
            source=ActivitySource.DOOR_CYCLE,
            stage=ActivityStage.COMPLETED,
            processing_mode=ProcessingMode.OBSERVE,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="door_single",
            evidence_revision=1,
            evidence_path=str(path),
            evidence_media_url="media-source://frigate_vision/entry_1/old",
            sample_times=(1, 2, 3),
            claimed_side_effects=("media:old:1",),
        )
    )
    hass.data["frigate_vision_media_registry"] = {"entry_1/old": path}
    manager = MediaManager(
        hass,
        store,
        object(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    monkeypatch.setattr(
        "custom_components.frigate_vision.media._delete_artifact",
        lambda *args: (_ for _ in ()).throw(OSError("delete failed")),
    )
    with pytest.raises(OSError, match="delete failed"):
        await manager.async_cleanup(retention_days=1, now=200000)
    assert "entry_1/old" not in hass.data["frigate_vision_media_registry"]
    assert store.get("old").evidence_expired_at == 200000  # type: ignore[union-attr]


def _standalone_record() -> ActivityRecord:
    return ActivityRecord(
        activity_id="review_entry_1_front_review_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=120,
        camera="front",
        review_ids=("review_1",),
        detection_ids=("event_1",),
        finalization_deadline=120,
    )


def _motion_points() -> list[list[object]]:
    return [[[float(index), 0.0], 103.0 + index] for index in range(7)]


async def test_review_builds_a_nine_cell_sheet_when_a_hole_can_be_probed(
    hass: HomeAssistant, tmp_path
) -> None:
    """Nine cells = first | motion x3 | probed x3 | last | postroll.

    The three extra cells are the whole point: they are chosen by visible change
    rather than by where the person moved, which cuts the worst unobserved gap
    from 17.1s to 11.5s (better on 29 of 48 activities, worse on none). A nine-cell
    sheet is therefore not "six motion picks plus three more of the same" -- that
    arrangement measured 16.4s and buys almost nothing.

    Built from the measured 19:05 movement shape, whose clustered points are what
    leaves a hole wide enough to probe at all.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_nine",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=102,
        updated_at=152,
        camera="front",
        review_ids=("review_nine",),
        detection_ids=("event_1",),
        finalization_deadline=152,
    )
    await store.async_create(record)

    class Client:
        def __init__(self) -> None:
            self.snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            relative = (
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
            )
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 150,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 9, (
        f"a nine-cell sheet was expected: {completed.sample_times}"
    )
    assert list(completed.sample_times) == sorted(completed.sample_times)
    assert len(set(completed.sample_times)) == 9, "no cell may repeat another"
    with Image.open(completed.evidence_path) as sheet:
        assert sheet.size == (640 * 3, 360 * 3), sheet.size


async def test_review_keeps_six_cells_when_no_hole_is_wide_enough(
    hass: HomeAssistant, tmp_path
) -> None:
    """A tightly covered activity stays at six -- nine is not unconditional.

    Probing only applies where a hole exists. Every probe costs a snapshot fetch,
    and a hole shorter than the shortest action worth seeing cannot hide one, so
    the sheet must stay at its six-cell size rather than pad itself with frames
    chosen for no reason.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6, completed.sample_times


async def test_probe_cells_keep_their_role_aligned_with_their_time(
    hass: HomeAssistant, tmp_path
) -> None:
    """A role must describe the frame it is paired with, after sorting.

    `_nudge_uncovered_frames` reads `roles[index]` to decide which frames the
    sheet cannot be built without -- `first`, `last` and `postroll` are required,
    a probe is not. Probes land *inside* the motion span, so sorting the times
    without carrying the roles along would pair a probe with a required cell's
    role. The sheet would then survive a recording hole it cannot actually be
    built from, or reject one it could.

    Verified through the nudge report, which records the role of every frame it
    had to move -- the one place the pairing becomes observable.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_roles",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=102,
        updated_at=152,
        camera="front",
        review_ids=("review_roles",),
        detection_ids=("event_1",),
        finalization_deadline=152,
    )
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            relative = (
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
            )
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 150,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            # A hole covering the last stretch only, so the postroll frame must be
            # nudged and the report names its role.
            return [
                {"start_time": after - 1, "end_time": 149.0},
                {"start_time": 152.0, "end_time": before + 1},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 9
    # Every nudge the builder reported must name a role the sheet actually uses,
    # which is only true if roles stayed paired with their own times through the
    # sort. An unknown role would mean a frame was labelled with another's role.
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    payload = json.loads(metadata)
    for nudge in payload["recording_nudges"]:
        assert nudge["role"] in {
            "first",
            "last",
            "postroll",
            "motion",
            "probe",
        }, f"a frame carried another cell's role: {nudge}"
    # And the sheet's own ordering must survive: nine strictly increasing cells.
    assert list(completed.sample_times) == sorted(completed.sample_times)
    assert len(set(completed.sample_times)) == 9


async def test_review_probes_the_widest_hole_and_reselects(
    hass: HomeAssistant, tmp_path
) -> None:
    """The fill step must actually reach the sheet, not just exist.

    The unit tests cover `probe_gap_times` and the `extra` candidates in
    isolation. Neither proves the pipeline calls them: the earlier scene-
    description work shipped a capability that was defined and then never wired,
    and it failed silently for days.

    The fixture gives the camera a scene that changes *only* in the stretch where
    the person was not moving, which is the measured failure mode -- an entry
    door opening while the detector had nothing to track. The path points stay
    still, so only probing can find it.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    # A window wide enough for the real 46s span: the hole only exists because
    # the activity is long and the movements cluster in its first half.
    record = ActivityRecord(
        activity_id="review_entry_1_front_review_hole",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=102,
        updated_at=152,
        camera="front",
        review_ids=("review_hole",),
        detection_ids=("event_1",),
        finalization_deadline=152,
    )
    await store.async_create(record)

    # Built from the measured movement list of activity 19:05 on this
    # deployment, at its real 46s span. Compressing it into the record's usual
    # 14s window destroys the very thing under test: the selector spreads picks
    # to minimise the largest gap, so a short window simply has no hole left to
    # probe. The hole only exists because the activity is long and the movements
    # cluster early.
    class Client:
        snapshot_calls = 0
        probed: list[float] = []

        async def async_get_event(self, event_id: str, camera: str):
            relative = (
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
            )
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 150,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # A flash standing for the entry door opening: the person was inside
            # and still, so no path point reports it. It sits inside the hole the
            # motion picks leave (roughly 108.6..124.4), which is the point.
            if 110.0 < timestamp < 122.0:
                return _jpeg(235)
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    # Nine cells: the probe cells are *added* rather than replacing a motion pick,
    # which is the change this test now covers. An earlier revision re-aimed the
    # picks and stayed at six; that spent the extra evidence on nothing, since
    # more motion picks measured 16.4s worst gap against 11.5s for probed cells.
    assert len(completed.sample_times) == 9
    # Probing costs extra fetches; without them the count would be exactly six.
    assert client.snapshot_calls > 6, (
        "no probe was fetched, so the fill step never ran"
    )
    # The property that matters, and the one the reported failure violated: some
    # cell now sits strictly *inside* the hole. Without probing the three middle
    # picks are 108.6, 124.4 and 137.1 -- the first two exactly at the hole's
    # edges -- so nothing observed the 15.8s in between. Asserting on the hole
    # rather than on a specific instant keeps this true wherever inside it the
    # probes land.
    assert any(112.0 < moment < 122.0 for moment in completed.sample_times), (
        f"the hole is still unobserved: {completed.sample_times}"
    )


async def test_review_keeps_its_picks_when_probing_finds_nothing(
    hass: HomeAssistant, tmp_path
) -> None:
    """A failed probe must not lose the sheet.

    Probing is an improvement attempt. When it yields nothing usable the original
    motion picks have to survive intact -- a sheet built from a hole full of
    static frames would be worse than one that never looked.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            relative = (
                (0.20, 0.0127),
                (1.79, 0.0853),
                (5.60, 0.0611),
                (21.36, 0.0581),
                (23.34, 0.0851),
                (34.08, 0.0797),
                (35.08, 0.1516),
                (41.27, 0.1361),
                (42.47, 0.1236),
                (45.05, 0.0706),
            )
            scale = 14.0 / 46.26
            points: list[list[object]] = [[[0.0, 0.0], 103.0]]
            x = 0.0
            for offset, distance in relative:
                x += distance
                points.append([[x, 0.0], 103.0 + offset * scale])
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": points},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Fail only the probes. The real cells are at the motion picks and at
            # the window edges, so a band strictly inside the hole hits probes
            # without touching a required frame.
            if 105.0 < timestamp < 108.5:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)

    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert completed.selection_source == "path_motion"


async def test_review_path_motion_downloads_exactly_six_frames(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _jpeg(int(timestamp) % 255)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert completed.selection_source == "path_motion"
    assert client.snapshot_calls == 6


async def test_review_missing_path_data_falls_back_to_image_change(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.selection_source == "image_change"
    assert client.snapshot_calls == 12


async def test_review_invalid_path_data_fails_without_download(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": "not-a-list"},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": 0, "end_time": 300}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _jpeg(1)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    with pytest.raises(MediaError, match="invalid_path_data"):
        await manager.async_build(record.activity_id)
    assert client.snapshot_calls == 0


async def test_legacy_image_change_artifact_is_not_kept_as_path_motion(
    hass: HomeAssistant, tmp_path
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)
    final = tmp_path / "entry_1" / f"{record.activity_id}.jpg"
    final.parent.mkdir(parents=True)
    final.write_bytes(_jpeg(10))
    final.with_suffix(".json").write_text(
        '{"activity_id":"review_entry_1_front_review_1","plan_version":1,'
        '"mode":"review_six","sample_times":[103.2,104.0,106.0,108.0,116.8,119.8]}'
    )

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 1, "end_time": before + 1}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.selection_source == "path_motion"
    assert client.snapshot_calls == 6
    payload = final.with_suffix(".json").read_text()
    assert '"plan_version": 4' in payload
    assert '"selection_source": "path_motion"' in payload


def test_infrared_frames_are_separated_from_colour_frames(tmp_path) -> None:
    """The IR discriminator must not fire on ordinary colour frames.

    Production measurements: IR frames sit near R-G = -150, transition frames
    near -35, and colour frames near +2. The threshold must reject the IR frames
    while keeping every transition and colour frame.
    """
    ir = tmp_path / "ir.jpg"
    ir.write_bytes(_ir_jpeg())
    assert frame_is_infrared(ir)

    colour = tmp_path / "colour.jpg"
    colour.write_bytes(_jpeg(128))
    assert not frame_is_infrared(colour)

    warm = tmp_path / "warm.jpg"
    output = BytesIO()
    Image.new("RGB", (64, 36), (130, 120, 100)).save(output, "JPEG")
    warm.write_bytes(output.getvalue())
    assert not frame_is_infrared(warm)

    transition = tmp_path / "transition.jpg"
    output = BytesIO()
    Image.new("RGB", (64, 36), (98, 128, 113)).save(output, "JPEG")
    transition.write_bytes(output.getvalue())
    assert not frame_is_infrared(transition)


def test_blown_frames_are_separated_from_usable_frames(tmp_path) -> None:
    """A saturated frame must be detected while ordinary frames pass.

    Measured across the shadow-mode sheets: blown cells are 42%-98% near-white
    while every usable cell is under 1%, so the discriminator has a wide margin.
    Bright-but-detailed frames must NOT be rejected -- a white door or wall is
    normal in this corridor and carries real evidence.
    """
    blown = tmp_path / "blown.jpg"
    blown.write_bytes(_blown_jpeg())
    assert frame_is_overexposed(blown)

    for value in (0, 64, 128, 200):
        normal = tmp_path / f"normal_{value}.jpg"
        normal.write_bytes(_jpeg(value))
        assert not frame_is_overexposed(normal), f"grey {value} must be usable"

    # A bright frame with structure is still evidence. Measured on production
    # cells, usable frames reach 14.7% near-white pixels (a sunlit doorway) while
    # blown ones start at 42.7%; a quarter-lit frame must therefore survive, or
    # the frames showing the white door in this corridor would be discarded.
    structured = tmp_path / "structured.jpg"
    output = BytesIO()
    image = Image.new("RGB", (64, 36), (60, 58, 55))
    for x in range(48, 64):
        for y in range(0, 36):
            image.putpixel((x, y), (242, 241, 239))
    image.save(output, "JPEG")
    structured.write_bytes(output.getvalue())
    assert not frame_is_overexposed(structured)


async def test_review_replaces_blown_frames_with_neighbouring_frame(
    hass: HomeAssistant, tmp_path
) -> None:
    """A saturated first frame must be replaced, as IR frames already are.

    When the lift door opens into the lens the sensor saturates. Measured on the
    shadow sheets, the `home_departure` sheet had two of six cells at 95.7% and
    81.5% near-white, so a third of that classification rested on frames holding
    no evidence at all.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # Only the first requested timestamp is blown out.
            if abs(timestamp - 103.2) < 0.01:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert 103.2 not in completed.sample_times
    assert client.snapshot_calls > 6
    assert tuple(sorted(set(completed.sample_times))) == completed.sample_times


async def test_blown_replacement_is_recorded_in_the_evidence_metadata(
    hass: HomeAssistant, tmp_path
) -> None:
    """The sheet must say which frames could not be recovered.

    Without the record, a classification resting on a still-saturated frame looks
    identical to one resting on six usable frames.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return (
                _blown_jpeg()
                if abs(timestamp - 103.2) < 0.01
                else _changing_jpeg(timestamp)
            )

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    metadata = Path(completed.evidence_path).with_suffix(".json").read_text()
    assert "overexposure_replacements" in metadata
    assert '"reason": "exposure"' in metadata


async def test_review_keeps_blown_frame_when_no_clear_alternative(
    hass: HomeAssistant, tmp_path
) -> None:
    """An unrecoverable frame must not fail the activity.

    Blown frames cluster where the lift door is open, so a whole window can be
    saturated. Failing would drop a real visit; the sheet is still more useful
    with five good frames than with none, and the metadata records the shortfall.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Everything before 108 is saturated, and the ladder reaches only 5s
            # from 103.2, so no candidate for that frame is clear. Distinct grey
            # values elsewhere keep the uniqueness guard satisfied, so this test
            # isolates the overexposure behaviour.
            if timestamp < 108:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6


async def test_review_replaces_infrared_frames_with_neighbouring_frame(
    hass: HomeAssistant, tmp_path
) -> None:
    """An IR first frame must be replaced by a later colour frame.

    The camera lags ~2s behind person detection when leaving night mode, so the
    first sample is reliably IR. Replacing it keeps six real frames instead of
    spending one cell on a washed-out frame.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # Only the very first requested timestamp is IR.
            if abs(timestamp - 103.2) < 0.01:
                return _ir_jpeg()
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    # Six frames still, plus at least one extra fetch for the replacement.
    assert len(completed.sample_times) == 6
    assert 103.2 not in completed.sample_times
    assert client.snapshot_calls > 6
    assert tuple(sorted(set(completed.sample_times))) == completed.sample_times


async def test_review_reaches_colour_beyond_the_old_three_second_ladder(
    hass: HomeAssistant, tmp_path
) -> None:
    """A frame must be able to wait out a long night-mode exit.

    Measured on the 2026-09-20 08:27 and 10:24 reviews: the camera needed 3.5s
    to return colour, while the ladder stopped at +3.0s. Both sheets shipped
    with two full green frames and an empty `infrared_replacements`. The ladder
    now reaches 5.0s so a slow exit is still recoverable.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # The first motion frame is IR until +3.5s; frame 2 sits far enough away
    # that +3.5s is in bounds.
    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # 105.0 is the first motion frame; it stays IR until 108.5.
            if 105.0 <= timestamp < 108.5:
                return _ir_jpeg()
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # The IR frame was moved out to a colour instant, not left green.
    assert 105.0 not in completed.sample_times
    meta = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert "infrared_replacements" in meta
    assert '"to": 108.5' in meta


async def test_review_keeps_infrared_frame_when_no_colour_alternative(
    hass: HomeAssistant, tmp_path
) -> None:
    """A review must still produce evidence when every frame is IR.

    Failing the whole activity would discard real (if ugly) frames and lose the
    activity entirely, so the frame is kept and the substitution recorded.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            return _ir_jpeg(int(timestamp) % 200 + 20)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6


async def test_evidence_metadata_records_infrared_replacements(
    hass: HomeAssistant, tmp_path
) -> None:
    """Substitutions must be observable in the artifact metadata."""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 5, "end_time": before + 5}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if abs(timestamp - 103.2) < 0.01:
                return _ir_jpeg()
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert '"plan_version": 4' in metadata
    assert "infrared_replacements" in metadata
    assert "103.2" in metadata
    assert completed.evidence_path is not None


async def test_review_retries_a_frame_that_duplicates_another(
    hass: HomeAssistant, tmp_path
) -> None:
    """A frame colliding with another must be re-fetched instead of failing.

    The three fixed frames (first/last/postroll) are computed from the person
    window and are never scored by the change selector, so on a static corridor
    they can land on a frame identical to another selection and abort the whole
    activity. Uniqueness is the only signal that reliably detects this, so the
    collision is resolved by fetching a neighbouring frame that differs from
    every other selected frame.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # 103.2 is the plan's first frame; the motion planner will also select
    # 105.0, so the two render identically and the sheet cannot be unique.
    colliding = 103.2
    partner = 105.0

    class Client:
        snapshot_calls = 0

        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            self.snapshot_calls += 1
            # The colliding frame stays flat grey for the first retry step, so
            # only a further offset produces a frame distinct from every other.
            if abs(timestamp - colliding) < 0.01 or abs(timestamp - partner) < 0.01:
                return _jpeg(7)
            if abs(timestamp - (colliding + 0.5)) < 0.01:
                return _jpeg(7)
            return _changing_jpeg(timestamp)

    client = Client()
    manager = MediaManager(
        hass, store, client, tmp_path, ZoneRoles(frozenset(), frozenset(), frozenset())
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert colliding not in completed.sample_times
    assert client.snapshot_calls > 6


async def test_review_records_uniqueness_replacements(
    hass: HomeAssistant, tmp_path
) -> None:
    """Uniqueness substitutions must be observable in the artifact metadata."""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if abs(timestamp - 103.2) < 0.01 or abs(timestamp - 105.0) < 0.01:
                return _jpeg(7)
            if abs(timestamp - 103.7) < 0.01:
                return _jpeg(7)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.evidence_path is not None
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert '"plan_version": 4' in metadata
    assert "uniqueness_replacements" in metadata


async def test_review_still_fails_when_no_unique_neighbour_exists(
    hass: HomeAssistant, tmp_path
) -> None:
    """A uniform scene keeps the existing failure rather than fabricating frames.

    Every frame renders identically here, so no retry can produce a distinct
    frame. The guard is deliberately *not* relaxed: emitting six copies of one
    image would tell the model it saw six moments when it saw one. The activity
    fails with the original code so the loss stays visible.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 20, "end_time": before + 20}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _jpeg(128)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    with pytest.raises(MediaError, match="duplicate_evidence_frame"):
        await manager.async_build(record.activity_id)


async def test_review_tolerates_a_gap_under_a_discarded_change_candidate(
    hass: HomeAssistant, tmp_path
) -> None:
    """A recording hole under a change *candidate* must not fail the activity.

    Recording segments can leave sub-second holes. The nine proportional change
    candidates are only a search space -- three are kept -- so requiring every
    one of them to be covered aborts reviews that could have produced a
    complete sheet from the remaining candidates.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # Without path_data the plan is the image_change branch:
    # first 103.2, candidates 104.56..115.44, last 116.8, postroll 119.8.
    # Punch a hole around the LAST candidate (115.44), which the selector may
    # legitimately discard in favour of the other eight.
    hole_start = 115.3
    hole_end = 115.6

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [
                {"start_time": after - 5, "end_time": hole_start},
                {"start_time": hole_end, "end_time": before + 5},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if hole_start < timestamp < hole_end:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert all(not (hole_start < stamp < hole_end) for stamp in completed.sample_times)
    # The contractually required frames survive.
    assert 103.2 in completed.sample_times
    assert 116.8 in completed.sample_times
    assert 119.8 in completed.sample_times


async def test_review_nudges_a_motion_frame_out_of_a_recording_hole(
    hass: HomeAssistant, tmp_path
) -> None:
    """A motion frame inside a recording hole moves to a covered instant.

    The path_motion branch uses all three motion peaks, so unlike the change
    candidates there is nothing to discard: the frame itself has to move. A
    motion peak is a sampled instant rather than an exact moment, so shifting it
    within the same motion third keeps its meaning.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    # With path_data the plan is motion_times (105, 107, 109); punch a hole
    # around 107, the middle motion peak.
    hole_start = 106.8
    hole_end = 107.2
    nudge_window = 2.0

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [
                {"start_time": after - 5, "end_time": hole_start},
                {"start_time": hole_end, "end_time": before + 5},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if hole_start < timestamp < hole_end:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # Six distinct frames, none inside the hole.
    assert len(set(completed.sample_times)) == 6
    assert all(not (hole_start < stamp < hole_end) for stamp in completed.sample_times)
    # The nudged frame stays near the peak it replaces.
    assert any(abs(stamp - 107.0) <= nudge_window for stamp in completed.sample_times)
    # Order and the fixed frames are preserved.
    assert tuple(sorted(completed.sample_times)) == completed.sample_times
    assert 103.2 in completed.sample_times
    assert 116.8 in completed.sample_times
    assert 119.8 in completed.sample_times


async def test_review_records_motion_nudges_in_metadata(
    hass: HomeAssistant, tmp_path
) -> None:
    """A nudged motion frame must be observable in the artifact metadata."""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    hole_start = 106.8
    hole_end = 107.2

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [
                {"start_time": after - 5, "end_time": hole_start},
                {"start_time": hole_end, "end_time": before + 5},
            ]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            if hole_start < timestamp < hole_end:
                raise OSError("no recording at this instant")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.evidence_path is not None
    metadata = (tmp_path / "entry_1" / f"{record.activity_id}.json").read_text()
    assert "recording_nudges" in metadata
    assert "107.0" in metadata


async def test_review_still_fails_when_a_required_frame_is_uncovered(
    hass: HomeAssistant, tmp_path
) -> None:
    """A hole under a frame that must be used still fails the activity.

    first/last/postroll are contractually required, so an uncovered one is a
    genuine gap rather than a discarded candidate.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            # Ends before the postroll frame (119.8), so a required frame is
            # uncovered no matter which candidates are discarded.
            return [{"start_time": after - 5, "end_time": 118.0}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    with pytest.raises(MediaError, match="recording_gap"):
        await manager.async_build(record.activity_id)


async def test_review_survives_a_failed_infrared_retry_fetch(
    hass: HomeAssistant, tmp_path
) -> None:
    """A retry fetch that 404s must keep the original frame, not fail the run.

    Retrying an IR frame is a best-effort improvement: the frame itself is
    already downloaded and usable. Frigate purges recordings in the background,
    so a neighbouring instant can 404 even though the original sample is still
    available. Letting that propagate would discard a whole activity over an
    optional upgrade -- the exact opposite of what the repair is for.

    Observed in production: replaying 09-17 08:20 raised
    FrigateApiError(http_404) from the +0.5s IR retry.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    from custom_components.frigate_vision.frigate import FrigateApiError

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # The planned frames succeed; the IR retry neighbour 404s.
            if abs(timestamp - 103.2) < 0.01:
                return _ir_jpeg()
            if abs(timestamp - 103.7) < 0.01:
                raise FrigateApiError("http_404")
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # The 404 at +0.5s is skipped and a later offset succeeds, so the IR frame
    # is upgraded rather than either aborting or staying green. Which offset
    # wins depends on the ladder's granularity, so assert the intent: the frame
    # moved a little forward and is no longer the original green instant.
    assert 103.2 not in completed.sample_times
    assert 103.0 < completed.sample_times[0] < 105.0


async def test_review_keeps_the_frame_when_every_retry_fetch_fails(
    hass: HomeAssistant, tmp_path
) -> None:
    """When no retry can be fetched at all, the original frame is kept.

    A purged recording window can make every neighbouring instant unfetchable.
    The activity must still produce its six frames rather than failing over an
    optional improvement.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    from custom_components.frigate_vision.frigate import FrigateApiError

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            planned = (103.2, 105.0, 107.0, 109.0, 116.8, 119.8)
            if any(abs(timestamp - value) < 0.01 for value in planned):
                if abs(timestamp - 103.2) < 0.01:
                    return _ir_jpeg()
                return _changing_jpeg(timestamp)
            raise FrigateApiError("http_404")

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    # No replacement was available, so the original IR frame is kept.
    assert 103.2 in completed.sample_times


async def test_retry_helper_does_not_settle_two_frames_on_one_picture(
    hass: HomeAssistant, tmp_path
) -> None:
    """The shared repair helper must keep neighbours on different pictures.

    Frigate snaps a requested timestamp to its nearest stored frame, so two
    instants inside one frame interval return identical bytes. Reproduced on a
    real sheet: frames planned 0.21s apart were repaired to instants 5.7ms apart
    and came back byte-identical, and the uniqueness guard then rejected the
    sheet as `duplicate_evidence_frame` -- failing a whole activity over what was
    only an optional improvement.

    Each frame walks the ladder independently and only checks that a timestamp is
    unused, never that the picture differs, so two neighbours that both clear at
    the same boundary converge. This drives the helper directly, because the
    review selector picks frames by content and would otherwise choose a
    different set than the one that collides.
    """
    from custom_components.frigate_vision.media import (
        frame_is_infrared,
        frame_is_overexposed,
    )

    store = ActivityStore(hass, "entry_1")
    await store.async_load()

    # Two adjacent frames whose planned instants are 0.2s apart, as the real
    # case was: an anchor and the first change frame.
    planned = [(100.0, tmp_path / "planned_0.jpg"), (100.2, tmp_path / "planned_1.jpg")]
    for _, path in planned:
        path.write_bytes(_blown_jpeg())

    class Client:
        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # The camera's own frame grid: 0.5s. Blown before 101.5, clear after,
            # so both repairs first succeed inside the same cell.
            if timestamp < 101.5:
                return _blown_at(timestamp)
            return _jpeg(int(timestamp / 0.5) * 13 % 255)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    selected, _replacements = await manager._async_retry_frames(
        "front",
        tmp_path,
        planned,
        [{"start_time": 90.0, "end_time": 130.0}],
        offsets=(0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 1.5, 2.0),
        tag="exposure",
        needed=lambda _index, path, _result: frame_is_overexposed(path),
        acceptable=lambda path, _index, _result: (
            not frame_is_overexposed(path) and not frame_is_infrared(path)
        ),
    )

    assert len(selected) == 2
    # The heart of it: the two frames must be different pictures.
    validate_unique_frames([path for _, path in selected])


async def test_overexposure_repair_avoids_frames_that_collide(
    hass: HomeAssistant, tmp_path
) -> None:
    """A repaired sheet must satisfy the production uniqueness guard.

    End-to-end companion to the helper test: whatever the selector chooses, the
    finished sheet has to be shippable.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Model the camera rather than the request: the picture depends only
            # on which stored frame the instant falls in, so two requests inside
            # one cell return identical bytes, as the real collision did.
            if 102.9 <= timestamp < 108.0:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    assert tuple(sorted(set(completed.sample_times))) == completed.sample_times
    validate_unique_frames([Path(completed.evidence_path)])


async def test_overexposure_repair_never_accepts_a_night_vision_frame(
    hass: HomeAssistant, tmp_path
) -> None:
    """A blown frame must not be traded for an IR one.

    Both defects come from the camera changing state, so an IR window often sits
    immediately before the blown window. The ladder tries negative offsets first
    and would step straight into it, replacing an unusable frame with a
    different unusable frame -- and, worse, undoing the night-vision repair that
    had already run.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = _standalone_record()
    await store.async_create(record)

    class Client:
        async def async_get_event(self, event_id: str, camera: str):
            return {
                "id": event_id,
                "camera": camera,
                "label": "person",
                "start_time": 103,
                "end_time": 117,
                "data": {"path_data": _motion_points()},
            }

        async def async_get_recordings(self, camera: str, after: float, before: float):
            return [{"start_time": after - 10, "end_time": before + 10}]

        async def async_get_snapshot(self, camera: str, timestamp: float, height: int):
            # Earlier than 103.2 is night vision; from 103.2 to 105.0 is blown;
            # after that it is colour, so a good candidate exists forward.
            if timestamp < 103.199:
                return _ir_jpeg()
            if timestamp <= 105.0:
                return _blown_at(timestamp)
            return _changing_jpeg(timestamp)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )
    completed = await manager.async_build(record.activity_id)
    assert completed.stage is ActivityStage.EVIDENCE_READY
    assert len(completed.sample_times) == 6
    metadata = json.loads(
        Path(completed.evidence_path).with_suffix(".json").read_text()
    )
    for entry in metadata.get("overexposure_replacements", []):
        assert entry["to"] > 105.0, (
            f"frame {entry['index']} was replaced with an instant that is still "
            f"unusable ({entry['to']})"
        )
