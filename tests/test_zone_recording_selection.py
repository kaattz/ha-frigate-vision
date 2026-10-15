"""Configured regions must also affect a build without cached preview frames."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import MediaManager
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.store import ActivityStore

DOOR = ((0.1, 0.15), (0.4, 0.15), (0.4, 0.9), (0.1, 0.9))
FIRST = 100.2
LAST = 120.8
POSTROLL = 123.8
TIMES = (
    FIRST,
    *(FIRST + (LAST - FIRST) * i / 10 for i in range(1, 10)),
    LAST,
    POSTROLL,
)
DOOR_TIME = TIMES[7]


class RecordingClient:
    def __init__(self, *, motion=False, nudge=False, infrared_door=False):
        self.motion = motion
        self.nudge = nudge
        self.infrared_door = infrared_door
        self.snapshot_times = []

    async def async_get_event(self, event_id, camera):
        payload = {
            "id": event_id,
            "camera": camera,
            "label": "person",
            "start_time": 100.0,
            "end_time": 121.0,
        }
        if self.motion:
            payload["data"] = {
                "path_data": [[[i / 20, 0.5], 100.0 + i] for i in range(21)]
            }
        return payload

    async def async_get_recordings(self, camera, after, before):
        return [
            {"start_time": 100.4 if self.nudge else after - 1, "end_time": before + 1}
        ]

    async def async_get_snapshot(self, camera, timestamp, height):
        self.snapshot_times.append(timestamp)
        image = Image.new("RGB", (320, 180), (70 + round((timestamp - 100) * 3),) * 3)
        index = min(range(len(TIMES)), key=lambda i: abs(TIMES[i] - timestamp))
        # Larger wall motion wins the old whole-frame ranking. The smaller door
        # motion can enter the sheet only when the region reaches the selector.
        if index in {1, 3, 5}:
            image.paste((235, 235, 235), (215, 75, 315, 175))
        if index == 7:
            image.paste((200, 200, 200), (40, 50, 80, 100))
        if self.infrared_door and timestamp == DOOR_TIME:
            # Green night vision with the same grayscale content: it must still
            # be selected as a door event, then repaired within that event.
            level = 70 + round((timestamp - 100) * 3)
            image = Image.new("RGB", (320, 180), (0, round(level / 0.587), 0))
            image.paste((120, 255, 120), (40, 50, 80, 100))
        output = BytesIO()
        image.save(output, "JPEG")
        return output.getvalue()


async def _build(hass, root, client, *, zones=True):
    entry_id = "entry_zone" if zones else "entry_plain"
    store = ActivityStore(hass, entry_id)
    await store.async_load()
    record = ActivityRecord(
        activity_id=f"review_{entry_id}_front_zone_recordings",
        entry_id=entry_id,
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.SEALED,
        camera="front",
        created_at=100,
        updated_at=121,
        finalization_deadline=121,
        review_ids=("review_1",),
        detection_ids=("event_1",),
    )
    await store.async_create(record)
    manager = MediaManager(
        hass,
        store,
        client,
        root,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
        focus_zones={"home_door": DOOR} if zones else {},
        person_highlight=False,
    )
    result = await manager.async_build(record.activity_id)
    metadata = json.loads(Path(result.evidence_path).with_suffix(".json").read_text())
    return manager, store, record, result, metadata


async def test_recording_regions_change_the_final_sheet(hass, tmp_path):
    focused = RecordingClient()
    plain = RecordingClient()
    _, _, _, with_zone, metadata = await _build(hass, tmp_path / "zone", focused)
    _, _, _, without_zone, _ = await _build(
        hass, tmp_path / "plain", plain, zones=False
    )
    assert DOOR_TIME in with_zone.sample_times, "已下载的门口变化被旧全图排名挤掉了"
    assert DOOR_TIME not in without_zone.sample_times
    assert with_zone.sample_times != without_zone.sample_times
    assert metadata["candidate_source"] == "recordings"
    assert with_zone.selection_source == metadata["selection_source"] == "image_change"
    assert len(focused.snapshot_times) == 12
    assert plain.snapshot_times[:12] == pytest.approx(TIMES)
    assert (
        len(plain.snapshot_times) == 15
    )  # The unchanged coverage pass adds three cells.
    assert len(with_zone.sample_times) in {6, 9}


async def test_motion_fallback_uses_twelve_candidates_only_with_regions(hass, tmp_path):
    focused = RecordingClient(motion=True)
    plain = RecordingClient(motion=True)
    _, _, _, result, metadata = await _build(hass, tmp_path / "zone", focused)
    _, _, _, old_result, _ = await _build(hass, tmp_path / "plain", plain, zones=False)
    assert focused.snapshot_times == pytest.approx(TIMES)
    assert len(plain.snapshot_times) == 6
    assert old_result.selection_source == "path_motion"
    assert result.selection_source == metadata["selection_source"] == "image_change"
    assert DOOR_TIME in result.sample_times
    assert len(focused.snapshot_times) <= 18


async def test_focused_recording_selection_receives_motion_phases(
    hass, tmp_path, monkeypatch
):
    import custom_components.frigate_vision.media as media

    original = media.choose_cells
    seen = []

    def choose(scored, segments, *, plan, cells, max_cells):
        seen.append(plan.motion_times)
        return original(scored, segments, plan=plan, cells=cells, max_cells=max_cells)

    monkeypatch.setattr(media, "choose_cells", choose)
    client = RecordingClient(motion=True)
    await _build(hass, tmp_path, client)
    assert seen and seen[0], "focus zones不能让预览/录像选择器丢弃已有轨迹"
    assert client.snapshot_times == pytest.approx(TIMES)


async def test_recording_selection_reuses_nudged_anchor(hass, tmp_path):
    client = RecordingClient(motion=True, nudge=True)
    _, _, _, result, metadata = await _build(hass, tmp_path, client)
    assert result.sample_times[0] == pytest.approx(FIRST + 0.5)
    assert FIRST not in client.snapshot_times
    assert metadata["recording_nudges"][0]["role"] == "first"
    assert DOOR_TIME in result.sample_times
    assert len(client.snapshot_times) <= 18


async def test_recording_zone_artifact_is_reused_from_sealed_stage(hass, tmp_path):
    client = RecordingClient(motion=True)
    manager, store, record, result, _ = await _build(hass, tmp_path, client)
    before = list(client.snapshot_times)
    # Simulate an artifact written just before the Store checkpoint succeeded.
    store._activities[record.activity_id] = replace(record)
    restored = await manager.async_build(record.activity_id)
    assert restored.sample_times == result.sample_times
    assert restored.selection_source == "image_change"
    assert client.snapshot_times == before


async def test_infrared_repair_keeps_the_selected_door_event(hass, tmp_path):
    client = RecordingClient(motion=True, infrared_door=True)
    _, _, _, result, metadata = await _build(hass, tmp_path, client)
    door_replacements = [
        item
        for item in metadata["infrared_replacements"]
        if item["from"] == pytest.approx(DOOR_TIME)
    ]
    assert door_replacements, "The selected night-vision door frame must be repaired."
    repaired = door_replacements[0]["to"]
    assert repaired in result.sample_times
    assert (TIMES[6] + DOOR_TIME) / 2 < repaired < (DOOR_TIME + TIMES[8]) / 2
    assert len(client.snapshot_times) <= 18


async def test_coverage_reuses_downloaded_candidates_and_keeps_door(hass, tmp_path):
    client = RecordingClient()
    manager, _, record, _, _ = await _build(hass, tmp_path, client)
    paths = {}
    for i, moment in enumerate((FIRST, 105.0, 110.0, DOOR_TIME, 150.0, 160.0, 162.8)):
        path = tmp_path / f"cached_{i}.jpg"
        path.write_bytes(await client.async_get_snapshot("front", moment, 360))
        paths[moment] = path
    selected = [(t, paths[t]) for t in (FIRST, 105.0, 110.0, DOOR_TIME, 160.0, 162.8)]
    from custom_components.frigate_vision.media import EvidencePlan

    plan = EvidencePlan(
        mode="review_six", first_time=FIRST, last_time=160.0, postroll_time=162.8
    )
    before = Counter(client.snapshot_times)
    repaired = await manager._async_enforce_coverage(
        record,
        tmp_path,
        selected,
        plan,
        [{"start_time": 90.0, "end_time": 170.0}],
        list(paths),
        cached=paths,
        protect=(DOOR_TIME,),
    )
    assert DOOR_TIME in dict(repaired)
    assert Counter(client.snapshot_times) == before
    assert 150.0 in dict(repaired)


async def test_coverage_keeps_an_already_selected_subthreshold_zone(hass, tmp_path):
    """A weak zone signal still wins selection priority and must survive relocation."""
    from custom_components.frigate_vision.changes import ScoredFrame, Segment
    from custom_components.frigate_vision.media import (
        EvidencePlan,
        _selected_zone_times,
    )

    client = RecordingClient()
    manager, _, record, _, _ = await _build(hass, tmp_path, client)
    chosen = (FIRST, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 160.0, 162.8)
    weak_door = 105.0
    scored = [
        ScoredFrame(
            timestamp=moment,
            sudden=0.7,
            cumulative=0.0,
            global_shift=0.0,
            zones={"home_door": 0.1 if moment == weak_door else 0.0},
        )
        for moment in chosen
    ]
    segments = [
        Segment(start=weak_door, end=weak_door, kind="change", score=0.7, indices=(5,))
    ]
    assert 0.0 < scored[5].best_zone < 0.25
    protected = _selected_zone_times(scored, segments, chosen)
    moments = (*chosen, 115.0, 125.0, 135.0, 145.0, 150.0, 155.0)
    paths = {}
    for index, moment in enumerate(moments):
        path = tmp_path / f"weak_cached_{index}.jpg"
        path.write_bytes(await client.async_get_snapshot("front", moment, 360))
        paths[moment] = path
    plan = EvidencePlan(
        mode="review_six", first_time=FIRST, last_time=160.0, postroll_time=162.8
    )
    before = Counter(client.snapshot_times)
    repaired = await manager._async_enforce_coverage(
        record,
        tmp_path,
        [(moment, paths[moment]) for moment in chosen],
        plan,
        [{"start_time": 90.0, "end_time": 170.0}],
        list(paths),
        cached=paths,
        protect=protected,
    )
    assert weak_door in dict(repaired), "Coverage erased the selected weak door event."
    assert tuple(moment for moment, _ in repaired) != chosen
    assert Counter(client.snapshot_times) == before
