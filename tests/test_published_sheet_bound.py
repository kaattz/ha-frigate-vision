"""The bound, asserted on a finished build with only pre-existing API.

Why this is its own file
------------------------
`tests/test_coverage_guarantee.py` imports `enforce_coverage`, `MAX_CELLS` and
`worst_unobserved_gap` at module scope. Against the pre-change code that fails at
COLLECTION with an ImportError, which proves the module changed but says nothing
about whether the published sheet got better -- a weak kind of failure.

This file deliberately imports only names that existed before the fix, and works
out the worst unobserved stretch with plain arithmetic. So on the old code it runs
and fails for the real reason: the build produced a sheet with a hole wider than
the bound the module itself declares. It is the behavioural regression test for
the defect the household reported twice.

The fixture reproduces the 六宫格 report's shape: a change selector that aims all
three of its picks at the opening seconds and leaves a hole running to the person's
last frame. Measured on this fixture before the repair existed: 13.72 s against a
12 s bound.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.correlation import ZoneRoles
from custom_components.frigate_vision.media import MAX_SAMPLE_GAP, MediaManager
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.store import ActivityStore


def _jpeg(colour: int) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (640, 360), (colour, colour, colour)).save(buffer, "JPEG")
    return buffer.getvalue()


def _worst_unobserved_gap(samples: list[float], start: float, end: float) -> float:
    """The widest stretch of [start, end] holding no cell, with no helpers.

    Both window EDGES are boundaries as well as the cells inside, because a
    person can walk in before the detector's first sample -- which is exactly what
    the reported 六宫格 shows.

    Sorted here rather than assumed, so a caller cannot get a wrong (too small)
    answer by handing this an unordered sheet: the boundaries only partition the
    span correctly in order.
    """
    bounds = [start, *(sorted(v for v in samples if start < v < end)), end]
    return max(
        right - left for left, right in zip(bounds, bounds[1:], strict=False)
    )


async def test_a_built_sheet_has_no_hole_wider_than_the_bound(
    hass: HomeAssistant, tmp_path: Path
) -> None:
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
            return _jpeg(int(timestamp) % 255)

    manager = MediaManager(
        hass,
        store,
        Client(),
        tmp_path,
        ZoneRoles(frozenset({"near"}), frozenset({"mid"}), frozenset({"far"})),
    )
    built = await manager.async_build(record.activity_id)
    samples = sorted(built.sample_times)
    assert len(samples) >= 6, samples
    assert len(samples) % 3 == 0, f"格子数必须是三的倍数：{len(samples)}"

    # The person's span is the event's own 101..121 window with the planner's
    # 0.2 s edge. The last stored cell is the postroll scene, which sits outside
    # that window on purpose and therefore closes nothing.
    start = 101 + min(0.2, 20 * 0.05)
    end = 121 - min(0.2, 20 * 0.05)
    worst = _worst_unobserved_gap(samples, start, end)
    assert worst <= MAX_SAMPLE_GAP, (
        f"发布的证据图最差盲区 {worst:.2f}s，超过它自己声明的 {MAX_SAMPLE_GAP}s "
        f"上限；格子={[round(t, 2) for t in samples]}"
    )


def test_the_caller_protects_the_last_frame() -> None:
    """The cell the layout calls 末帧 must stay in the caller's protected set.

    Why this is a structural assertion rather than a behavioural one, and why that
    is the right call here. `enforce_coverage` honours whatever `protect` list it
    is handed, and `test_the_motion_route_keeps_the_last_frame_the_layout_names`
    pins that. Neither pins the CALLER -- and measured: reverting
    `_async_enforce_coverage` to its pre-fix two-cell tuple leaves the FULL suite
    green (1055 passed). The regression can therefore come back silently.

    A behavioural end-to-end test was written first and did NOT work, which is
    worth recording so nobody re-tries it. `last_time` can only be dropped while
    the SPREADING stage runs, and that stage needs the ADD stage to have failed on
    a pool richer than the sheet. Measured across spans 20-400 s and pools of
    6/13/40 instants:

      pool == sheet          ADD can add nothing and spread cannot improve either,
                             so the sheet is returned untouched either way
      dense grid (>= 9)      ADD reaches the bound, so spread never runs
      rich pool, span >= 61  spread runs and, unprotected, drops `last_time`

    No fixture reachable through `MediaManager.async_build` reliably lands in the
    third case: the image-change route's pool is a dense nine-point grid, and the
    motion route's `change_candidates` is empty. So the property is asserted where
    it actually lives -- the call site -- in the same spirit as
    `test_the_coverage_guard_is_wired_to_the_semantic_requirement`, which exists
    for exactly the same reason: a mutation the behavioural tests cannot see.
    """
    import inspect

    from custom_components.frigate_vision import media as media_module

    source = inspect.getsource(media_module.MediaManager._async_enforce_coverage)
    assert "plan.last_time" in source, (
        "the coverage repair no longer protects plan.last_time; measured, the "
        "spreading stage then drops the 末帧 in 71.6% of motion-shaped sheets"
    )
    assert "plan.first_time" in source and "plan.postroll_time" in source, (
        "the coverage repair must keep protecting the person's first frame and "
        "the emptied scene as well"
    )
    # The pre-fix shape must not come back.
    assert "protect=(plan.first_time, plan.postroll_time)" not in source, (
        "the two-cell protect tuple is back: `plan.last_time` is unprotected again"
    )
