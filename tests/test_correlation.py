from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.correlation import (
    CorrelationEngine,
    CorrelationError,
    find_review_owner,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    IngressKind,
    IngressMessage,
)
from custom_components.frigate_vision.runtime import (
    RecoveryAction,
    build_recovery_work,
)
from custom_components.frigate_vision.store import ActivityStore


def _activity(
    activity_id: str,
    detections: tuple[str, ...],
    *,
    stage: ActivityStage = ActivityStage.SEALED,
    error_code: str | None = None,
) -> ActivityRecord:
    return ActivityRecord(
        activity_id=activity_id,
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=stage,
        created_at=1,
        updated_at=2,
        camera="front",
        detection_ids=detections,
        error_code=error_code,
    )


def test_review_owner_uses_detection_ids_not_nearest_time() -> None:
    first = _activity("activity_1", ("event_1",))
    second = replace(
        _activity("activity_2", ("event_2",)), created_at=100, updated_at=100
    )
    assert find_review_owner([first, second], {"event_1"}, "front") == first
    assert find_review_owner([first, second], {"unknown"}, "front") is None
    with pytest.raises(CorrelationError, match="ambiguous_review_ownership"):
        find_review_owner(
            [first, replace(second, detection_ids=("event_1",))],
            {"event_1"},
            "front",
        )


async def test_immediate_ambiguous_review_creates_stable_failed_activity(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    for activity_id in ("door_1", "door_2"):
        await store.async_create(_activity(activity_id, ("shared",)))
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
    )
    failed = await engine.async_handle(
        IngressMessage(
            kind=IngressKind.FRIGATE_REVIEW,
            entry_id="entry_1",
            source_id="review_ambiguous",
            review_id="review_ambiguous",
            started_at=2,
            occurred_at=3,
            camera="front",
            detection_ids=("shared",),
        )
    )
    assert failed is not None
    assert failed.stage is ActivityStage.FAILED
    assert failed.error_code == "ambiguous_review_ownership"
    assert store.get(failed.activity_id) == failed


async def test_engine_attaches_detection_and_review_once(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    activity = replace(
        _activity("activity_1", ()),
        association_deadline=11,
        finalization_deadline=121,
    )
    await store.async_create(activity)
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
    )
    await engine.async_handle(
        IngressMessage(
            kind=IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="event_1",
            event_id="event_1",
            occurred_at=2,
            camera="front",
            current_zones=("near",),
        )
    )
    review = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_1",
        review_id="review_1",
        occurred_at=3,
        camera="front",
        detection_ids=("event_1",),
    )
    first = await engine.async_handle(review)
    second = await engine.async_handle(review)
    assert first.activity_id == "activity_1"
    assert second == first
    assert first.detection_ids == ("event_1",)
    assert first.review_ids == ("review_1",)
    assert first.detection_zone_updates == (("event_1", 2, ("near",)),)


async def test_engine_records_each_event_box_against_its_timestamp(
    hass: HomeAssistant,
) -> None:
    """每个事件消息的 box 都要落到活动上，和时间戳绑在一起。

    Frigate 在目标移动时反复发 `update`，每条带一个当刻的 box。这里要的就是那串
    「逐时刻的 box」，后续才能挑出 box 面积最大的那一帧。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        replace(
            _activity("activity_1", ()),
            association_deadline=11,
            finalization_deadline=121,
        )
    )
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
    )
    for occurred_at, box in (
        (2, (0.10, 0.20, 0.30, 0.40)),
        (3, (0.25, 0.44, 0.23, 0.52)),
    ):
        await engine.async_handle(
            IngressMessage(
                kind=IngressKind.FRIGATE_EVENT,
                entry_id="entry_1",
                source_id="event_1",
                event_id="event_1",
                event_type="update",
                occurred_at=occurred_at,
                camera="front",
                current_zones=("near",),
                box=box,
            )
        )
    record = store.get("activity_1")
    assert record is not None
    assert record.box_updates == (
        (2, (0.10, 0.20, 0.30, 0.40)),
        (3, (0.25, 0.44, 0.23, 0.52)),
    )


async def test_engine_records_no_box_when_the_event_has_none(
    hass: HomeAssistant,
) -> None:
    """没有 box 的事件不留空洞：序列保持为空，而不是写入一个占位。"""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        replace(
            _activity("activity_1", ()),
            association_deadline=11,
            finalization_deadline=121,
        )
    )
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
    )
    await engine.async_handle(
        IngressMessage(
            kind=IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="event_1",
            event_id="event_1",
            event_type="update",
            occurred_at=2,
            camera="front",
            current_zones=("near",),
        )
    )
    record = store.get("activity_1")
    assert record is not None
    assert record.box_updates == ()


async def test_new_detection_respects_association_deadline_but_existing_continues(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = replace(
        _activity("activity_1", ("existing",)),
        association_deadline=10,
        finalization_deadline=120,
    )
    await store.async_create(record)
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
    )
    with pytest.raises(CorrelationError, match="event_owner_missing"):
        await engine.async_handle(
            IngressMessage(
                kind=IngressKind.FRIGATE_EVENT,
                entry_id="entry_1",
                source_id="new",
                event_id="new",
                event_type="new",
                occurred_at=20,
                camera="front",
            )
        )
    continued = await engine.async_handle(
        IngressMessage(
            kind=IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="existing",
            event_id="existing",
            event_type="update",
            occurred_at=20,
            camera="front",
        )
    )
    assert continued.activity_id == "activity_1"


async def test_failed_review_delete_failure_retries_as_terminal_cleanup(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        clock=lambda: 100,
        review_settle_seconds=10,
    )
    review = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_ambiguous",
        review_id="review_ambiguous",
        started_at=80,
        occurred_at=90,
        camera="front",
        detection_ids=("shared",),
    )
    # A second buffered review of the same ambiguous detection: both entries
    # must survive a failed settle and be cleaned up together once it succeeds.
    # (This scenario once used a buffered doorbell for the second entry; the
    # doorbell buffer is gone with the door cycle, but the cleanup guarantee --
    # a failed delete must not silently drop buffered work -- is unchanged.)
    second_review = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_ambiguous_2",
        review_id="review_ambiguous_2",
        started_at=80,
        occurred_at=90,
        camera="front",
        detection_ids=("shared",),
    )
    assert await engine.async_handle(review) is None
    assert await engine.async_handle(second_review) is None
    for activity_id in ("door_1", "door_2"):
        await store.async_create(_activity(activity_id, ("shared",)))

    store._store.async_save = AsyncMock(  # type: ignore[method-assign]
        side_effect=[None, OSError("delete failed")]
    )
    with pytest.raises(OSError, match="delete failed"):
        await engine.async_settle_due(110)
    failed = store.get("review_entry_1_front_review_ambiguous")
    assert failed is not None and failed.stage is ActivityStage.FAILED
    assert len(store.buffered_ingress()) == 2

    store._store.async_save = AsyncMock()  # type: ignore[method-assign]
    settled = await engine.async_settle_due(110)
    # Both buffered reviews settle as failed activities now -- the doorbell
    # entry once skipped this by attaching as context, but it is gone.
    assert sorted(item.activity_id for item in settled) == [
        "review_entry_1_front_review_ambiguous",
        "review_entry_1_front_review_ambiguous_2",
    ]
    assert all(item.stage is ActivityStage.FAILED for item in settled)
    assert store.buffered_ingress() == ()


async def test_unowned_review_waits_then_settles_once(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        clock=lambda: 100,
        review_settle_seconds=10,
    )
    review = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_1",
        review_id="review_1",
        started_at=70,
        occurred_at=90,
        camera="front",
        detection_ids=("event_1",),
    )
    assert await engine.async_handle(review) is None
    assert await engine.async_settle_due(109) == ()
    settled = await engine.async_settle_due(110)
    assert len(settled) == 1
    assert settled[0].source is ActivitySource.STANDALONE_REVIEW
    assert settled[0].finalization_deadline == 90
    recovered = await store.async_recover(200)
    assert [
        (item.activity_id, item.action, item.run_at)
        for item in build_recovery_work(recovered, now=200)
    ] == [
        (
            "review_entry_1_front_review_1",
            RecoveryAction.FINALIZE_SEALED,
            90,
        )
    ]
    assert await engine.async_settle_due(120) == ()


async def test_ambiguous_buffered_review_fails_without_blocking_next_review(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        clock=lambda: 100,
        review_settle_seconds=10,
    )
    ambiguous = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_ambiguous",
        review_id="review_ambiguous",
        started_at=80,
        occurred_at=90,
        camera="front",
        detection_ids=("shared",),
    )
    normal = replace(
        ambiguous,
        source_id="review_normal",
        review_id="review_normal",
        detection_ids=("unowned",),
    )
    assert await engine.async_handle(ambiguous) is None
    assert await engine.async_handle(normal) is None
    for activity_id in ("door_1", "door_2"):
        await store.async_create(
            replace(
                _activity(activity_id, ("shared",)),
                association_deadline=10,
                finalization_deadline=120,
            )
        )
    settled = await engine.async_settle_due(110)
    failed = store.get("review_entry_1_front_review_ambiguous")
    assert failed is not None
    assert failed.stage is ActivityStage.FAILED
    assert failed.error_code == "ambiguous_review_ownership"
    assert any(record.activity_id.endswith("review_normal") for record in settled)
    assert store.buffered_ingress() == ()


def test_terminal_activity_is_not_a_review_owner() -> None:
    failed = _activity(
        "door_failed",
        ("event_1",),
        stage=ActivityStage.FAILED,
        error_code="direction_ambiguous",
    )
    completed = _activity("door_completed", ("event_2",))
    completed = replace(completed, stage=ActivityStage.COMPLETED)

    assert find_review_owner([failed], {"event_1"}, "front") is None
    assert find_review_owner([completed], {"event_2"}, "front") is None
    mixed = find_review_owner([failed, completed], {"event_1", "event_2"}, "front")
    assert mixed is None


async def test_short_review_is_dropped_before_any_work(hass: HomeAssistant) -> None:
    """A brief pass must not become an activity.

    The user asked for short events to stop producing notifications. Dropping
    them at correlation means no evidence is fetched and no vision call is made,
    so a passer-by costs nothing.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        clock=lambda: 0.0,
        min_review_seconds=10,
    )
    brief = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_short",
        review_id="review_short",
        started_at=100.0,
        occurred_at=105.0,  # 5 seconds
        camera="front",
        detection_ids=("event_1",),
    )
    assert await engine.async_handle(brief) is None
    # Nothing is held either: a dropped review must not linger and settle later.
    assert store.buffered_ingress() == ()
    assert all("review_short" not in r.review_ids for r in store.all())


async def test_review_at_the_threshold_is_kept(hass: HomeAssistant) -> None:
    """The boundary itself must be kept, so the setting means what it says."""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        clock=lambda: 0.0,
        min_review_seconds=10,
    )
    exactly = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_exact",
        review_id="review_exact",
        started_at=100.0,
        occurred_at=110.0,  # exactly 10 seconds
        camera="front",
        detection_ids=("event_1",),
    )
    await engine.async_handle(exactly)
    assert len(store.buffered_ingress()) == 1


async def test_minimum_of_zero_disables_the_filter(hass: HomeAssistant) -> None:
    """Zero must mean "no minimum", not "drop everything".

    The default for existing entries is 0, so getting this backwards would
    silently stop every notification after an upgrade.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        clock=lambda: 0.0,
        min_review_seconds=0,
    )
    instantaneous = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_zero",
        review_id="review_zero",
        started_at=100.0,
        occurred_at=100.0,
        camera="front",
        detection_ids=("event_1",),
    )
    await engine.async_handle(instantaneous)
    assert len(store.buffered_ingress()) == 1
