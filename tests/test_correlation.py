from __future__ import annotations

from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.correlation import (
    CorrelationEngine,
    CorrelationError,
    ZoneRoles,
    anchor_sequence,
    find_review_owner,
    infer_direction,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    IngressKind,
    IngressMessage,
    ProcessingMode,
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
        source=ActivitySource.DOOR_CYCLE,
        stage=stage,
        processing_mode=ProcessingMode.OBSERVE,
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
        processing_mode=ProcessingMode.OBSERVE,
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


def test_overlapping_roles_do_not_create_direction_anchor() -> None:
    roles = ZoneRoles(
        near=frozenset({"near"}),
        transition=frozenset({"mid"}),
        far=frozenset({"far"}),
    )
    anchors = anchor_sequence(
        [
            (1, ("near",)),
            (2, ("near", "mid")),
            (3, ("mid",)),
            (4, ("far",)),
        ],
        roles,
    )
    assert [anchor.role for anchor in anchors] == ["near", "transition", "far"]
    assert infer_direction(anchors) == "outbound"


def test_direction_requires_stable_start_and_end() -> None:
    roles = ZoneRoles(
        near=frozenset({"near"}), transition=frozenset(), far=frozenset({"far"})
    )
    assert (
        infer_direction(anchor_sequence([(1, ("far",)), (2, ("near",))], roles))
        == "inbound"
    )
    assert (
        infer_direction(
            anchor_sequence([(1, ("near",)), (2, ("far",)), (3, ("near",))], roles)
        )
        == "roundtrip"
    )
    assert infer_direction(anchor_sequence([(1, ("mid",))], roles)) == "ambiguous"


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
        processing_mode=ProcessingMode.OBSERVE,
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
        processing_mode=ProcessingMode.OBSERVE,
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


async def test_pre_open_event_is_buffered_then_replayed_into_door_cycle(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
    )
    message = IngressMessage(
        kind=IngressKind.FRIGATE_EVENT,
        entry_id="entry_1",
        source_id="event_1",
        event_id="event_1",
        event_type="new",
        occurred_at=95,
        camera="front",
        current_zones=("near",),
    )
    assert await engine.async_handle(message) is None
    await store.async_create(
        replace(
            _activity("activity_1", ()),
            stage=ActivityStage.COLLECTING,
            created_at=100,
            updated_at=100,
        )
    )
    replayed = await engine.async_replay_for_activity("activity_1")
    assert [record.activity_id for record in replayed] == ["activity_1"]
    assert store.get("activity_1").detection_ids == ("event_1",)  # type: ignore[union-attr]
    assert store.buffered_ingress() == ()


async def test_expired_unowned_event_and_doorbell_are_pruned(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        clock=lambda: 100,
    )
    for message in (
        IngressMessage(
            kind=IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="event_1",
            event_id="event_1",
            event_type="new",
            occurred_at=95,
            camera="front",
        ),
        IngressMessage(
            kind=IngressKind.DOORBELL,
            entry_id="entry_1",
            source_id="doorbell_1",
            occurred_at=95,
            camera="front",
        ),
    ):
        assert await engine.async_handle(message) is None
    assert len(store.buffered_ingress()) == 2
    assert await engine.async_settle_due(240) == ()
    assert store.buffered_ingress() == ()


async def test_pre_open_events_keep_each_detection_track_separate(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        clock=lambda: 100,
    )
    for event_id, occurred_at, zone in (
        ("person_a", 95, "near"),
        ("person_b", 96, "far"),
    ):
        assert (
            await engine.async_handle(
                IngressMessage(
                    kind=IngressKind.FRIGATE_EVENT,
                    entry_id="entry_1",
                    source_id=event_id,
                    event_id=event_id,
                    event_type="new",
                    occurred_at=occurred_at,
                    camera="front",
                    current_zones=(zone,),
                )
            )
            is None
        )
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.DOOR_CYCLE,
            stage=ActivityStage.COLLECTING,
            processing_mode=ProcessingMode.OBSERVE,
            created_at=100,
            updated_at=100,
            camera="front",
        )
    )
    await engine.async_replay_for_activity("activity_1")
    record = store.get("activity_1")
    assert record is not None
    assert record.zone_updates == ()
    assert record.detection_zone_updates == (
        ("person_a", 95, ("near",)),
        ("person_b", 96, ("far",)),
    )


async def test_late_event_attaching_buffered_review_also_attaches_doorbell(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        clock=lambda: 100,
    )
    doorbell = IngressMessage(
        kind=IngressKind.DOORBELL,
        entry_id="entry_1",
        source_id="doorbell_1",
        occurred_at=70,
        camera="front",
    )
    review = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_1",
        review_id="review_1",
        started_at=65,
        occurred_at=90,
        camera="front",
        detection_ids=("event_1",),
    )
    assert await engine.async_handle(doorbell) is None
    assert await engine.async_handle(review) is None
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.DOOR_CYCLE,
            stage=ActivityStage.COLLECTING,
            processing_mode=ProcessingMode.OBSERVE,
            created_at=80,
            updated_at=80,
            camera="front",
        )
    )
    attached = await engine.async_handle(
        IngressMessage(
            kind=IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="event_1",
            event_id="event_1",
            event_type="new",
            occurred_at=80,
            camera="front",
            current_zones=("near",),
        )
    )
    assert attached is not None
    record = store.get("activity_1")
    assert record is not None
    assert record.review_ids == ("review_1",)
    assert record.doorbell_at == 70


async def test_failed_review_delete_failure_retries_as_terminal_cleanup(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
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
    doorbell = IngressMessage(
        kind=IngressKind.DOORBELL,
        entry_id="entry_1",
        source_id="bell_1",
        occurred_at=85,
        camera="front",
    )
    assert await engine.async_handle(review) is None
    assert await engine.async_handle(doorbell) is None
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
    assert settled == (failed,)
    assert store.buffered_ingress() == ()


async def test_unowned_review_waits_then_settles_once(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
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


async def test_buffered_review_keeps_original_processing_mode(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    shadow = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.SHADOW,
        clock=lambda: 100,
        review_settle_seconds=10,
    )
    review = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_mode",
        review_id="review_mode",
        started_at=80,
        occurred_at=90,
        camera="front",
        detection_ids=("event_1",),
    )
    assert await shadow.async_handle(review) is None
    live = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.LIVE,
        clock=lambda: 200,
    )
    settled = await live.async_settle_due(110)
    assert settled[0].processing_mode is ProcessingMode.SHADOW


async def test_ambiguous_buffered_review_fails_without_blocking_next_review(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
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


async def test_doorbell_without_active_cycle_is_buffered_and_attached_to_review(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        clock=lambda: 100,
        review_settle_seconds=10,
    )
    assert (
        await engine.async_handle(
            IngressMessage(
                kind=IngressKind.DOORBELL,
                entry_id="entry_1",
                source_id="doorbell_1",
                occurred_at=80,
                camera="front",
            )
        )
        is None
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
    settled = await engine.async_settle_due(110)
    assert len(settled) == 1
    assert settled[0].doorbell_at == 80
    assert store.buffered_ingress() == ()


async def test_multiple_doorbells_are_all_attached_without_blocking_review(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        clock=lambda: 100,
        review_settle_seconds=10,
    )
    for source_id, occurred_at in (("bell_1", 80), ("bell_2", 82)):
        assert (
            await engine.async_handle(
                IngressMessage(
                    kind=IngressKind.DOORBELL,
                    entry_id="entry_1",
                    source_id=source_id,
                    occurred_at=occurred_at,
                    camera="front",
                )
            )
            is None
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
    settled = await engine.async_settle_due(110)
    assert len(settled) == 1
    assert settled[0].doorbell_times == (80, 82)
    assert settled[0].doorbell_at == 80
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


async def test_review_over_failed_door_cycle_creates_standalone_activity(
    hass: HomeAssistant,
) -> None:
    """A Review over only terminal activities becomes its own standalone activity.

    A Review is held for the settle window before it is turned into an activity,
    so that a door cycle still in progress can claim it. Only once the window
    closes does an unclaimed Review become standalone -- which is also what lets
    a Review that overlaps a FAILED door cycle avoid writing to it, since the
    Store rejects writes to terminal records.

    The clock is injected because the deadline is derived from it. With the
    default wall clock, a test using synthetic timestamps schedules the deadline
    decades away and no settle ever comes due.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        replace(
            _activity("door_failed", ("event_1",)),
            stage=ActivityStage.FAILED,
            error_code="direction_ambiguous",
        )
    )
    engine = CorrelationEngine(
        store,
        entry_id="entry_1",
        camera="front",
        processing_mode=ProcessingMode.OBSERVE,
        clock=lambda: 0.0,
    )
    review = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_1",
        review_id="review_1",
        started_at=2,
        occurred_at=3,
        camera="front",
        detection_ids=("event_1",),
    )

    # Held, not created: the failed cycle is terminal, so it cannot own this.
    assert await engine.async_handle(review) is None
    buffered = store.buffered_ingress()
    assert len(buffered) == 1

    settled = await engine.async_settle_due(buffered[0].settle_after)
    assert len(settled) == 1
    result = settled[0]
    assert result.source is ActivitySource.STANDALONE_REVIEW
    assert result.stage is ActivityStage.SEALED
    assert result.review_ids == ("review_1",)
    assert store.buffered_ingress() == ()
    # The terminal cycle is untouched.
    failed = store.get("door_failed")
    assert failed is not None and failed.stage is ActivityStage.FAILED


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
        processing_mode=ProcessingMode.OBSERVE,
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
        processing_mode=ProcessingMode.OBSERVE,
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
        processing_mode=ProcessingMode.OBSERVE,
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
