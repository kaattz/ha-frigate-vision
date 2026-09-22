from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    IngressKind,
    IngressMessage,
    ModelValidationError,
    ProcessingMode,
    analysis_key,
)
from custom_components.frigate_vision.store import (
    ActivityStore,
    StoreConflictError,
)


def _record() -> ActivityRecord:
    return ActivityRecord(
        activity_id="activity_1",
        entry_id="entry_1",
        source=ActivitySource.DOOR_CYCLE,
        stage=ActivityStage.COLLECTING,
        processing_mode=ProcessingMode.OBSERVE,
        created_at=100,
        updated_at=100,
        camera="front",
    )


async def test_store_compare_and_set_and_terminal_guard(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    sealed = await store.async_transition(
        "activity_1", ActivityStage.COLLECTING, ActivityStage.SEALED, updated_at=110
    )
    assert sealed.stage is ActivityStage.SEALED
    failed = await store.async_transition(
        "activity_1",
        ActivityStage.SEALED,
        ActivityStage.FAILED,
        updated_at=120,
        error_code="evidence_incomplete",
    )
    assert failed.error_code == "evidence_incomplete"
    with pytest.raises(StoreConflictError, match="terminal_activity"):
        await store.async_transition(
            "activity_1", ActivityStage.FAILED, ActivityStage.COLLECTING, updated_at=130
        )


async def test_store_same_identity_is_idempotent(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    first = await store.async_create(_record())
    second = await store.async_create(_record())
    assert first == second


async def test_recovery_preserves_conflicting_cycles_as_failures(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    await store.async_create(replace(_record(), activity_id="activity_2"))
    recovered = await store.async_recover(200)
    assert len(recovered) == 2
    assert all(item.stage is ActivityStage.FAILED for item in recovered)
    assert all(item.error_code == "conflicting_door_cycles" for item in recovered)
    assert len(store.all()) == 2


async def test_duplicate_original_message_after_transition_returns_current_state(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    original = _record()
    await store.async_create(original)
    await store.async_transition(
        "activity_1", ActivityStage.COLLECTING, ActivityStage.SEALED, updated_at=110
    )
    replay = await store.async_create(original)
    assert replay.stage is ActivityStage.SEALED


async def test_record_is_immutable() -> None:
    record = _record()
    with pytest.raises(FrozenInstanceError):
        record.stage = ActivityStage.FAILED  # type: ignore[misc]


async def test_store_rejects_same_id_with_different_identity(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    changed = replace(_record(), camera="other")
    with pytest.raises(StoreConflictError, match="identity_conflict"):
        await store.async_create(changed)


async def test_store_rejects_record_for_another_entry(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    with pytest.raises(StoreConflictError, match="entry_mismatch"):
        await store.async_create(replace(_record(), entry_id="entry_2"))


async def test_failed_save_does_not_commit_memory(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    store._store.async_save = AsyncMock(side_effect=OSError("disk full"))  # type: ignore[method-assign]
    with pytest.raises(OSError, match="disk full"):
        await store.async_create(_record())
    assert store.get("activity_1") is None


async def test_failed_transition_save_keeps_previous_stage(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    store._store.async_save = AsyncMock(side_effect=OSError("disk full"))  # type: ignore[method-assign]
    with pytest.raises(OSError, match="disk full"):
        await store.async_transition(
            "activity_1", ActivityStage.COLLECTING, ActivityStage.SEALED, updated_at=110
        )
    assert store.get("activity_1").stage is ActivityStage.COLLECTING  # type: ignore[union-attr]


async def test_recovery_fails_uncertain_external_side_effects(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = replace(_record(), stage=ActivityStage.ANALYSIS_STARTED)
    await store.async_create(record)
    recovered = await store.async_recover(200)
    assert recovered[0].stage is ActivityStage.FAILED
    assert recovered[0].error_code == "analysis_outcome_unknown"


async def test_store_rejects_unknown_storage_schema(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    store._store.async_load = AsyncMock(  # type: ignore[method-assign]
        return_value={"schema_version": 2, "activities": {}}
    )
    with pytest.raises(ModelValidationError, match="unsupported_store_version"):
        await store.async_load()


async def test_terminal_activity_rejects_context_updates(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(replace(_record(), stage=ActivityStage.COMPLETED))
    with pytest.raises(StoreConflictError, match="terminal_activity"):
        await store.async_merge_context(
            "activity_1", detection_ids=("event_1",), updated_at=110
        )


async def test_same_timestamp_zone_updates_merge_deterministically(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    await store.async_merge_context(
        "activity_1", zone_update=(110, ("near",)), updated_at=110
    )
    merged = await store.async_merge_context(
        "activity_1", zone_update=(110, ("far",)), updated_at=110
    )
    assert merged.zone_updates == ((110, ("far", "near")),)


async def test_store_prunes_oldest_terminal_history(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1", max_activities=2)
    await store.async_load()
    for index in range(3):
        record = replace(
            _record(),
            activity_id=f"activity_{index}",
            stage=ActivityStage.COMPLETED,
            created_at=100 + index,
            updated_at=100 + index,
        )
        await store.async_create(record)
    assert store.get("activity_0") is None
    assert store.get("activity_1") is not None
    assert store.get("activity_2") is not None


async def test_history_pruning_never_evicts_new_record(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1", max_activities=2)
    await store.async_load()
    for index in (1, 2):
        await store.async_create(
            replace(
                _record(),
                activity_id=f"activity_{index}",
                stage=ActivityStage.COMPLETED,
                created_at=100 + index,
                updated_at=100 + index,
            )
        )
    new_record = replace(
        _record(),
        activity_id="activity_new",
        stage=ActivityStage.COMPLETED,
        created_at=90,
        updated_at=90,
    )
    assert await store.async_create(new_record) == new_record
    assert store.get("activity_new") == new_record


async def test_ingress_buffer_is_saved_and_removed_idempotently(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    message = IngressMessage(
        kind=IngressKind.FRIGATE_EVENT,
        entry_id="entry_1",
        source_id="event_1",
        event_id="event_1",
        event_type="new",
        occurred_at=95,
        camera="front",
    )
    first = await store.async_buffer_ingress(message, settle_after=130)
    second = await store.async_buffer_ingress(message, settle_after=140)
    assert first == second
    assert store.buffered_ingress() == (first,)
    assert await store.async_remove_buffered_ingress(first.buffer_id)
    assert not await store.async_remove_buffered_ingress(first.buffer_id)
    assert store.buffered_ingress() == ()


async def test_ingress_buffer_survives_reload(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    store._store.async_save = AsyncMock()  # type: ignore[method-assign]
    message = IngressMessage(
        kind=IngressKind.FRIGATE_REVIEW,
        entry_id="entry_1",
        source_id="review_1",
        review_id="review_1",
        occurred_at=100,
        camera="front",
        detection_ids=("event_1",),
    )
    buffered = await store.async_buffer_ingress(message, settle_after=110)
    raw = store._store.async_save.await_args.args[0]  # type: ignore[attr-defined]
    reloaded = ActivityStore(hass, "entry_1")
    reloaded._store.async_load = AsyncMock(return_value=raw)  # type: ignore[method-assign]
    await reloaded.async_load()
    assert reloaded.buffered_ingress() == (buffered,)


async def test_buffer_keeps_distinct_updates_for_the_same_detection(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    first = IngressMessage(
        kind=IngressKind.FRIGATE_EVENT,
        entry_id="entry_1",
        source_id="event_1",
        event_id="event_1",
        event_type="new",
        occurred_at=95,
        camera="front",
        current_zones=("near",),
    )
    second = replace(
        first,
        event_type="update",
        occurred_at=96,
        current_zones=("far",),
    )
    await store.async_buffer_ingress(first, settle_after=130)
    await store.async_buffer_ingress(second, settle_after=131)
    assert [item.message for item in store.buffered_ingress()] == [first, second]


async def test_complete_media_persists_selection_source(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(replace(_record(), stage=ActivityStage.SEALED))
    completed = await store.async_complete_media(
        "activity_1",
        key="media:activity_1:1",
        evidence_mode="review_six",
        evidence_revision=1,
        evidence_path="/tmp/activity_1.jpg",
        evidence_media_url="media-source://frigate_vision/entry_1/activity_1",
        sample_times=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        selection_source="path_motion",
        updated_at=120,
    )
    assert completed.selection_source == "path_motion"


async def test_complete_analysis_accepts_the_key_that_was_actually_claimed(
    hass: HomeAssistant,
) -> None:
    """The completion must look for the same key the start persisted.

    `VisionAnalyzer` claims its side effect with `analysis_key(...)`, which
    includes the scene: `analysis:<id>:<scene_mode>:<prompt_version>`. This
    method previously rebuilt the key as `analysis:<id>:<prompt_version>`, so
    the membership test could never succeed. Every analysis reached
    `analysis_started` and then failed with `side_effect_key_mismatch`; because
    the stage was already `analysis_started`, the retry policy also refused to
    re-run it, and no delivery was ever attempted.

    The key must therefore be derived, not re-spelled here.
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(replace(_record(), stage=ActivityStage.EVIDENCE_READY))

    scene = "review_six"
    version = "prompt_3"
    claimed = analysis_key("activity_1", scene, version)
    assert await store.async_start_side_effect(
        "activity_1",
        claimed,
        ActivityStage.EVIDENCE_READY,
        ActivityStage.ANALYSIS_STARTED,
        updated_at=110,
    )

    done = await store.async_complete_analysis(
        "activity_1",
        scene_mode=scene,
        prompt_version=version,
        classification="visitor",
        description="一人经过。",
        confidence=60,
        updated_at=120,
    )
    assert done.stage is ActivityStage.ANALYSIS_DONE
    assert done.classification == "visitor"
