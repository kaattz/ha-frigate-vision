from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    ProcessingMode,
    analysis_key,
    delivery_key,
    media_key,
    retry_is_safe,
)
from custom_components.frigate_vision.store import (
    ActivityStore,
    StoreConflictError,
)


async def test_hundred_replays_keep_one_activity(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.COLLECTING,
        processing_mode=ProcessingMode.SHADOW,
        created_at=1,
        updated_at=1,
        camera="front",
    )
    for _ in range(100):
        assert await store.async_create(record) == record
    assert store.get("activity_1") == record
    await store.async_transition(
        "activity_1",
        ActivityStage.COLLECTING,
        ActivityStage.SEALED,
        updated_at=2,
    )
    await store.async_transition(
        "activity_1",
        ActivityStage.SEALED,
        ActivityStage.EVIDENCE_READY,
        updated_at=3,
    )
    key = analysis_key("activity_1", "review_six", "prompt_1")
    claims = [
        await store.async_start_side_effect(
            "activity_1",
            key,
            ActivityStage.EVIDENCE_READY,
            ActivityStage.ANALYSIS_STARTED,
            updated_at=4,
        )
        for _ in range(100)
    ]
    assert claims.count(True) == 1
    assert claims.count(False) == 99
    assert store.get("activity_1").stage is ActivityStage.ANALYSIS_STARTED  # type: ignore[union-attr]


async def test_side_effect_start_is_atomic_when_store_write_fails(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    record = ActivityRecord(
        activity_id="activity_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.EVIDENCE_READY,
        processing_mode=ProcessingMode.SHADOW,
        created_at=1,
        updated_at=1,
        camera="front",
    )
    await store.async_create(record)
    store._store.async_save = AsyncMock(side_effect=OSError("disk full"))  # type: ignore[method-assign]
    with pytest.raises(OSError, match="disk full"):
        await store.async_start_side_effect(
            "activity_1",
            analysis_key("activity_1", "review_six", "prompt_1"),
            ActivityStage.EVIDENCE_READY,
            ActivityStage.ANALYSIS_STARTED,
            updated_at=2,
        )
    unchanged = store.get("activity_1")
    assert unchanged is not None
    assert unchanged.stage is ActivityStage.EVIDENCE_READY
    assert unchanged.claimed_side_effects == ()


async def test_side_effect_start_rejects_wrong_key_family(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=1,
            camera="front",
        )
    )
    with pytest.raises(StoreConflictError, match="side_effect_key_mismatch"):
        await store.async_start_side_effect(
            "activity_1",
            delivery_key("activity_1", "attempt_1"),
            ActivityStage.EVIDENCE_READY,
            ActivityStage.ANALYSIS_STARTED,
            updated_at=2,
        )
    with pytest.raises(StoreConflictError, match="side_effect_key_mismatch"):
        await store.async_start_side_effect(
            "activity_1",
            analysis_key("another_activity", "review_six", "prompt_1"),
            ActivityStage.EVIDENCE_READY,
            ActivityStage.ANALYSIS_STARTED,
            updated_at=2,
        )


def test_side_effect_keys_and_safe_retry_policy() -> None:
    assert media_key("activity_1", 2) == "media:activity_1:2"
    assert (
        analysis_key("activity_1", "review_six", "prompt_1")
        == "analysis:activity_1:review_six:prompt_1"
    )
    assert delivery_key("activity_1", "attempt_1") == "delivery:activity_1:attempt_1"
    assert retry_is_safe("evidence_incomplete")
    assert retry_is_safe("media_retry_exhausted")
    assert not retry_is_safe("analysis_outcome_unknown")
    assert not retry_is_safe("delivery_outcome_unknown")


async def test_concurrent_retry_creates_one_active_attempt_and_keeps_source(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="door_original",
            entry_id="entry_1",
            source=ActivitySource.DOOR_CYCLE,
            stage=ActivityStage.FAILED,
            processing_mode=ProcessingMode.SHADOW,
            created_at=1,
            updated_at=2,
            camera="front",
            error_code="media_retry_exhausted",
            finalization_deadline=3,
        )
    )
    results = await asyncio.gather(
        store.async_create_retry("door_original", now=4),
        store.async_create_retry("door_original", now=4),
        return_exceptions=True,
    )
    attempts = [item for item in results if isinstance(item, ActivityRecord)]
    assert len(attempts) == 1
    assert attempts[0].source is ActivitySource.DOOR_CYCLE
    assert attempts[0].stage is ActivityStage.SEALED
    with pytest.raises(StoreConflictError, match="retry_requires_root_activity"):
        await store.async_create_retry(attempts[0].activity_id, now=5)
