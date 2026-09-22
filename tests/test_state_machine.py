from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    ProcessingMode,
)
from custom_components.frigate_vision.store import (
    ActivityStore,
    StoreConflictError,
)


async def test_state_machine_rejects_skipped_analysis_stage(
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
    with pytest.raises(StoreConflictError, match="invalid_transition"):
        await store.async_transition(
            "activity_1",
            ActivityStage.EVIDENCE_READY,
            ActivityStage.DELIVERY_STARTED,
            updated_at=2,
        )
