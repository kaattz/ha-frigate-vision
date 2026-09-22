from __future__ import annotations

import pytest

import custom_components.frigate_vision.models as models


def test_activity_round_trip_keeps_only_whitelisted_fields() -> None:
    record = models.ActivityRecord(
        activity_id="activity_123",
        entry_id="entry_123",
        source=models.ActivitySource.DOOR_CYCLE,
        stage=models.ActivityStage.COLLECTING,
        processing_mode=models.ProcessingMode.OBSERVE,
        created_at=100.0,
        updated_at=100.0,
        camera="front_door",
        association_deadline=110.0,
        finalization_deadline=220.0,
    )
    payload = record.to_dict()
    assert "raw_payload" not in payload
    assert models.ActivityRecord.from_dict(payload) == record

    legacy = payload | {"doorbell_at": 101.0}
    legacy.pop("doorbell_times")
    restored = models.ActivityRecord.from_dict(legacy)
    assert restored.doorbell_at == 101
    assert restored.doorbell_times == (101,)


def test_activity_rejects_unsafe_identity_and_unknown_stage() -> None:
    with pytest.raises(models.ModelValidationError, match="invalid_activity_id"):
        models.ActivityRecord(
            activity_id="../bad",
            entry_id="entry",
            source=models.ActivitySource.DOOR_CYCLE,
            stage=models.ActivityStage.COLLECTING,
            processing_mode=models.ProcessingMode.OBSERVE,
            created_at=1,
            updated_at=1,
            camera="front",
        )

    with pytest.raises(models.ModelValidationError, match="invalid_activity_ids"):
        models.ActivityRecord(
            activity_id="activity_1",
            entry_id="entry",
            source=models.ActivitySource.DOOR_CYCLE,
            stage=models.ActivityStage.COLLECTING,
            processing_mode=models.ProcessingMode.OBSERVE,
            created_at=1,
            updated_at=1,
            camera="front",
            detection_ids=("../unsafe",),
        )
    with pytest.raises(models.ModelValidationError, match="invalid_stage"):
        models.ActivityRecord.from_dict(
            {
                "schema_version": 1,
                "activity_id": "activity_1",
                "entry_id": "entry_1",
                "source": "door_cycle",
                "stage": "invented",
                "processing_mode": "observe",
                "created_at": 1,
                "updated_at": 1,
                "camera": "front",
            }
        )
    with pytest.raises(models.ModelValidationError, match="invalid_door_state"):
        models.ActivityRecord.from_dict(
            {
                "schema_version": 1,
                "activity_id": "activity_1",
                "entry_id": "entry_1",
                "source": "door_cycle",
                "stage": "collecting",
                "processing_mode": "observe",
                "created_at": 1,
                "updated_at": 1,
                "camera": "front",
                "door_remained_open": "yes",
            }
        )
    with pytest.raises(
        models.ModelValidationError, match="invalid_detection_zone_updates"
    ):
        models.ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=models.ActivitySource.DOOR_CYCLE,
            stage=models.ActivityStage.COLLECTING,
            processing_mode=models.ProcessingMode.OBSERVE,
            created_at=1,
            updated_at=1,
            camera="front",
            detection_ids=("event_1",),
            detection_zone_updates=(("missing", -1, ("near",)),),
        )


def test_stable_keys_are_deterministic_and_attempt_is_explicit() -> None:
    assert models.door_activity_id("entry_1", 123.5) == "door_entry_1_123500"
    assert models.review_activity_id("entry_1", "front", "review.abc") == (
        "review_entry_1_front_review.abc"
    )
    assert models.attempt_activity_id("review_entry_1_front_review.abc", 2) == (
        "review_entry_1_front_review.abc_attempt_2"
    )


def test_ingress_message_rejects_duplicate_zones_and_invalid_time() -> None:
    with pytest.raises(models.ModelValidationError, match="invalid_ingress_zones"):
        models.IngressMessage(
            kind=models.IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="event_1",
            occurred_at=10,
            camera="front",
            current_zones=("near", "near"),
        )
    with pytest.raises(models.ModelValidationError, match="invalid_timestamp"):
        models.IngressMessage(
            kind=models.IngressKind.DOOR,
            entry_id="entry_1",
            source_id="door_1",
            occurred_at=float("nan"),
        )


def test_ingress_message_round_trip_is_persistable() -> None:
    message = models.IngressMessage(
        kind=models.IngressKind.FRIGATE_EVENT,
        entry_id="entry_1",
        source_id="event_1",
        event_id="event_1",
        event_type="new",
        occurred_at=10,
        camera="front",
        current_zones=("near",),
        entered_zones=("near",),
    )
    assert models.IngressMessage.from_dict(message.to_dict()) == message


def test_ingress_message_rejects_unknown_event_type() -> None:
    with pytest.raises(models.ModelValidationError, match="invalid_event_type"):
        models.IngressMessage(
            kind=models.IngressKind.FRIGATE_EVENT,
            entry_id="entry_1",
            source_id="event_1",
            event_id="event_1",
            event_type="invented",
            occurred_at=10,
            camera="front",
        )


def _standalone_payload() -> dict:
    return {
        "schema_version": 1,
        "activity_id": "activity_1",
        "entry_id": "entry_1",
        "source": "standalone_review",
        "stage": "sealed",
        "processing_mode": "observe",
        "created_at": 100,
        "updated_at": 120,
        "camera": "front",
    }


def test_activity_round_trip_persists_selection_source() -> None:
    record = models.ActivityRecord.from_dict(
        _standalone_payload() | {"selection_source": "path_motion"}
    )
    assert record.selection_source == "path_motion"
    payload = record.to_dict()
    assert payload["selection_source"] == "path_motion"
    assert models.ActivityRecord.from_dict(payload) == record


def test_legacy_activity_without_selection_source_reads_none() -> None:
    record = models.ActivityRecord.from_dict(_standalone_payload())
    assert record.selection_source is None


def test_activity_rejects_unknown_selection_source() -> None:
    with pytest.raises(models.ModelValidationError, match="invalid_selection_source"):
        models.ActivityRecord.from_dict(
            _standalone_payload() | {"selection_source": "guessed"}
        )
