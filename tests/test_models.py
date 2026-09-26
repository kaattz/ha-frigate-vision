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


def test_activity_accepts_nine_sample_times() -> None:
    """A nine-cell sheet stores nine sample times, not six.

    The sheet grew from 2x3 to 3x3, so the stored evidence offsets grow with it.
    A whitelist of {0, 3, 6} would reject the record outright -- and it does so at
    read time, on every load, which would look like a corrupted store rather than
    a schema that had not been updated.
    """
    times = [100.0 + index * 5 for index in range(9)]
    record = models.ActivityRecord.from_dict(
        _standalone_payload() | {"sample_times": times}
    )
    assert record.sample_times == tuple(times)
    assert models.ActivityRecord.from_dict(record.to_dict()) == record


def test_activity_still_rejects_a_count_that_cannot_be_a_sheet() -> None:
    """The count must remain one a contact sheet can actually be built from."""
    with pytest.raises(models.ModelValidationError, match="invalid_sample_times"):
        models.ActivityRecord.from_dict(
            _standalone_payload()
            | {"sample_times": [100.0 + index * 5 for index in range(7)]}
        )


def test_activity_still_rejects_unsorted_or_duplicate_sample_times() -> None:
    with pytest.raises(models.ModelValidationError, match="invalid_sample_times"):
        models.ActivityRecord.from_dict(
            _standalone_payload()
            | {"sample_times": [110.0, 100.0, 120.0, 130.0, 140.0, 150.0]}
        )
    with pytest.raises(models.ModelValidationError, match="invalid_sample_times"):
        models.ActivityRecord.from_dict(
            _standalone_payload()
            | {"sample_times": [100.0, 100.0, 120.0, 130.0, 140.0, 150.0]}
        )


def test_a_chinese_classification_is_accepted() -> None:
    """标签名允许中文：用户要能写「宠物」而不是被迫写 `pet_only`。

    实测：分类值只在 models.py 一处被 SAFE_ID 挡下，而它不进入任何存储键
    （analysis_key 只用 activity_id/scene_mode/prompt_version），所以放宽它
    不影响缓存与幂等，历史记录也不受影响。
    """
    record = models.ActivityRecord.from_dict(
        _standalone_payload() | {"classification": "宠物"}
    )
    assert record.classification == "宠物"
    assert models.ActivityRecord.from_dict(record.to_dict()) == record


def test_a_classification_with_a_control_character_is_still_rejected() -> None:
    """放宽字母表不等于允许任意字符串：换行与控制字符会破坏显示和日志。"""
    with pytest.raises(models.ModelValidationError, match="invalid_classification"):
        models.ActivityRecord.from_dict(
            _standalone_payload() | {"classification": "宠物\n无人"}
        )
    with pytest.raises(models.ModelValidationError, match="invalid_classification"):
        models.ActivityRecord.from_dict(
            _standalone_payload() | {"classification": "bad\x00label"}
        )
    # C1 控制字符与 Unicode 行/段分隔符。U+2028/U+2029 是 JavaScript 的行
    # 终止符，会真的终止 JS 语句，而分类值会流向 HA 前端与蓝图模板。
    for bad in ("nel\u0085x", "ls\u2028x", "ps\u2029x"):
        with pytest.raises(
            models.ModelValidationError, match="invalid_classification"
        ):
            models.ActivityRecord.from_dict(
                _standalone_payload() | {"classification": bad}
            )


def _ingress(**overrides) -> models.IngressMessage:
    return models.IngressMessage(
        kind=models.IngressKind.FRIGATE_EVENT,
        entry_id="entry_1",
        source_id="event_1",
        event_id="event_1",
        event_type="update",
        occurred_at=1000.0,
        camera="front",
        **overrides,
    )


def test_an_ingress_message_carries_the_person_box() -> None:
    """box 必须能进消息 —— 这是「取最大 box 那一帧」的数据来源。

    Frigate 的 MQTT payload 里 `data.box` 是归一化的 `[x, y, w, h]`（0..1 的
    小数），所以这里存的是同一个四元组，不做像素换算。
    """
    message = _ingress(box=(0.2484375, 0.438888888888889, 0.23125, 0.516666666666667))
    assert message.box == (
        0.2484375,
        0.438888888888889,
        0.23125,
        0.516666666666667,
    )
    assert models.IngressMessage.from_dict(message.to_dict()) == message


def test_an_ingress_message_without_a_box_is_still_valid() -> None:
    """box 可选：不是每条消息都带（review 消息、或 Frigate 未上报）。

    缺 box 时 `to_dict` 必须**省略**该键，而不只是写一个 null：`BufferedIngress`
    的 `buffer_id` 是 `to_dict()` 的哈希，多一个键会让升级后每条未决消息换 id，
    缓冲区里的消息会被当成新的重新投递一遍。
    """
    message = _ingress()
    assert message.box is None
    assert "box" not in message.to_dict()


def test_an_ingress_message_rejects_a_malformed_box() -> None:
    """畸形 box 要拒绝，否则会一路传到裁剪函数才炸。"""
    for bad in (
        (0.1, 0.2, 0.3),  # 长度不足
        (0.1, 0.2, 0.3, 0.4, 0.5),  # 长度超出
        (0.1, 0.2, 0.0, 0.3),  # 零宽
        (0.1, 0.2, 0.3, 0.0),  # 零高
        (0.1, 0.2, -0.3, 0.4),  # 负宽
        (0.1, 0.2, 0.3, -0.4),  # 负高
        (float("nan"), 0.2, 0.3, 0.4),
        (0.1, float("inf"), 0.3, 0.4),
        ("a", 0.2, 0.3, 0.4),
        (True, 0.2, 0.3, 0.4),
        "nope",
    ):
        with pytest.raises(models.ModelValidationError, match="invalid_box"):
            _ingress(box=bad)


def test_a_legacy_payload_without_a_box_still_loads() -> None:
    """旧存档没有 box 键，必须仍能读回 —— 否则升级后历史活动全部解析失败。

    `ActivityRecord` 会写到磁盘，`IngressMessage` 也会进缓冲区存档。升级后第一次
    读到升级前写下的 payload 时，两层缺的键都必须落到默认值，而不是抛
    `ModelValidationError`（那会让整个 store 读不出来，看起来像存档损坏）。
    """
    legacy_ingress = {
        "kind": "frigate_event",
        "entry_id": "entry_1",
        "source_id": "event_1",
        "event_id": "event_1",
        "event_type": "update",
        "occurred_at": 1000.0,
        "started_at": None,
        "camera": "front",
        "review_id": None,
        "current_zones": ["near"],
        "entered_zones": ["near"],
        "detection_ids": [],
        "processing_mode": None,
        "manual": False,
    }
    assert "box" not in legacy_ingress
    message = models.IngressMessage.from_dict(legacy_ingress)
    assert message.box is None
    assert message.current_zones == ("near",)

    legacy_activity = _standalone_payload()
    assert "box_updates" not in legacy_activity
    record = models.ActivityRecord.from_dict(legacy_activity)
    assert record.box_updates == ()


def test_the_activity_record_round_trips_box_updates() -> None:
    """box 序列必须能存档再读回。"""
    updates = (
        (1000.5, (0.1, 0.2, 0.3, 0.4)),
        (1001.0, (0.2, 0.3, 0.4, 0.5)),
    )
    record = models.ActivityRecord.from_dict(
        _standalone_payload()
        | {"box_updates": [[timestamp, list(box)] for timestamp, box in updates]}
    )
    assert record.box_updates == updates
    assert models.ActivityRecord.from_dict(record.to_dict()) == record


def test_the_activity_record_rejects_unordered_box_updates() -> None:
    """逐时刻的 box 必须严格递增 —— 每条记录是「那一刻的 box」。

    顺序错乱或同一时刻两条，会让「取 box 最大的一帧」在时间上无法定位：后续
    步骤要按时间把 box 配到已选帧上，配错就会裁到另一时刻的画面。
    """
    with pytest.raises(models.ModelValidationError, match="invalid_box_updates"):
        models.ActivityRecord.from_dict(
            _standalone_payload()
            | {
                "box_updates": [
                    [1001.0, [0.1, 0.2, 0.3, 0.4]],
                    [1000.0, [0.2, 0.3, 0.4, 0.5]],
                ]
            }
        )
    with pytest.raises(models.ModelValidationError, match="invalid_box_updates"):
        models.ActivityRecord.from_dict(
            _standalone_payload()
            | {
                "box_updates": [
                    [1000.0, [0.1, 0.2, 0.3, 0.4]],
                    [1000.0, [0.2, 0.3, 0.4, 0.5]],
                ]
            }
        )
    with pytest.raises(models.ModelValidationError, match="invalid_box_updates"):
        models.ActivityRecord.from_dict(
            _standalone_payload()
            | {"box_updates": [[-1.0, [0.1, 0.2, 0.3, 0.4]]]}
        )
    with pytest.raises(models.ModelValidationError, match="invalid_box_updates"):
        models.ActivityRecord.from_dict(
            _standalone_payload()
            | {"box_updates": [[1000.0, [0.1, 0.2, 0.0, 0.4]]]}
        )
