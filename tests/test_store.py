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
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.COLLECTING,
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


async def test_recovery_leaves_collecting_records_alone(
    hass: HomeAssistant,
) -> None:
    """恢复不得碰它无法判定结果未知与否的阶段。

    门周期删除前，「同一摄像头多个 collecting」会被恢复成 FAILED——那是门锁
    状态机的专有冲突。现在 `async_recover` 只标记 ANALYSIS_STARTED /
    DELIVERY_STARTED（结果未知，可能已扣费），其余非终态留给 RecoveryWork。
    一份 COLLECTING 记录经过恢复后必须原样保留。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    await store.async_create(replace(_record(), activity_id="activity_2"))
    recovered = await store.async_recover(200)
    assert all(item.stage is ActivityStage.COLLECTING for item in recovered)
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


async def test_merge_context_accumulates_box_updates(hass: HomeAssistant) -> None:
    """逐时刻的 box 要在合并处累积，和时间戳一起，按时间排序。

    这是「取 box 最大的那一帧」的唯一数据来源：MQTT 每条 update 带一个 box，
    这里把它们按发生时间串起来。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    await store.async_merge_context(
        "activity_1", box_update=(110, (0.1, 0.2, 0.3, 0.4)), updated_at=110
    )
    merged = await store.async_merge_context(
        "activity_1", box_update=(120, (0.2, 0.3, 0.4, 0.5)), updated_at=120
    )
    assert merged.box_updates == (
        (110, (0.1, 0.2, 0.3, 0.4)),
        (120, (0.2, 0.3, 0.4, 0.5)),
    )


async def test_box_updates_stay_strictly_increasing_on_a_repeat_timestamp(
    hass: HomeAssistant,
) -> None:
    """同一时刻重复上报时替换而不是追加 —— 模型层要求时间戳严格递增。

    Frigate 会在同一 frame_time 上重发；若照抄 zone_updates 的「并集」写法会得到
    两条同时间戳的记录，`ActivityRecord` 会拒绝，于是**之后所有**上下文合并都开始
    抛错，连 zone 一起丢。box 是单值，取后到的那条即可。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    await store.async_merge_context(
        "activity_1", box_update=(110, (0.1, 0.2, 0.3, 0.4)), updated_at=110
    )
    merged = await store.async_merge_context(
        "activity_1", box_update=(110, (0.9, 0.9, 0.9, 0.9)), updated_at=110
    )
    assert merged.box_updates == ((110, (0.9, 0.9, 0.9, 0.9)),)


async def test_box_update_is_ignored_when_absent(hass: HomeAssistant) -> None:
    """没有 box 的消息不动序列：缺 box 只是不生成特写，不该写入空洞。"""
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(_record())
    merged = await store.async_merge_context("activity_1", updated_at=130)
    assert merged.box_updates == ()


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


async def test_a_rebuilt_replay_is_not_reported_as_expired(
    hass: HomeAssistant,
) -> None:
    """重建好的 replay 不能被当成"证据已过期"——那会让用户看到"证据没了"。

    缺陷实测：`async_complete_media` 用 `replace()` 更新记录，但**没有清掉**
    `evidence_expired_at`；而 `async_create_retry` 又是 `replace(original, ...)`，
    于是 replay **继承**了 root 的过期标记。一个过期过的 root，其 replay 即使在
    `SEALED → EVIDENCE_READY` 之后拿到了**全新**的拼图，读出来仍是过期的：
    `services.py` 报 `evidence_expired=True` 且 `evidence_url=None`。

    也就是说：文件好好地在磁盘上，服务却告诉用户"证据已删除"。判据应当以**这次**
    写入为准，而不是沿用上一次的过期时间。
    """
    now = 1_700_000_000.0
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="review_expired",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.FAILED,
            created_at=now - 100_000,
            updated_at=now - 100_000,
            camera="front",
            error_code="media_retry_exhausted",
            # The old evidence was deleted by the retention sweep.
            evidence_expired_at=now - 50_000,
            sample_times=(1.0, 2.0, 3.0),
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path="/media/frigate_vision/entry_1/review_expired.jpg",
        )
    )

    retry = await store.async_create_retry("review_expired", now=now)
    # The retry starts from scratch: the old evidence is gone, so it re-collects.
    assert retry.stage is ActivityStage.SEALED, (
        "过期证据不该被当成可复用的证据"
    )

    rebuilt = await store.async_complete_media(
        retry.activity_id,
        key=f"media:{retry.activity_id}:1",
        evidence_mode="review_six",
        evidence_revision=1,
        evidence_path="/media/frigate_vision/entry_1/rebuilt.jpg",
        evidence_media_url=(
            f"media-source://frigate_vision/entry_1/{retry.activity_id}"
        ),
        sample_times=(1.0, 2.0, 3.0),
        updated_at=now + 10,
    )

    assert rebuilt.evidence_expired_at is None, (
        "刚写好的新拼图仍被标记为已过期——服务会告诉用户证据没了，而文件就在盘上"
    )
    assert rebuilt.evidence_path == "/media/frigate_vision/entry_1/rebuilt.jpg"


async def test_a_swallowed_write_is_detected_when_a_file_exists(
    hass: HomeAssistant, tmp_path
) -> None:
    """有文件可比对时，被吞掉的写必须被发现。

    这是契约的核心，用**真实文件**验证：HA 的 `Store._async_handle_write_data` 把
    `WriteError` 吞成一条日志（`helpers/storage.py:588`），所以"保存成功了没有"只能
    靠观察文件本身。把 `_async_write_data` 换成不写任何东西，然后确认守卫报错。

    用真实 `Store`（指向 tmp_path）而不是 harness 的 store double：后者本就不写文件，
    在那里这个失败模式不可观测——这也是守卫把"看不到文件"当作"不表态"的原因，
    见 `test_an_unobservable_store_is_not_treated_as_a_failure`。
    """
    from homeassistant.helpers.storage import Store as HAStore

    target = tmp_path / "frigate_vision.probe"
    target.write_text('{"seeded": true}', encoding="utf-8")

    store = ActivityStore(hass, "entry_1")
    # A real Store writing to a real file, so the guard has something to compare.
    real = HAStore(hass, 1, "probe", atomic_writes=True)
    real.path = str(target)  # type: ignore[misc]
    store._store = real  # type: ignore[assignment]
    await store.async_load()
    await store.async_create(_record())
    await store.async_transition(
        "activity_1", ActivityStage.COLLECTING, ActivityStage.SEALED, updated_at=101
    )
    await store.async_transition(
        "activity_1", ActivityStage.SEALED, ActivityStage.EVIDENCE_READY, updated_at=102
    )

    before = store._observed_write_state()
    assert before is not None, "guard should be able to observe the seeded file"

    # Silently do nothing: exactly what HA does when the write fails.
    async def silent_noop(_data):
        return None

    real._async_write_data = silent_noop  # type: ignore[method-assign]

    with pytest.raises(StoreConflictError, match="store_write_failed"):
        await store.async_start_side_effect(
            "activity_1",
            "analysis:activity_1:1",
            ActivityStage.EVIDENCE_READY,
            ActivityStage.ANALYSIS_STARTED,
            updated_at=110,
        )

    # And the consequence that makes it matter: memory must not have advanced past
    # what disk holds, or a restart replays the effect this claim was guarding.
    current = store.get("activity_1")
    assert current is not None
    assert current.claimed_side_effects == (), (
        "写没落盘，内存却记下了副作用声明——重启后会重放这次付费调用"
    )
    assert current.stage is ActivityStage.EVIDENCE_READY, (
        "写没落盘，内存阶段却推进了"
    )


async def test_an_unobservable_store_is_not_treated_as_a_failure(
    hass: HomeAssistant,
) -> None:
    """看不到文件时必须放行，而不是一律判失败。

    首次保存（文件尚不存在）与测试用的 store double 都属于"没有可比对的对象"。此时
    一律拒绝会让**全新安装的第一次声明**就失败，比它要防的风险更糟。守卫只在真正
    有东西可比对、且对比结果说明没写进去时才说话。
    """
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    assert store._observed_write_state() is None, (
        "harness store writes no file, so it must report 'no opinion'"
    )

    await store.async_create(_record())
    await store.async_transition(
        "activity_1", ActivityStage.COLLECTING, ActivityStage.SEALED, updated_at=101
    )
    await store.async_transition(
        "activity_1", ActivityStage.SEALED, ActivityStage.EVIDENCE_READY, updated_at=102
    )
    # Must not raise: there is nothing to compare, so the guard abstains.
    claimed = await store.async_start_side_effect(
        "activity_1",
        "analysis:activity_1:1",
        ActivityStage.EVIDENCE_READY,
        ActivityStage.ANALYSIS_STARTED,
        updated_at=110,
    )
    assert claimed is True
