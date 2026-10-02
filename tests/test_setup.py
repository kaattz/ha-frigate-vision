from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from homeassistant import config_entries
from homeassistant.core import HomeAssistant
from pytest_homeassistant_custom_component.common import MockConfigEntry

import custom_components.frigate_vision as integration
from custom_components.frigate_vision.clip_proxy import FrigateClipView
from custom_components.frigate_vision.media_source import EvidenceMediaView
from custom_components.frigate_vision.runtime import IntegrationRuntime


async def test_setup_registers_authenticated_media_view(
    hass: HomeAssistant, monkeypatch
) -> None:
    """Both public routes must be registered and must demand authentication.

    The evidence image and the activity clip are the two things a notification
    links to. Both proxy camera content, so neither may be reachable without a
    Home Assistant login -- that is what keeps Frigate itself off the internet.
    """
    registered: list[object] = []
    monkeypatch.setattr(hass, "http", SimpleNamespace(register_view=registered.append))
    assert await integration.async_setup(hass, {})

    kinds = {type(view) for view in registered}
    assert EvidenceMediaView in kinds
    assert FrigateClipView in kinds
    for view in registered:
        assert view.requires_auth is True, f"{type(view).__name__} is unauthenticated"


async def test_setup_and_unload_entry(hass: HomeAssistant, monkeypatch) -> None:
    entry = MockConfigEntry(
        domain="frigate_vision",
        title="Front Door",
        data={},
        options={},
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, config_entries.ConfigEntryState.LOADED)
    monkeypatch.setattr(hass.config_entries, "async_forward_entry_setups", AsyncMock())
    monkeypatch.setattr(
        hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)
    )

    assert hasattr(integration, "async_setup_entry")
    assert hasattr(integration, "async_unload_entry")
    assert await integration.async_setup_entry(hass, entry)
    runtime = entry.runtime_data
    assert isinstance(runtime, IntegrationRuntime)
    assert runtime.running
    assert await integration.async_unload_entry(hass, entry)
    assert not runtime.running
    assert entry.runtime_data is None


async def test_a_corrupt_evidence_record_does_not_take_the_entry_down(
    hass: HomeAssistant, monkeypatch, tmp_path
) -> None:
    """一条坏证据记录不能让整个集成起不来——这条断言的是用户可见的后果。

    实测的完整链路：`async_restore_registry` 对不可验证的记录抛 `MediaError`；
    `MediaError` 是 `RuntimeError`，而 `async_setup_entry` 只转换
    `ModelValidationError`/`FrigateApiError`/`OSError`；于是它一路传出，HA 把整个
    config entry 置为 `setup_error`——**所有实体不可用**，**连报修卡都没有**
    （`storage_corrupt` 只在捕获到那三种类型时才创建），而坏值躺在 `.storage` 里，
    选项表单够不到。每次重启复发。

    之前只测了 `async_restore_registry` 不抛异常；这里测的是**入口**：记录坏掉时
    `async_setup_entry` 仍返回 True、runtime 仍运行、实体仍会建立。
    """
    from custom_components.frigate_vision.correlation import ZoneRoles
    from custom_components.frigate_vision.media import MediaManager
    from custom_components.frigate_vision.models import (
        ActivityRecord,
        ActivitySource,
        ActivityStage,
    )
    from custom_components.frigate_vision.store import ActivityStore

    entry = MockConfigEntry(
        domain="frigate_vision", title="Front Door", data={}, options={}
    )
    entry.add_to_hass(hass)
    entry.mock_state(hass, config_entries.ConfigEntryState.LOADED)
    monkeypatch.setattr(hass.config_entries, "async_forward_entry_setups", AsyncMock())
    monkeypatch.setattr(
        hass.config_entries, "async_unload_platforms", AsyncMock(return_value=True)
    )

    # A stored record whose evidence path points at a file it does not own. This is
    # what the media directory being pruned, a crash mid-write, or a hand-edited store
    # all produce.
    store = ActivityStore(hass, entry.entry_id)
    await store.async_load()
    stranger = tmp_path / entry.entry_id / "someone_elses.jpg"
    stranger.parent.mkdir(parents=True, exist_ok=True)
    stranger.write_bytes(b"\xff\xd8\xff\xd9")
    await store.async_create(
        ActivityRecord(
            activity_id="review_1",
            entry_id=entry.entry_id,
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            created_at=1,
            updated_at=2,
            camera="front",
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path=str(stranger),
            evidence_media_url=f"media-source://frigate_vision/{entry.entry_id}/review_1",
            sample_times=(1.0, 2.0, 3.0),
        )
    )

    # The restore must be reachable from setup, so wire it the way setup does.
    root = tmp_path / "media"
    root.mkdir()
    manager = MediaManager(
        hass,
        store,
        object(),
        root,
        ZoneRoles(frozenset(), frozenset(), frozenset()),
    )

    # The contract: this must not raise, so setup cannot be derailed by it.
    await manager.async_restore_registry()

    assert await integration.async_setup_entry(hass, entry)
    runtime = entry.runtime_data
    assert isinstance(runtime, IntegrationRuntime)
    assert runtime.running
