from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.setup import async_setup_component
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision.delivery import (
    EVENT_ACTIVITY,
    DeliveryManager,
)
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


async def test_delivery_fires_once_and_requires_matching_ack(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    events = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=60)
    started = await manager.async_start("activity_1")
    await hass.async_block_till_done()
    assert len(events) == 1
    with pytest.raises(StoreConflictError, match="delivery_identity_mismatch"):
        await manager.async_ack("activity_1", "wrong")
    completed = await manager.async_ack("activity_1", started.delivery_attempt_id)
    assert completed.stage is ActivityStage.COMPLETED
    assert len(events) == 1
    await manager.async_stop()


async def test_delivery_timeout_becomes_unknown_without_refiring(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    events = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=0)
    await manager.async_start("activity_1")
    await hass.async_block_till_done()
    failed = store.get("activity_1")
    assert failed is not None and failed.stage is ActivityStage.FAILED
    assert failed.error_code == "delivery_outcome_unknown"
    assert len(events) == 1
    await manager.async_stop()


async def test_timeout_retries_temporary_store_write_failure(
    hass: HomeAssistant,
) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    manager = DeliveryManager(hass, entry, store, ack_timeout=0.02)
    completed = asyncio.Event()
    unsubscribe = store.async_subscribe(lambda record: completed.set())
    await manager.async_start("activity_1")
    original = store._store.async_save  # type: ignore[attr-defined]
    failures = 0

    async def flaky(payload):
        nonlocal failures
        if failures == 0:
            failures += 1
            raise OSError("temporary")
        return await original(payload)

    store._store.async_save = AsyncMock(side_effect=flaky)  # type: ignore[method-assign]
    await asyncio.wait_for(completed.wait(), timeout=2)
    assert failures == 1
    assert store.get("activity_1").error_code == "delivery_outcome_unknown"  # type: ignore[union-attr]
    unsubscribe()
    await manager.async_stop()


async def test_clip_link_carries_a_signature_so_it_opens_without_a_login(
    hass: HomeAssistant,
) -> None:
    """The delivered link must not be refused with 401.

    The link is read from a notification, usually on a phone whose browser has
    no Home Assistant session. Measured on this deployment, every tap on the
    unsigned URL was logged as `invalid authentication`, and the user sees that
    as the page simply not opening.

    A signed URL carries HA's own `authSig` token, so the click works while the
    endpoint stays authenticated.
    """
    from custom_components.frigate_vision.clip_proxy import FrigateClipView

    # Signing depends on the http component's own auth setup, which the bare
    # `hass` fixture does not perform. In production the integration registers
    # HTTP views, so http is always loaded.
    assert await async_setup_component(hass, "http", {})
    hass.config.external_url = "https://ha.example.com"

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    record = store.get("activity_1")
    assert record is not None

    events: list[dict] = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=60)
    await manager.async_start("activity_1")
    await hass.async_block_till_done()

    link = events[0]["clip_url"]
    assert link, "a clip link must be delivered when an external URL is set"
    # The signature is what makes the link open for a logged-out browser.
    assert "authSig=" in link, f"clip link is unsigned and will 401: {link}"
    # It must still address the same, authenticated endpoint.
    assert FrigateClipView.url.split("{")[0] in link
    await manager.async_stop()


async def test_signed_link_is_bound_to_one_activity(hass: HomeAssistant) -> None:
    """A signature must not be replayable against a different clip.

    `async_sign_path` binds the token to the path, so leaking one activity's
    link cannot be turned into access to another's footage.
    """
    from custom_components.frigate_vision.clip_proxy import signed_clip_url_for

    assert await async_setup_component(hass, "http", {})

    record = ActivityRecord(
        activity_id="activity_1",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.COMPLETED,
        processing_mode=ProcessingMode.LIVE,
        created_at=1,
        updated_at=2,
        camera="front",
    )
    other = ActivityRecord(
        activity_id="activity_2",
        entry_id="entry_1",
        source=ActivitySource.STANDALONE_REVIEW,
        stage=ActivityStage.COMPLETED,
        processing_mode=ProcessingMode.LIVE,
        created_at=1,
        updated_at=2,
        camera="front",
    )
    first = signed_clip_url_for(hass, "https://ha.example.com", "entry_1", record)
    second = signed_clip_url_for(hass, "https://ha.example.com", "entry_1", other)
    assert first and second
    assert "activity_1" in first and "activity_2" in second
    assert first != second
    # No origin means no absolute link, rather than a host-less path.
    assert signed_clip_url_for(hass, "", "entry_1", record) is None


async def test_activity_event_carries_a_playable_hls_url(hass: HomeAssistant) -> None:
    """The popup player needs a URL it can actually stream.

    `clip_url` is kept: it is the shareable absolute link. `hls_url` is what
    the in-app player uses, because no MP4 endpoint here can be
    range-requested and a mobile browser will not stream without that.
    """
    # `clip_url` is absolute, so it needs an origin. The bare `hass` fixture has
    # none, and `signed_clip_url_for` returns None without one -- the same
    # reason the signing test above sets it.
    hass.config.external_url = "https://ha.example.com"

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=100.0,
            updated_at=140.0,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
            sample_times=(100.0, 105.0, 110.0, 118.0, 130.0, 133.0),
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    events: list[dict] = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=60)
    await manager.async_start("activity_1")
    await hass.async_block_till_done()

    data = events[0]
    assert data["clip_url"], "the shareable link must still be delivered"
    hls = data["hls_url"]
    assert hls.startswith("/api/frigate/vod/front/start/")
    assert hls.endswith("/index.m3u8")
    # Relative, so the same stored value works on the LAN and via the tunnel.
    assert "://" not in hls
    await manager.async_stop()


async def test_delivery_survives_an_unbuildable_hls_url(
    hass: HomeAssistant, monkeypatch
) -> None:
    """A missing play URL must never cost the notification itself.

    `hls_path_for` validates the camera name, so a record with an odd camera
    raises. Delivery is the product; the play link is an enhancement. Letting
    the error escape would turn a cosmetic gap into a lost notification.
    """
    import custom_components.frigate_vision.delivery as delivery_module
    from custom_components.frigate_vision.clip_proxy import ClipUrlError

    def boom(camera: str, record: object) -> str:
        raise ClipUrlError("invalid_camera")

    monkeypatch.setattr(delivery_module, "hls_path_for", boom)

    # Same as above: the shareable link is absolute and needs an origin.
    hass.config.external_url = "https://ha.example.com"

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.ANALYSIS_DONE,
            processing_mode=ProcessingMode.LIVE,
            created_at=100.0,
            updated_at=140.0,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
            sample_times=(100.0, 105.0, 110.0, 118.0, 130.0, 133.0),
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    events: list[dict] = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=60)
    started = await manager.async_start("activity_1")
    await hass.async_block_till_done()

    # The activity still reached DELIVERY_STARTED and still fired its event.
    assert started.stage is ActivityStage.DELIVERY_STARTED
    assert len(events) == 1
    assert events[0]["hls_url"] is None
    assert events[0]["clip_url"], "the shareable link must survive too"
    await manager.async_stop()
