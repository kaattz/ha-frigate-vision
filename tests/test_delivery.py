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

    The manifest must carry `authSig`, exactly as `clip_url` does, because the
    HA Frigate integration gates *every segment* on its own signature: an
    unsigned manifest loads and then nothing in it ever plays.
    """
    # `clip_url` is absolute, so it needs an origin. The bare `hass` fixture has
    # none, and `signed_clip_url_for` returns None without one.
    # Signing additionally needs the http component's auth: without this setup
    # `async_sign_path` raises, and both links silently degrade -- the clip to
    # its unsigned fallback and the manifest to no link at all -- so the test
    # would never touch the path production uses.
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
    # Still Frigate's manifest path -- the signature is appended as a query
    # parameter, so the path itself must be untouched.
    assert hls.split("?", 1)[0].endswith("/index.m3u8")
    # The signature is what authorises the segments that Frigate derives from
    # this manifest; without it every segment request is refused.
    assert "authSig=" in hls, f"the HLS manifest is unsigned and will not play: {hls}"
    # Relative, so the same stored value works on the LAN and via the tunnel.
    assert "://" not in hls, f"the HLS manifest must stay origin-less: {hls}"
    await manager.async_stop()


async def test_delivery_survives_an_unbuildable_hls_url(
    hass: HomeAssistant, monkeypatch
) -> None:
    """A missing play URL must never cost the notification itself.

    `signed_hls_url_for` answers `None` itself rather than raising, so this
    branch can only be reached if it starts raising again -- which is exactly
    the regression this pins. The `except` in `delivery` is a defensive
    backstop, and the behaviour it protects is the requirement that outlives
    the current signatures: delivery is the product; the play link is an
    enhancement. Letting such an error escape would turn a cosmetic gap into a
    lost notification.
    """
    import custom_components.frigate_vision.delivery as delivery_module
    from custom_components.frigate_vision.clip_proxy import ClipUrlError

    def boom(hass_: HomeAssistant, record: object) -> str:
        raise ClipUrlError("invalid_camera")

    monkeypatch.setattr(delivery_module, "signed_hls_url_for", boom)

    # As above: the shareable link is absolute and needs an origin, and signing
    # it needs the http component's auth.
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


async def test_activity_event_carries_the_evidence_sheet(hass: HomeAssistant) -> None:
    """The popup offers the model's own six-cell sheet beside the clip.

    Two fields travel: a signed image URL and the per-cell offsets into the
    clip, so a tap can seek to the frame a cell came from.

    The image needs its *own* signature. An `<img>` cannot send an
    `Authorization` header and HA's auth middleware has no cookie path, so an
    unsigned `/api/` image is a 401 and a broken image; and because a signature
    is bound to one exact path, the clip's token cannot be reused for it.
    """
    # Signing needs the http component's auth: without it `async_sign_path`
    # raises and the URL degrades to None, so the test would never touch the
    # path production uses.
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
            created_at=100.0,
            updated_at=140.0,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
            sample_times=(100.0, 105.0, 110.0, 118.0, 130.0, 133.0),
            evidence_mode="review_six",
            evidence_revision=1,
            evidence_path="/media/frigate_vision/entry_1/activity_1.jpg",
            evidence_media_url="media-source://frigate_vision/entry_1/activity_1",
        )
    )
    entry = MockConfigEntry(domain="frigate_vision", entry_id="entry_1", title="Front")
    events: list[dict] = []
    hass.bus.async_listen(EVENT_ACTIVITY, lambda event: events.append(event.data))
    manager = DeliveryManager(hass, entry, store, ack_timeout=60)
    await manager.async_start("activity_1")
    await hass.async_block_till_done()

    data = events[0]
    image = data["evidence_image_url"]
    assert image, "the comparison sheet must be delivered with the clip"
    # The HTTP route, not the `media-source://` identifier: only this one can be
    # put in an `<img src>`. `evidence_url` keeps its own meaning alongside it.
    assert image.split("?", 1)[0] == (
        "/api/frigate_vision/media/entry_1/activity_1.jpg"
    )
    assert "authSig=" in image, f"the sheet image is unsigned and will 401: {image}"
    assert "://" not in image, f"the sheet image must stay origin-less: {image}"
    assert data["evidence_url"] == (
        "media-source://frigate_vision/entry_1/activity_1"
    ), "the media-source identifier must not be repurposed"

    # One offset per cell, in cell order, relative to the clip's start.
    # Split on a pipe: a comma-separated value would be parsed as a tuple by
    # HA's template engine downstream and break the notification script.
    assert "," not in data["evidence_offsets"]
    offsets = [float(part) for part in data["evidence_offsets"].split("|")]
    assert len(offsets) == 6
    assert offsets == sorted(offsets)
    assert offsets[0] > 0.0

    # Neither existing link may regress.
    assert data["clip_url"], "the shareable link must still be delivered"
    assert data["hls_url"], "the playable manifest must still be delivered"
    await manager.async_stop()


async def test_delivery_survives_an_unbuildable_evidence_image(
    hass: HomeAssistant, monkeypatch
) -> None:
    """A sheet that cannot be signed must cost the grid, not the notification.

    `signed_evidence_url_for` answers `None` itself rather than raising, so this
    branch is reachable only if it starts raising again -- which is the
    regression this pins. The requirement that outlives the current signatures:
    delivery is the product, the comparison sheet is an enhancement.
    """
    from custom_components.frigate_vision import delivery as delivery_module

    def boom(*args, **kwargs):
        raise RuntimeError("signing exploded")

    monkeypatch.setattr(delivery_module, "signed_evidence_url_for", boom)

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

    assert len(events) == 1, "the notification must still be delivered"
    assert events[0]["evidence_image_url"] is None
    assert events[0]["clip_url"], "the shareable link must survive too"
    assert events[0]["hls_url"], "the playable manifest must survive too"
    await manager.async_stop()
