"""Tests for the authenticated Frigate clip proxy."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import Mock

import pytest
from aiohttp import web
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.clip_proxy import (
    CLIP_URL_PREFIX,
    FrigateClipView,
    clip_url_for,
    clip_window,
)
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    ProcessingMode,
)


def _record(**overrides: object) -> ActivityRecord:
    base: dict[str, object] = {
        "activity_id": "review_entry_1_front_review_1",
        "entry_id": "entry_1",
        "source": ActivitySource.STANDALONE_REVIEW,
        "stage": ActivityStage.COMPLETED,
        "processing_mode": ProcessingMode.LIVE,
        "created_at": 100.0,
        "updated_at": 140.0,
        "camera": "front",
        "review_ids": ("review_1",),
        "sample_times": (100.0, 105.0, 110.0, 118.0, 130.0, 133.0),
    }
    base.update(overrides)
    return ActivityRecord(**base)  # type: ignore[arg-type]


def test_clip_window_starts_before_the_person_and_ends_after() -> None:
    """The clip must show the whole activity, not start at its first frame.

    The first sample is already the first person frame, so a window beginning
    there would cut off the approach. The last sample sits before the postroll,
    so ending there would cut off the departure. A margin on both sides keeps the
    movement that gives the clip its meaning.
    """
    start, end = clip_window(_record())
    assert start < 100.0
    assert end > 133.0
    # Long enough to be useful, short enough not to pull in unrelated footage.
    assert 8.0 <= end - start <= 90.0


def test_clip_window_stays_bounded_for_a_very_long_activity() -> None:
    """The clip must not grow without limit.

    Sample times are validated by the model, but a long activity is legitimate:
    recorded reviews run past 700 seconds. Left unbounded the window would follow
    it, and a phone would be asked to fetch an enormous video.
    """
    start, end = clip_window(
        _record(sample_times=(100.0, 300.0, 500.0, 700.0, 900.0, 1100.0))
    )
    assert end - start <= 90.0
    # Still anchored to the activity's beginning, not to its tail.
    assert start < 100.0


def test_clip_window_handles_a_record_with_no_samples() -> None:
    """A record whose samples are missing still needs a usable window."""
    start, end = clip_window(_record(sample_times=()))
    assert end > start
    assert 8.0 <= end - start <= 90.0


def test_clip_url_is_absolute_and_uses_the_ha_origin() -> None:
    """The link goes into a chat message, so it must be a full URL.

    A relative path would be meaningless once pasted into WeChat, and the
    external origin is the only host that resolves away from home.
    """
    url = clip_url_for("https://ha.example.com", "entry_1", _record())
    assert url is not None
    assert url.startswith("https://ha.example.com" + CLIP_URL_PREFIX)
    assert url.endswith(".mp4")
    # No doubled separators from a trailing slash on the configured origin.
    assert "//api" not in url.replace("https://", "")


def test_clip_url_is_omitted_without_an_origin() -> None:
    """With no known origin the caller must fall back, not emit a broken link."""
    assert clip_url_for("", "entry_1", _record()) is None


def test_clip_url_encodes_the_identifier() -> None:
    """Identifiers reach the URL, so they must be escaped rather than pasted."""
    url = clip_url_for("https://ha.example.com", "entry 1", _record())
    assert url is not None
    assert " " not in url


async def test_clip_view_builds_frigates_range_endpoint(hass: HomeAssistant) -> None:
    """The view must translate an activity into Frigate's range endpoint.

    Frigate serves a clip for any start/end pair, but only this integration knows
    which window an activity occupies. The view exists so the message can carry
    one stable, authenticated link.

    The upstream URL is checked through the view's own builder rather than by
    driving a full HTTP response: `web.StreamResponse.prepare` needs a real
    payload writer, so a mocked request cannot exercise the streaming path
    meaningfully -- it would only assert that aiohttp's internals ran.
    """
    record = _record()
    view = FrigateClipView(hass, "http://frigate.test:5001")
    url = view.upstream_url(record)

    assert url.startswith("http://frigate.test:5001/api/front/start/")
    assert url.endswith("/clip.mp4")
    start, end = clip_window(record)
    assert f"{start:.6f}" in url
    assert f"{end:.6f}" in url


def test_clip_view_url_escapes_the_camera() -> None:
    """The camera name reaches a URL path, so it must be escaped.

    `ActivityRecord` already constrains camera names to safe identifiers, so this
    guards the boundary rather than a reachable state: quoting here means a future
    relaxation of that rule cannot silently produce a malformed URL.
    """
    view = FrigateClipView(Mock(), "http://frigate")
    record = _record()
    object.__setattr__(record, "camera", "front door")
    url = view.upstream_url(record)
    assert "front door" not in url
    assert "front%20door" in url


def test_clip_url_survives_an_origin_with_a_trailing_slash() -> None:
    """A configured origin may carry a trailing slash; the result must not."""
    with_slash = clip_url_for("https://ha.example.com/", "entry_1", _record())
    without = clip_url_for("https://ha.example.com", "entry_1", _record())
    assert with_slash == without
    assert "example.com//" not in str(with_slash)


async def test_clip_view_refuses_an_unknown_activity(hass: HomeAssistant) -> None:
    """An unknown identifier must 404, not reach Frigate.

    The route is available to any logged-in user, so it must not become a way to
    ask Frigate for arbitrary footage.
    """
    store = Mock()
    store.get = Mock(return_value=None)
    runtime = Mock()
    runtime.store = store
    entry = Mock()
    entry.runtime_data = runtime
    hass.config_entries.async_get_entry = Mock(return_value=entry)  # type: ignore[method-assign]

    view = FrigateClipView(hass, "http://frigate.test:5001")
    with pytest.raises(web.HTTPNotFound):
        await view.get(Mock(spec=web.Request), "entry_1", "nope")


def test_hls_path_uses_integer_seconds_and_no_instance_id() -> None:
    """Frigate's VOD path segments accept integers, and this proxy form needs
    no instance id.

    Seconds are truncated to integers: a floating-point timestamp in the URL
    would make one activity produce two different URLs, and therefore two
    cache entries for identical footage.
    """
    from custom_components.frigate_vision.clip_proxy import hls_path_for

    record = _record(
        created_at=1789987214.292209,
        updated_at=1789987281.0,
        sample_times=(1789987214.29, 1789987247.5, 1789987281.0),
    )
    path = hls_path_for("front", record)
    assert path.startswith("/api/frigate/vod/front/start/")
    assert path.endswith("/index.m3u8")
    # No scheme and no host: a relative path is what makes the same stored
    # value work on the LAN and through the reverse tunnel.
    assert "://" not in path
    # Integer seconds only, in the variable part of the path. The fixed
    # `index.m3u8` suffix is excluded because its own dot is not a timestamp.
    tail = path.split("/vod/front/", 1)[1].removesuffix("index.m3u8")
    assert "." not in tail


def test_hls_path_covers_the_same_window_as_the_clip() -> None:
    """Both links must describe the same footage, or the popup would show a
    different span than the shareable link."""
    from custom_components.frigate_vision.clip_proxy import (
        clip_window,
        hls_path_for,
    )

    record = _record()
    start, end = clip_window(record)
    path = hls_path_for("front", record)
    assert f"/start/{int(start)}/" in path
    assert f"/end/{int(end)}/" in path


def test_hls_path_rejects_a_camera_name_that_cannot_be_a_url_segment() -> None:
    """The camera name reaches a URL path, so it is constrained, not escaped
    blindly."""
    from custom_components.frigate_vision.clip_proxy import (
        ClipUrlError,
        hls_path_for,
    )

    with pytest.raises(ClipUrlError, match="invalid_camera"):
        hls_path_for("front/door", _record())
    with pytest.raises(ClipUrlError, match="invalid_camera"):
        hls_path_for("", _record())


def test_hls_path_delegates_the_window_to_clip_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The path must carry `clip_window`'s answer, not a second opinion.

    Asserting the path against `clip_window`'s own output would also pass for a
    reimplementation that happened to use the same formula, so the window is
    replaced with values nothing else could derive: the popup and the clip link
    must describe one window, and only a call through proves they share it.
    """
    from custom_components.frigate_vision import clip_proxy

    calls: list[ActivityRecord] = []
    window = (4242.0, 4343.0)

    def fake_window(record: ActivityRecord) -> tuple[float, float]:
        calls.append(record)
        return window

    monkeypatch.setattr(clip_proxy, "clip_window", fake_window)
    record = _record()
    path = clip_proxy.hls_path_for("front", record)

    assert calls == [record]
    assert path == "/api/frigate/vod/front/start/4242/end/4343/index.m3u8"


def test_hls_path_accepts_a_camera_at_the_models_length_limit() -> None:
    """The guard must not be narrower than the model that feeds it.

    `ActivityRecord` accepts camera names up to 192 characters, and the popup
    degrades a rejected name to no HLS link at all -- a silent, successful-looking
    delivery with a working clip_url next to a missing hls_url. Frigate itself
    imposes no bound, and `quote` already reduces any string to one path segment,
    so the record's own limit is the only limit justified here.
    """
    from custom_components.frigate_vision.clip_proxy import (
        ClipUrlError,
        hls_path_for,
    )

    longest = "a" * 192
    # The record model is the source of the bound, so it must accept what the
    # path builder accepts -- otherwise the guard is stricter than its input.
    record = _record(camera=longest)
    path = hls_path_for(longest, record)
    assert path.startswith(f"/api/frigate/vod/{longest}/start/")

    with pytest.raises(ClipUrlError, match="invalid_camera"):
        hls_path_for("a" * 193, record)


async def test_signed_hls_url_carries_authSig_and_stays_relative(
    hass: HomeAssistant,
) -> None:
    """Every HLS segment is auth-gated, so the manifest must be signed.

    The HA Frigate integration's `VodSegmentProxyView` rejects any segment
    without a valid `authSig`, and Frigate echoes the manifest's query string
    into the segment URIs -- so signing the manifest is what authorises the
    whole stream.
    """
    from homeassistant.setup import async_setup_component

    from custom_components.frigate_vision.clip_proxy import signed_hls_url_for

    assert await async_setup_component(hass, "http", {})
    url = signed_hls_url_for(hass, _record())
    assert url is not None
    assert url.startswith("/api/frigate/vod/front/start/")
    assert "authSig=" in url
    # Still relative: the player runs same-origin inside the HA frontend, so a
    # relative URL works on the LAN and through the tunnel alike.
    assert "://" not in url


async def test_signed_hls_url_degrades_to_none_when_signing_fails(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A manifest that cannot be signed must not take the delivery down with it.

    `async_sign_path` reaches for `hass.data["http.auth"]`, which is absent
    until the http component is set up, so this is a real state rather than a
    contrived one. The play link is an enhancement and the notification is the
    product: the helper answers None and lets the caller deliver without a
    player.
    """
    from custom_components.frigate_vision import clip_proxy

    def boom(*args: object, **kwargs: object) -> str:
        raise KeyError("http.auth")

    monkeypatch.setattr(clip_proxy, "async_sign_path", boom)
    assert clip_proxy.signed_hls_url_for(hass, _record()) is None


async def test_signed_hls_url_degrades_to_none_for_an_unbuildable_path(
    hass: HomeAssistant, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unbuildable path is the same outcome as an unsignable one: no player.

    Reached through `hls_path_for` raising rather than by patching the signer,
    so the helper's own `try` is what is under test -- a `ClipUrlError` from
    the path builder must degrade, not propagate.
    """
    from custom_components.frigate_vision import clip_proxy

    def boom(camera: str, record: object) -> str:
        raise clip_proxy.ClipUrlError("invalid_camera")

    monkeypatch.setattr(clip_proxy, "hls_path_for", boom)
    assert clip_proxy.signed_hls_url_for(hass, _record()) is None


def test_signed_hls_url_delegates_the_path_to_hls_path_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Signing must wrap `hls_path_for`, not re-derive the manifest path.

    Two builders for one URL would drift: a window change in `hls_path_for`
    would silently stop applying to the delivered link. The path is replaced
    with one nothing else could produce, so only a call through can satisfy the
    assertion.
    """
    from custom_components.frigate_vision import clip_proxy

    seen: list[tuple[str, ActivityRecord]] = []
    ttls: list[timedelta] = []

    def fake_path(camera: str, record: ActivityRecord) -> str:
        seen.append((camera, record))
        return "/api/frigate/vod/side/start/1/end/2/index.m3u8"

    def fake_sign(hass: object, path: str, ttl: timedelta) -> str:
        ttls.append(ttl)
        return f"{path}?authSig=SIG"

    monkeypatch.setattr(clip_proxy, "hls_path_for", fake_path)
    monkeypatch.setattr(clip_proxy, "async_sign_path", fake_sign)

    record = _record()
    assert (
        clip_proxy.signed_hls_url_for(Mock(), record)
        == "/api/frigate/vod/side/start/1/end/2/index.m3u8?authSig=SIG"
    )
    assert seen == [("front", record)]
    # The same TTL as the shareable link: both are opened from one
    # notification, so a shorter one here would break the player while
    # `clip_url` still worked.
    assert ttls == [clip_proxy.CLIP_LINK_TTL]
