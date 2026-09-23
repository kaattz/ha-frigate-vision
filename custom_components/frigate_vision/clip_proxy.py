"""Authenticated proxy for an activity's video clip.

Frigate can render a clip for any start/end pair, but only this integration knows
which window an activity occupies, and Frigate itself has no authentication on
this deployment. Routing the link through Home Assistant gives the message one
stable URL that:

  * resolves from outside the house, because it uses Home Assistant's own origin,
  * requires a Home Assistant login, so the camera is not exposed to the internet,
  * survives Frigate's frontend being redesigned, because it proxies one media
    URL instead of the review page and its bundles.

A clip is offered because it shows the whole activity -- several Frigate events
can belong to one -- where the previous video-based automation linked only a
single event's recording.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from urllib.parse import quote

import aiohttp
from aiohttp import web
from homeassistant.components.http.auth import async_sign_path
from homeassistant.core import HomeAssistant
from homeassistant.helpers.http import HomeAssistantView

from .media_source import evidence_media_path
from .models import SAFE_ID, ActivityRecord, ActivityStage

_LOGGER = logging.getLogger(__name__)

CLIP_URL_PREFIX = "/api/frigate_vision/clip/"

HLS_PATH_PREFIX = "/api/frigate/vod/"

# Seconds are reported to a tenth, which is far below the seek precision a phone
# can act on. Rounding keeps the delivered string short and stable.
OFFSET_DECIMALS = 1

# Deliberately not a comma. Home Assistant's native template parser reads a
# comma-separated template *result* as a tuple, so `"1.5,9.2"` arrives at
# `to_json` as a `TupleWrapper` and the script dies with
# `TypeError: Object of type TupleWrapper is not JSON serializable` -- measured
# on this deployment, where it broke the notification entirely rather than just
# the offsets. A pipe has no meaning to that parser, so the value stays the
# string it was written as.
OFFSET_SEPARATOR = "|"


class ClipUrlError(ValueError):
    """A clip URL cannot be built for this record."""


# How long a signed clip link stays usable. Long enough that a notification
# read the next morning still opens, short enough that a leaked URL stops
# working rather than exposing the camera indefinitely.
#
# Measured need: the activity is delivered once and the notification persists,
# so the window has to outlive "I'll look at it later" rather than the event.
CLIP_LINK_TTL = timedelta(hours=24)

# Seconds of footage kept either side of the activity. The first sample is
# already the first person frame and the last is before the postroll, so a
# window flush with those would cut off the approach and the departure.
CLIP_LEAD_SECONDS = 5.0
CLIP_TAIL_SECONDS = 6.0

# An activity can legitimately run for minutes, but a corrupt record should not
# produce an hour-long video.
MAX_CLIP_SECONDS = 90.0

# Streamed rather than buffered: a clip can be tens of megabytes, and the phone
# should start playing while the rest is still being fetched.
_CHUNK_SIZE = 64 * 1024
_UPSTREAM_TIMEOUT = aiohttp.ClientTimeout(total=None, sock_read=60)


def clip_window(record: ActivityRecord) -> tuple[float, float]:
    """Return the start and end of the footage for one activity.

    Bounded by `MAX_CLIP_SECONDS` so a record cannot yield an hour-long video,
    and widened by the margin on both sides so the approach and departure are
    included rather than clipped off.
    """
    # The model already guarantees sample times are finite, sorted and
    # non-negative, so the only thing to defend against here is span.
    last = max(record.sample_times) if record.sample_times else record.created_at
    last = max(last, record.created_at)
    start = max(0.0, record.created_at - CLIP_LEAD_SECONDS)
    end = min(last + CLIP_TAIL_SECONDS, start + MAX_CLIP_SECONDS)
    return start, end


def evidence_offsets(record: ActivityRecord) -> tuple[float, ...]:
    """Return each evidence sheet cell's offset into the clip, in seconds.

    The offsets are relative to the clip's own start rather than absolute wall
    clock times, so the player can assign one straight to `video.currentTime`
    without knowing anything about `clip_window()`. Recomputing the window in
    the browser would be a second implementation of it, free to drift from this
    one; deriving the offsets here makes the sheet and the video agree by
    construction.

    Every offset is clamped into the window. `clip_window()` caps the clip at
    `MAX_CLIP_SECONDS`, but `sample_times` is not capped the same way: a record
    whose first sample is long before its last can plan a final frame past the
    end of the clip. Left unclamped, tapping that cell would seek to a time the
    video does not contain and simply appear to do nothing.

    Order matches the sheet's cells: the model validates `sample_times` as
    sorted and strictly increasing, and `build_contact_sheet` places them in
    that same order, so cell *i* of the image is `sample_times[i]`.
    """
    start, end = clip_window(record)
    span = end - start
    return tuple(
        round(min(max(sample - start, 0.0), span), OFFSET_DECIMALS)
        for sample in record.sample_times
    )


def evidence_offsets_text(record: ActivityRecord) -> str:
    """Return the offsets as one separator-joined string, or "" when there are none.

    A string rather than a JSON array because this value crosses seven layers on
    its way to the card -- delivery event, blueprint variable, automation action,
    script field, `to_json`, `jq -s`, and finally a sensor attribute -- and a
    string stays a string in every one of them. The frontend already carries a
    comment about a value silently changing shape in that chain; sending text
    removes the question instead of testing for it.

    The separator is `OFFSET_SEPARATOR`, and it must not be a comma: HA's native
    template parser reads a comma-separated result as a tuple, which then fails
    to serialise inside the notification script and takes the whole notification
    down with it.

    Empty for a record with no samples, which the card reads as "no sheet" -- the
    same answer as a missing field, so there is only one case to handle.
    """
    offsets = evidence_offsets(record)
    if not offsets:
        return ""
    return OFFSET_SEPARATOR.join(f"{offset:g}" for offset in offsets)


def signed_evidence_url_for(
    hass: HomeAssistant, record: ActivityRecord
) -> str | None:
    """Return a loadable, signed URL for one activity's evidence sheet, or None.

    Signed for the same reason the clip and the manifest are, and the reason is
    stronger here than it looks: an `<img>` cannot carry an `Authorization`
    header, and Home Assistant's auth middleware accepts exactly two things --
    that header, or an `authSig` query parameter (`components/http/auth.py`,
    `auth_middleware`). There is no cookie path. So a bare `/api/` image URL is
    a 401 and a broken image, every time, with no way for the frontend to fix it
    by asking differently.

    The signature cannot be borrowed from the clip: HA validates it with
    `claims["path"] != request.path`, an exact match, so a token signed for the
    clip is refused for the image. Each route needs its own.

    Relative, like the manifest, so the browser supplies the origin and one
    stored value works on the LAN and through the tunnel alike. The TTL matches
    `CLIP_LINK_TTL` so the sheet and the video age out together rather than the
    picture breaking first.

    Returns None rather than raising. The sheet is a comparison aid and the
    notification is the product, so a URL that cannot be built costs the user
    the grid, not the delivery.
    """
    try:
        path = evidence_media_path(record.entry_id, record.activity_id)
        return async_sign_path(hass, path, CLIP_LINK_TTL)
    except Exception:  # noqa: BLE001
        # Broad for the same reason `signed_hls_url_for` is: `async_sign_path`
        # reaches for `hass.data["http.auth"]`, which does not exist until the
        # http component is set up, so a bare `hass` raises a `KeyError` that is
        # no signing error and has no narrower type worth catching.
        _LOGGER.warning(
            "Could not build or sign the evidence image for %s; delivering the "
            "activity without a comparison sheet",
            record.activity_id,
            exc_info=True,
        )
        return None


def clip_url_for(base_url: str, entry_id: str, record: ActivityRecord) -> str | None:
    """Return the absolute clip URL for a record, or None without an origin.

    Returns None rather than a relative path: the value is pasted into a chat
    message, where a path with no host is meaningless.
    """
    origin = base_url.strip().rstrip("/")
    if not origin:
        return None
    return (
        f"{origin}{clip_path_for(entry_id, record)}"
    )


def clip_path_for(entry_id: str, record: ActivityRecord) -> str:
    """Return the server-relative path of one activity's clip."""
    return (
        f"{CLIP_URL_PREFIX}"
        f"{quote(entry_id, safe='')}/{quote(record.activity_id, safe='')}.mp4"
    )


def hls_path_for(camera: str, record: ActivityRecord) -> str:
    """Return the HLS playlist path for one activity's footage.

    Deliberately server-relative and deliberately without the Frigate instance
    id. Both choices are load-bearing:

    * HLS rather than the MP4 clip, because every MP4 endpoint measured on this
      deployment answers `200` with `Transfer-Encoding: chunked` and no
      `Accept-Ranges`, and a mobile browser will not stream a 17 MB video it
      cannot range-request. HLS is segmented and its segments answer `206`.
    * Relative, because the link is opened both on the LAN and through the
      reverse tunnel; keeping the browser's current origin makes one stored URL
      work in both places, with no `external_url` change.
    * No instance id, because Frigate registers an `extra_urls` form
      (`/api/frigate/vod/...`) alongside the id-bearing one. The id comes from
      Frigate's own MQTT `client_id`, which this integration never reads.

    Seconds are truncated to integers so identical footage always yields an
    identical URL.
    """
    if not SAFE_ID.fullmatch(camera):
        raise ClipUrlError("invalid_camera")
    start, end = clip_window(record)
    return (
        f"{HLS_PATH_PREFIX}{quote(camera, safe='')}"
        f"/start/{int(start)}/end/{int(end)}/index.m3u8"
    )


def signed_clip_url_for(
    hass: HomeAssistant,
    base_url: str,
    entry_id: str,
    record: ActivityRecord,
) -> str | None:
    """Return a clip URL that opens without a Home Assistant session.

    A bare `/api/...` link is refused with 401 for anyone who is not already
    logged in. The link is read from a notification -- frequently on a phone
    that has no session in the browser it opens in -- so a bare link is a link
    that does nothing. Measured on this deployment: every tap on the unsigned
    URL was logged as `invalid authentication`, which the user sees as the page
    simply not loading.

    `async_sign_path` appends HA's own `authSig` token, so the endpoint stays
    authenticated while the URL itself carries the authorisation. The signature
    is bound to the path, so it cannot be replayed against another activity.

    Falls back to the unsigned URL if signing is unavailable: a link that needs
    a login still beats no link at all, and delivery must not fail over it.
    """
    origin = base_url.strip().rstrip("/")
    if not origin:
        return None
    try:
        path = async_sign_path(
            hass, clip_path_for(entry_id, record), CLIP_LINK_TTL
        )
    except Exception:  # noqa: BLE001
        _LOGGER.warning(
            "Could not sign the clip link for %s; falling back to an "
            "authenticated URL",
            record.activity_id,
            exc_info=True,
        )
        return f"{origin}{clip_path_for(entry_id, record)}"
    return f"{origin}{path}"


def signed_hls_url_for(hass: HomeAssistant, record: ActivityRecord) -> str | None:
    """Return a playable HLS manifest for one activity, or None.

    The manifest has to be signed even though the player runs inside an
    authenticated frontend. The HA Frigate integration's `VodSegmentProxyView`
    validates an `authSig` on *every segment request* and refuses the request
    without one, so an unsigned manifest loads and then nothing in it ever
    plays. Frigate echoes the manifest's own query string into the segment URIs
    it writes, so signing the manifest authorises the whole stream in one step:
    the segments inherit the token by themselves. Signing is also what the
    signature check expects -- it compares the token's `path` claim against the
    segment's parent directory with `startswith`, which the manifest's path
    satisfies.

    Deliberately server-relative, unlike `signed_clip_url_for`, and for the
    reason `hls_path_for` gives: the player resolves the URL against whatever
    origin the frontend is on, so one delivered value works on the LAN and
    through the reverse tunnel alike, with no `external_url` change. The
    signature is bound to the path rather than the host, so it stays valid
    whichever of those origins the request arrives on.

    The TTL is `CLIP_LINK_TTL` rather than a shorter player-sized window: the
    popup is opened from the same notification as the shareable link, so the
    two age out together instead of the player breaking first.

    Returns None instead of raising. The play link is an enhancement and the
    notification is the product, so a manifest that cannot be built or signed
    must cost the user the player, not the delivery.
    """
    try:
        # Both steps share one `try` because from the caller's side they are one
        # outcome: a manifest that cannot be built is no more playable than one
        # that cannot be signed, and there is a single answer for either.
        path = hls_path_for(record.camera, record)
        return async_sign_path(hass, path, CLIP_LINK_TTL)
    except Exception:  # noqa: BLE001
        # Broad on purpose, mirroring `signed_clip_url_for`: `async_sign_path`
        # reaches for `hass.data["http.auth"]`, which does not exist until the
        # http component is set up, so a bare `hass` fails with a `KeyError`
        # that is not a signing error and has no narrower type worth catching.
        # Letting it escape would fail the delivery over the play link.
        _LOGGER.warning(
            "Could not build or sign the HLS manifest for %s; delivering the "
            "activity without a playable link",
            record.activity_id,
            exc_info=True,
        )
        return None


class FrigateClipView(HomeAssistantView):
    """Stream one activity's clip from Frigate, behind Home Assistant's auth."""

    url = CLIP_URL_PREFIX + "{entry_id}/{activity_id}.mp4"
    name = "api:frigate_vision:clip"
    # The whole point of proxying: the camera stays reachable only to a caller
    # who has already authenticated with Home Assistant.
    requires_auth = True

    def __init__(self, hass: HomeAssistant, base_url: str) -> None:
        self._hass = hass
        self._base_url = base_url.rstrip("/")

    def upstream_url(self, record: ActivityRecord) -> str:
        """Return Frigate's clip URL for this activity's window."""
        start, end = clip_window(record)
        return (
            f"{self._base_url}/api/{quote(record.camera, safe='')}"
            f"/start/{start:.6f}/end/{end:.6f}/clip.mp4"
        )

    async def get(
        self, request: web.Request, entry_id: str, activity_id: str
    ) -> web.StreamResponse:
        entry = self._hass.config_entries.async_get_entry(entry_id)
        runtime = getattr(entry, "runtime_data", None) if entry else None
        if runtime is None or getattr(runtime, "store", None) is None:
            raise web.HTTPNotFound
        record = runtime.store.get(activity_id)
        if record is None or record.stage is ActivityStage.FAILED:
            raise web.HTTPNotFound

        upstream = self.upstream_url(record)
        client = getattr(runtime, "frigate_client", None)
        if client is None:
            raise web.HTTPServiceUnavailable
        session = getattr(client, "session", None) or aiohttp.ClientSession()
        try:
            async with session.get(
                upstream, timeout=_UPSTREAM_TIMEOUT, allow_redirects=True
            ) as source:
                if source.status != 200:
                    raise web.HTTPBadGateway
                response = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "video/mp4",
                        "Cache-Control": "private, max-age=300",
                        # Without this a browser may try to render inline and
                        # seek, which a proxied stream cannot always satisfy.
                        "Content-Disposition": "inline",
                    },
                )
                if source.content_length is not None:
                    response.content_length = source.content_length
                await response.prepare(request)
                async for chunk in source.content.iter_chunked(_CHUNK_SIZE):
                    await response.write(chunk)
                await response.write_eof()
                return response
        except aiohttp.ClientError as exc:
            _LOGGER.debug("clip upstream failed for %s: %s", activity_id, exc)
            raise web.HTTPBadGateway from exc
