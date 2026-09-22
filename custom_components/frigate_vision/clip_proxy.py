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

from .models import ActivityRecord, ActivityStage

_LOGGER = logging.getLogger(__name__)

CLIP_URL_PREFIX = "/api/frigate_vision/clip/"

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
