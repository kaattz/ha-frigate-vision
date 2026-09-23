"""Media source exposing only registered evidence artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from aiohttp import web
from homeassistant.components.media_player.const import MediaClass, MediaType
from homeassistant.components.media_source.error import Unresolvable
from homeassistant.components.media_source.models import (
    BrowseMediaSource,
    MediaSource,
    MediaSourceItem,
    PlayMedia,
)
from homeassistant.core import HomeAssistant
from homeassistant.helpers.http import HomeAssistantView

from .const import DOMAIN
from .models import SAFE_ID

DATA_MEDIA_REGISTRY = f"{DOMAIN}_media_registry"
DATA_MEDIA_ROOT = f"{DOMAIN}_media_root"

# The one place the evidence route is spelled out. `EvidenceMediaView.url` and
# `evidence_media_path()` both derive from it, because the two must agree
# exactly: HA validates a signature with `claims["path"] != request.path`
# (an equality check, not a prefix match), so a path built even slightly
# differently from the registered route is refused with 401.
EVIDENCE_MEDIA_URL_TEMPLATE = f"/api/{DOMAIN}/media/{{entry_id}}/{{activity_id}}.jpg"


class EvidenceMediaSource(MediaSource):
    """Resolve a whitelist of evidence files, never an arbitrary path."""

    name = "Frigate Vision"

    def __init__(
        self, hass: HomeAssistant, registry: dict[str, Path], *, root: Path
    ) -> None:
        super().__init__(DOMAIN)
        self._hass = hass
        self._registry = registry
        self._root = root.resolve()

    def _resolve(self, identifier: str) -> Path:
        if identifier not in self._registry:
            raise Unresolvable("Unknown evidence item")
        path = self._registry[identifier].resolve()
        try:
            path.relative_to(self._root)
        except ValueError as exc:
            raise Unresolvable("Evidence path outside root") from exc
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise Unresolvable("Evidence file unavailable")
        return path

    async def async_resolve_media(self, item: MediaSourceItem) -> PlayMedia:
        if item.domain != DOMAIN:
            raise Unresolvable("Unknown media source")
        path = await self._hass.async_add_executor_job(self._resolve, item.identifier)
        return PlayMedia(
            f"/api/{DOMAIN}/media/{item.identifier}.jpg",
            "image/jpeg",
            path=path,
        )

    async def async_browse_media(self, item: MediaSourceItem) -> BrowseMediaSource:
        if item.identifier:
            path = await self._hass.async_add_executor_job(
                self._resolve, item.identifier
            )
            return BrowseMediaSource(
                domain=DOMAIN,
                identifier=item.identifier,
                media_class=MediaClass.IMAGE,
                media_content_type="image/jpeg",
                title=path.name,
                can_play=True,
                can_expand=False,
            )
        root = BrowseMediaSource(
            domain=DOMAIN,
            identifier=None,
            media_class=MediaClass.DIRECTORY,
            media_content_type=MediaType.APP,
            title=self.name,
            can_play=False,
            can_expand=True,
            children_media_class=MediaClass.IMAGE,
        )
        root.children = [
            BrowseMediaSource(
                domain=DOMAIN,
                identifier=identifier,
                media_class=MediaClass.IMAGE,
                media_content_type="image/jpeg",
                title=path.name,
                can_play=True,
                can_expand=False,
            )
            for identifier, path in sorted(self._registry.items())
        ]
        return root


async def async_get_media_source(hass: HomeAssistant) -> MediaSource:
    """Return the HA-loadable media source platform."""
    registry = hass.data.setdefault(DATA_MEDIA_REGISTRY, {})
    root = await async_default_media_root(hass)
    return EvidenceMediaSource(hass, registry, root=root)


async def async_default_media_root(hass: HomeAssistant) -> Path:
    """Resolve the configured HA media directory for integration artifacts."""
    if DATA_MEDIA_ROOT in hass.data:
        return cast(Path, hass.data[DATA_MEDIA_ROOT])
    media_dirs = hass.config.media_dirs
    base = (
        Path(media_dirs.get("local", next(iter(media_dirs.values()))))
        if media_dirs
        else Path(hass.config.path("media"))
    )
    root = await hass.async_add_executor_job((base / DOMAIN).resolve)
    hass.data[DATA_MEDIA_ROOT] = root
    return root


def evidence_media_path(entry_id: str, activity_id: str) -> str:
    """Return the browser-loadable path of one activity's evidence sheet.

    Distinct from `record.evidence_media_url`, which is a `media-source://`
    identifier and only resolvable by Home Assistant's own media browser -- it
    cannot be put in an `<img src>`. This is the HTTP route instead.

    Relative on purpose, like the HLS manifest path: the popup is opened both on
    the LAN and through the reverse tunnel, so letting the browser keep the
    current origin makes one delivered value work in both places.

    Identifiers are validated rather than escaped. Both are model-enforced
    `SAFE_ID`s, and a value that needs escaping is a bug upstream, not something
    to paper over here -- escaping it would also change the path the signature
    is bound to.
    """
    if not SAFE_ID.fullmatch(entry_id) or not SAFE_ID.fullmatch(activity_id):
        raise ValueError("invalid_evidence_identifier")
    return EVIDENCE_MEDIA_URL_TEMPLATE.format(
        entry_id=entry_id, activity_id=activity_id
    )


class EvidenceMediaView(HomeAssistantView):
    """Serve registered evidence through an authenticated HA endpoint."""

    url = EVIDENCE_MEDIA_URL_TEMPLATE
    name = f"api:{DOMAIN}:media"
    requires_auth = True

    def __init__(
        self, hass: HomeAssistant, registry: dict[str, Path], root: Path
    ) -> None:
        self._source = EvidenceMediaSource(hass, registry, root=root)

    async def get(
        self, request: web.Request, entry_id: str, activity_id: str
    ) -> web.FileResponse:
        try:
            path = await self._source._hass.async_add_executor_job(  # noqa: SLF001
                self._source._resolve,
                f"{entry_id}/{activity_id}",  # noqa: SLF001
            )
        except Unresolvable as exc:
            raise web.HTTPNotFound from exc
        return web.FileResponse(path)
