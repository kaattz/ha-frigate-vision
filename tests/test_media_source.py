from __future__ import annotations

import pytest
from homeassistant.components.media_source.error import Unresolvable
from homeassistant.components.media_source.models import MediaSourceItem
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.media_source import (
    DATA_MEDIA_REGISTRY,
    DATA_MEDIA_ROOT,
    EVIDENCE_MEDIA_URL_TEMPLATE,
    EvidenceMediaSource,
    EvidenceMediaView,
    async_get_media_source,
    evidence_media_path,
)


async def test_media_source_resolves_only_registered_safe_files(
    hass: HomeAssistant, tmp_path
) -> None:
    path = tmp_path / "entry_1" / "activity_1.jpg"
    path.parent.mkdir()
    path.write_bytes(b"jpeg")
    source = EvidenceMediaSource(hass, {"entry_1/activity_1": path}, root=tmp_path)
    item = MediaSourceItem(hass, "frigate_vision", "entry_1/activity_1", None)
    resolved = await source.async_resolve_media(item)
    assert resolved.path == path.resolve()
    assert resolved.mime_type == "image/jpeg"
    with pytest.raises(Unresolvable):
        await source.async_resolve_media(
            MediaSourceItem(
                hass,
                "frigate_vision",
                "../activity_1",
                None,
            )
        )


async def test_media_source_platform_factory_uses_ha_registry(
    hass: HomeAssistant, tmp_path
) -> None:
    hass.data[DATA_MEDIA_ROOT] = tmp_path
    hass.data[DATA_MEDIA_REGISTRY] = {}
    source = await async_get_media_source(hass)
    assert isinstance(source, EvidenceMediaSource)


def test_evidence_media_path_matches_the_registered_route() -> None:
    """The path handed to the signer must be the path HA serves.

    HA validates a signature with `claims["path"] != request.path` -- equality,
    not a prefix match -- so a path built even slightly differently from the
    registered route is refused with 401. Deriving both from one template is
    what keeps them identical; this asserts the derivation holds.
    """
    path = evidence_media_path("entry_1", "activity_1")
    assert path == "/api/frigate_vision/media/entry_1/activity_1.jpg"
    # The registered route must accept exactly what the signer was handed.
    assert EvidenceMediaView.url == EVIDENCE_MEDIA_URL_TEMPLATE
    assert path.replace("entry_1", "{entry_id}").replace(
        "activity_1", "{activity_id}"
    ) == EvidenceMediaView.url


def test_evidence_media_path_is_relative() -> None:
    """Relative, so the browser's own origin serves on the LAN and via a tunnel."""
    path = evidence_media_path("entry_1", "activity_1")
    assert "://" not in path
    assert path.startswith("/api/")


def test_evidence_media_path_rejects_unsafe_identifiers() -> None:
    """Rejected rather than escaped: escaping would move the signed path."""
    for entry_id, activity_id in (
        ("", "activity_1"),
        ("entry_1", ""),
        ("../etc", "activity_1"),
        ("entry_1", "activity/1"),
        ("entry_1", "activity 1"),
    ):
        with pytest.raises(ValueError):
            evidence_media_path(entry_id, activity_id)
