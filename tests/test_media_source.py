from __future__ import annotations

import pytest
from homeassistant.components.media_source.error import Unresolvable
from homeassistant.components.media_source.models import MediaSourceItem
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.media_source import (
    DATA_MEDIA_REGISTRY,
    DATA_MEDIA_ROOT,
    EvidenceMediaSource,
    async_get_media_source,
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
