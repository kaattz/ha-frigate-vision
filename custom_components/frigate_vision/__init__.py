"""Frigate Vision integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant

from .clip_proxy import FrigateClipView
from .const import DOMAIN
from .frigate import FrigateApiError
from .media_source import (
    DATA_MEDIA_REGISTRY,
    EvidenceMediaView,
    async_default_media_root,
)
from .models import ModelValidationError
from .repairs import async_clear_issue, async_set_issue
from .runtime import IntegrationRuntime
from .services import async_register_services

PLATFORMS = [Platform.SELECT, Platform.SENSOR, Platform.BINARY_SENSOR, Platform.EVENT]


async def async_setup(hass: HomeAssistant, config: dict[str, Any]) -> bool:
    """Register the authenticated evidence and clip endpoints once."""
    registry: dict[str, Path] = hass.data.setdefault(DATA_MEDIA_REGISTRY, {})
    root = await async_default_media_root(hass)
    hass.http.register_view(EvidenceMediaView(hass, registry, root))
    hass.http.register_view(FrigateClipView(hass, _frigate_base_url(hass, config)))
    await async_register_services(hass)
    return True


def _frigate_base_url(hass: HomeAssistant, config: dict[str, Any]) -> str:
    """Resolve the Frigate base URL for the clip proxy.

    Read from the first configured entry, because the view is registered before
    any entry is set up and therefore cannot be handed one. Only the URL is
    needed here; the record itself is looked up per request, so an entry that
    appears later still works.
    """
    for entry in hass.config_entries.async_entries(DOMAIN):
        frigate = entry.data.get("frigate")
        if isinstance(frigate, dict) and frigate.get("base_url"):
            return str(frigate["base_url"]).rstrip("/")
    return ""


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up an entry without starting production consumers yet."""
    try:
        entry.runtime_data = await IntegrationRuntime.async_create(
            hass, entry, int(entry.options.get("queue_size", 10))
        )
    except ModelValidationError:
        async_set_issue(hass, entry.entry_id, "storage_corrupt")
        raise
    except FrigateApiError:
        async_set_issue(hass, entry.entry_id, "frigate_unavailable")
        raise
    except OSError:
        async_set_issue(hass, entry.entry_id, "storage_corrupt")
        raise
    async_clear_issue(hass, entry.entry_id, "storage_corrupt")
    async_clear_issue(hass, entry.entry_id, "frigate_unavailable")
    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except Exception:
        await entry.runtime_data.async_stop()
        entry.runtime_data = None
        raise
    return True


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Unload an entry."""
    unloaded = await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
    if not unloaded:
        return False
    runtime = entry.runtime_data
    if isinstance(runtime, IntegrationRuntime):
        await runtime.async_stop()
    entry.runtime_data = None
    return True
