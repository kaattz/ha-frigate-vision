"""Stable Home Assistant Repair issue helpers."""

from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN

ISSUES = {
    "vision_not_configured",
    "provider_unavailable",
    "frigate_unavailable",
    "door_mapping_invalid",
    "door_open_too_long",
    "storage_corrupt",
    "media_cleanup_failed",
    "delivery_outcome_unknown",
}


def async_set_issue(hass: HomeAssistant, entry_id: str, code: str) -> None:
    if code not in ISSUES:
        raise ValueError("unknown_repair_issue")
    ir.async_create_issue(
        hass,
        DOMAIN,
        f"{entry_id}_{code}",
        is_fixable=False,
        severity=ir.IssueSeverity.ERROR,
        translation_key=code,
        translation_placeholders={"entry_id": entry_id},
    )


def async_clear_issue(hass: HomeAssistant, entry_id: str, code: str) -> None:
    if code not in ISSUES:
        raise ValueError("unknown_repair_issue")
    ir.async_delete_issue(hass, DOMAIN, f"{entry_id}_{code}")
