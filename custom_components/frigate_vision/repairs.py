"""Stable Home Assistant Repair issue helpers."""

from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from .const import DOMAIN

ISSUES = {
    "vision_not_configured",
    "provider_unavailable",
    "frigate_unavailable",
    "storage_corrupt",
    "media_cleanup_failed",
    "delivery_outcome_unknown",
    # One alert for every 5xx, rather than one per status.
    #
    # The status is a per-attempt detail: today it is 503, tomorrow a 502 from
    # the same overloaded gateway. A user does not act differently on the two --
    # the situation is "the provider is answering with errors and activities are
    # being lost" -- and a per-status issue would need a translation for every
    # status the provider can invent, with an untranslated one rendering as a
    # raw key in the UI. The specific status still travels on the
    # `last_error` sensor, which is the diagnostic surface.
    "provider_error",
}

# The shared code for "the provider answered 5xx". Named rather than spelled out
# at each use: `record_error` raises it and `clear_error` clears it, and the two
# have to agree or the alert would never come down.
PROVIDER_ERROR = "provider_error"


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
