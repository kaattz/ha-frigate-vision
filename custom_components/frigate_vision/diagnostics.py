"""Redacted config-entry diagnostics."""

from collections import Counter
from typing import Any

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .repairs import ISSUES
from .runtime import IntegrationRuntime


async def async_get_config_entry_diagnostics(
    hass: HomeAssistant, entry: ConfigEntry
) -> dict[str, Any]:
    runtime = entry.runtime_data
    if not isinstance(runtime, IntegrationRuntime):
        return {"loaded": False}
    records = runtime.store.all()
    return {
        "loaded": True,
        "entry_id": entry.entry_id,
        "config_fields": {
            key: sorted(value) if isinstance(value, dict) else type(value).__name__
            for key, value in entry.data.items()
        },
        "option_fields": sorted(entry.options),
        "running": runtime.running,
        "healthy": runtime.is_healthy,
        "last_error_code": (
            runtime.last_error
            if runtime.last_error in ISSUES
            else ("runtime_error" if runtime.last_error else None)
        ),
        "stage_counts": dict(Counter(record.stage.value for record in records)),
        "error_counts": dict(
            Counter(record.error_code for record in records if record.error_code)
        ),
        "selection_source_counts": dict(
            Counter(
                record.selection_source
                for record in records
                if record.selection_source is not None
            )
        ),
        "buffered_ingress": len(runtime.store.buffered_ingress()),
        "activity_count": len(records),
    }
