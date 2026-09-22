"""Activity event entity."""

from homeassistant.components.event import EventEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .entity import EntryEntity
from .models import ActivityRecord, ActivityStage
from .runtime import IntegrationRuntime


class ActivityEvent(EntryEntity, EventEntity):
    _attr_name = "Activity"
    _attr_event_types = ["activity_completed", "activity_failed"]

    async def async_added_to_hass(self) -> None:
        await super().async_added_to_hass()

        def handle(record: ActivityRecord) -> None:
            event_type = (
                "activity_completed"
                if record.stage is ActivityStage.COMPLETED
                else "activity_failed"
            )
            self._trigger_event(
                event_type,
                {
                    "activity_id": record.activity_id,
                    "classification": record.classification,
                    "confidence": record.confidence,
                    "error_code": record.error_code,
                },
            )
            self.async_write_ha_state()

        self.async_on_remove(self.runtime.store.async_subscribe(handle))


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime = entry.runtime_data
    if isinstance(runtime, IntegrationRuntime):
        async_add_entities([ActivityEvent(entry, runtime, "activity")])
