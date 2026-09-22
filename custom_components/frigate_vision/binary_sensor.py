"""Runtime health binary sensor."""

from homeassistant.components.binary_sensor import BinarySensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .entity import EntryEntity
from .runtime import IntegrationRuntime


class HealthySensor(EntryEntity, BinarySensorEntity):
    _attr_name = "Healthy"

    @property
    def is_on(self) -> bool:
        return self.runtime.is_healthy


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime = entry.runtime_data
    if isinstance(runtime, IntegrationRuntime):
        async_add_entities([HealthySensor(entry, runtime, "healthy")])
