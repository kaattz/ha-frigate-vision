"""Processing mode select."""

from homeassistant.components.select import SelectEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from .const import PROCESSING_MODES
from .entity import EntryEntity
from .runtime import IntegrationRuntime


class ProcessingModeSelect(EntryEntity, SelectEntity):
    _attr_name = "Processing mode"
    _attr_options = list(PROCESSING_MODES)

    @property
    def current_option(self) -> str:
        return str(self._entry.options.get("processing_mode", "observe"))

    async def async_select_option(self, option: str) -> None:
        if option not in PROCESSING_MODES:
            raise ValueError("invalid_processing_mode")
        self.hass.config_entries.async_update_entry(
            self._entry, options={**self._entry.options, "processing_mode": option}
        )
        await self.hass.config_entries.async_reload(self._entry.entry_id)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime = entry.runtime_data
    if isinstance(runtime, IntegrationRuntime):
        async_add_entities([ProcessingModeSelect(entry, runtime, "processing_mode")])
