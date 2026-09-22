"""Shared memory-only entity base."""

from homeassistant.config_entries import ConfigEntry
from homeassistant.helpers.entity import Entity

from .const import DOMAIN
from .runtime import IntegrationRuntime


class EntryEntity(Entity):
    _attr_has_entity_name = True

    def __init__(
        self, entry: ConfigEntry, runtime: IntegrationRuntime, key: str
    ) -> None:
        self._entry = entry
        self.runtime = runtime
        self._attr_unique_id = f"{entry.entry_id}_{key}"
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
        }
