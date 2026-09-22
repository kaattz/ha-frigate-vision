"""Memory-only status sensors."""

from collections.abc import Callable

from homeassistant.components.sensor import SensorEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.helpers.typing import StateType

from .entity import EntryEntity
from .models import ActivityRecord, ActivityStage
from .runtime import IntegrationRuntime


class RuntimeSensor(EntryEntity, SensorEntity):
    def __init__(
        self,
        entry: ConfigEntry,
        runtime: IntegrationRuntime,
        key: str,
        name: str,
        getter: Callable[[], StateType],
    ) -> None:
        super().__init__(entry, runtime, key)
        self._attr_name = name
        self._getter = getter

    @property
    def native_value(self) -> StateType:
        return self._getter()


def _latest(runtime: IntegrationRuntime) -> ActivityRecord | None:
    records = runtime.store.all()
    return (
        max(records, key=lambda item: (item.updated_at, item.activity_id))
        if records
        else None
    )


def _classification(runtime: IntegrationRuntime) -> StateType:
    records = [item for item in runtime.store.all() if item.classification is not None]
    record = (
        max(records, key=lambda item: (item.updated_at, item.activity_id))
        if records
        else None
    )
    return record.classification if record else None


def _confidence(runtime: IntegrationRuntime) -> StateType:
    records = [item for item in runtime.store.all() if item.confidence is not None]
    record = (
        max(records, key=lambda item: (item.updated_at, item.activity_id))
        if records
        else None
    )
    return record.confidence if record else None


def _error(runtime: IntegrationRuntime) -> StateType:
    records = [item for item in runtime.store.all() if item.error_code is not None]
    record = (
        max(records, key=lambda item: (item.updated_at, item.activity_id))
        if records
        else None
    )
    return runtime.last_error or (record.error_code if record else None)


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    runtime = entry.runtime_data
    if not isinstance(runtime, IntegrationRuntime):
        return
    async_add_entities(
        [
            RuntimeSensor(
                entry,
                runtime,
                "last_classification",
                "Last classification",
                lambda: _classification(runtime),
            ),
            RuntimeSensor(
                entry,
                runtime,
                "last_confidence",
                "Last confidence",
                lambda: _confidence(runtime),
            ),
            RuntimeSensor(
                entry,
                runtime,
                "last_error",
                "Last error",
                lambda: _error(runtime),
            ),
            RuntimeSensor(
                entry,
                runtime,
                "pending",
                "Pending activities",
                lambda: sum(
                    record.stage not in {ActivityStage.COMPLETED, ActivityStage.FAILED}
                    for record in runtime.store.all()
                ),
            ),
        ]
    )
