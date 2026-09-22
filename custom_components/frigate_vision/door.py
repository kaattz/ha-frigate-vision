"""Door lock mapping and cycle boundaries."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

from homeassistant.core import (
    CALLBACK_TYPE,
    Event,
    EventStateChangedData,
    HomeAssistant,
)
from homeassistant.helpers.event import async_track_state_change_event

from .models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    IngressKind,
    IngressMessage,
    ProcessingMode,
    door_activity_id,
)
from .store import ActivityStore

_LOGGER = logging.getLogger(__name__)


class DoorCycleError(RuntimeError):
    """Door input cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class DoorMapping:
    action_attribute: str
    open_values: frozenset[str]
    close_values: frozenset[str]
    side_attribute: str
    inside_values: frozenset[str]
    outside_values: frozenset[str]


@dataclass(frozen=True, slots=True)
class DoorResult:
    status: str
    record: ActivityRecord


def normalize_lock_event(
    attributes: dict[str, object], mapping: DoorMapping
) -> tuple[str, str]:
    if (
        mapping.action_attribute not in attributes
        or mapping.side_attribute not in attributes
    ):
        raise DoorCycleError("door_attribute_missing")
    action_value = str(attributes[mapping.action_attribute])
    side_value = str(attributes[mapping.side_attribute])
    if action_value in mapping.open_values:
        action = "open"
    elif action_value in mapping.close_values:
        action = "close"
    else:
        raise DoorCycleError("door_action_unknown")
    if side_value in mapping.inside_values:
        side = "inside"
    elif side_value in mapping.outside_values:
        side = "outside"
    else:
        side = "unknown"
    if action == "open" and side == "unknown":
        raise DoorCycleError("door_side_unknown")
    return action, side


def async_subscribe_lock(
    hass: HomeAssistant,
    entity_id: str,
    mapping: DoorMapping,
    callback: Callable[[str, str, float], Awaitable[None]],
    error_callback: Callable[[DoorCycleError], None] | None = None,
) -> CALLBACK_TYPE:
    """Subscribe to a stable HA event entity."""
    initial = hass.states.get(entity_id)
    last_event = initial.state if initial is not None else None

    async def handle_event(event: Event[EventStateChangedData]) -> None:
        nonlocal last_event
        new_state = event.data.get("new_state")
        if new_state is None or new_state.state in {"unknown", "unavailable"}:
            return
        if new_state.state == last_event:
            return
        try:
            action, side = normalize_lock_event(dict(new_state.attributes), mapping)
            try:
                event_time = datetime.fromisoformat(new_state.state)
                if event_time.tzinfo is None:
                    raise ValueError
            except ValueError as exc:
                raise DoorCycleError("invalid_door_event_timestamp") from exc
        except DoorCycleError as exc:
            _LOGGER.error("Rejected door event: %s", exc)
            if error_callback is not None:
                error_callback(exc)
            return
        last_event = new_state.state
        await callback(action, side, event_time.timestamp())

    return async_track_state_change_event(hass, [entity_id], handle_event)


def contact_is_open(state: str) -> bool | None:
    """Interpret a standard binary_sensor state without inference."""
    if state == "on":
        return True
    if state == "off":
        return False
    return None


def doorbell_message(
    entry_id: str,
    source_id: str,
    occurred_at: float,
    camera: str | None = None,
) -> IngressMessage:
    """Create context-only doorbell ingress."""
    return IngressMessage(
        kind=IngressKind.DOORBELL,
        entry_id=entry_id,
        source_id=source_id,
        occurred_at=occurred_at,
        camera=camera,
    )


class DoorCycleCoordinator:
    """Create one stable activity between open and close."""

    def __init__(
        self,
        store: ActivityStore,
        *,
        entry_id: str,
        camera: str,
        processing_mode: ProcessingMode,
        active_id: str | None = None,
    ) -> None:
        self._store = store
        self._entry_id = entry_id
        self._camera = camera
        self._processing_mode = processing_mode
        if active_id is not None:
            active = store.get(active_id)
            if (
                active is None
                or active.entry_id != entry_id
                or active.camera != camera
                or active.source is not ActivitySource.DOOR_CYCLE
                or active.stage is not ActivityStage.COLLECTING
            ):
                raise DoorCycleError("invalid_active_cycle")
        self._active_id = active_id
        self._cycle_lock = asyncio.Lock()

    async def async_open(self, occurred_at: float, side: str) -> DoorResult:
        async with self._cycle_lock:
            return await self._async_open_locked(occurred_at, side)

    async def _async_open_locked(self, occurred_at: float, side: str) -> DoorResult:
        if self._active_id is not None:
            existing = self._store.get(self._active_id)
            if existing is None:
                raise DoorCycleError("active_cycle_missing")
            return DoorResult("duplicate_open", existing)
        activity_id = door_activity_id(self._entry_id, occurred_at)
        record = ActivityRecord(
            activity_id=activity_id,
            entry_id=self._entry_id,
            source=ActivitySource.DOOR_CYCLE,
            stage=ActivityStage.COLLECTING,
            processing_mode=self._processing_mode,
            created_at=occurred_at,
            updated_at=occurred_at,
            camera=self._camera,
            opening_side=side,
        )
        record = await self._store.async_create(record)
        self._active_id = record.activity_id
        return DoorResult("opened", record)

    async def async_close(self, occurred_at: float) -> DoorResult:
        async with self._cycle_lock:
            return await self._async_close_locked(occurred_at)

    async def _async_close_locked(self, occurred_at: float) -> DoorResult:
        if self._active_id is None:
            raise DoorCycleError("close_without_open")
        activity_id = self._active_id
        existing = self._store.get(activity_id)
        if existing is None:
            raise DoorCycleError("active_cycle_missing")
        door_remained_open = (
            not existing.contact_reopened if existing.contact_seen_open else None
        )
        record = await self._store.async_transition(
            activity_id,
            ActivityStage.COLLECTING,
            ActivityStage.SEALED,
            updated_at=occurred_at,
            association_deadline=occurred_at + 10,
            finalization_deadline=occurred_at + 120,
            door_closed_at=occurred_at,
            door_remained_open=door_remained_open,
        )
        self._active_id = None
        return DoorResult("closed", record)

    async def async_abandon_stale_open(
        self, occurred_at: float, *, threshold: float
    ) -> DoorResult | None:
        """Fail one cycle whose close event never arrived."""
        async with self._cycle_lock:
            if self._active_id is None:
                return None
            activity_id = self._active_id
            existing = self._store.get(activity_id)
            if existing is None:
                raise DoorCycleError("active_cycle_missing")
            if occurred_at - existing.created_at < threshold:
                return None
            record = await self._store.async_transition(
                activity_id,
                ActivityStage.COLLECTING,
                ActivityStage.FAILED,
                updated_at=occurred_at,
                error_code="door_open_too_long",
            )
            self._active_id = None
            return DoorResult("abandoned", record)

    async def async_contact(
        self, is_open: bool, occurred_at: float
    ) -> ActivityRecord | None:
        """Track optional contact continuity without changing cycle boundaries."""
        if self._active_id is None:
            return None
        return await self._store.async_update_contact(
            self._active_id, is_open=is_open, updated_at=occurred_at
        )

    async def async_doorbell(self, occurred_at: float) -> ActivityRecord | None:
        if self._active_id is None:
            return None
        return await self._store.async_merge_context(
            self._active_id,
            doorbell_at=occurred_at,
            updated_at=occurred_at,
        )
