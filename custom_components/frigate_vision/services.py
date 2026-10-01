"""Native Home Assistant services."""

from __future__ import annotations

import time

import voluptuous as vol
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)

from .const import DOMAIN
from .models import IngressKind, IngressMessage
from .runtime import IntegrationRuntime


def _runtime(
    hass: HomeAssistant, entry_id: str
) -> tuple[ConfigEntry, IntegrationRuntime]:
    entry = hass.config_entries.async_get_entry(entry_id)
    if entry is None or not isinstance(entry.runtime_data, IntegrationRuntime):
        raise ValueError("entry_not_loaded")
    return entry, entry.runtime_data


async def _require_admin(hass: HomeAssistant, call: ServiceCall) -> None:
    """Refuse a call from anything that is not an administrator.

    Two of these services spend the owner's provider credits (`retry_failed`,
    `process_review` enqueue a real analysis), and two return prose about who was at
    the camera (`get_activity`) or change a delivery's state (`ack_delivery`). None of
    that should be available to every logged-in account.

    Home Assistant does not enforce this for us: the `call_service` WebSocket command
    carries no `require_admin`, and the entity permission check in
    `helpers/service.py` only engages for services that target entities -- these
    target an `entry_id`, and `services.yaml` declares no entity selector, so nothing
    runs. The check has to live here. `context.user_id` is the only caller identity a
    service receives.

    A call with no `user_id` is allowed: that is an automation, a script, or the
    integration itself, none of which has a user and all of which the owner already
    controls. Refusing those would break the shipped blueprint, which calls
    `ack_delivery` from an automation.

    `async_get_user` is awaited rather than called: it is a coroutine, so a missing
    `await` would return a truthy coroutine object and admit every caller while
    looking like a check.
    """
    user_id = call.context.user_id
    if user_id is None:
        return
    user = await hass.auth.async_get_user(user_id)
    if user is None or not user.is_admin:
        # An unknown or deleted user id cannot be shown to be an administrator, so it
        # is refused rather than allowed: the default must be closed.
        raise ValueError("admin_required")


async def async_register_services(hass: HomeAssistant) -> None:
    if hass.services.has_service(DOMAIN, "get_activity"):
        return

    async def get_activity(call: ServiceCall) -> ServiceResponse:
        await _require_admin(hass, call)
        _, runtime = _runtime(hass, call.data["entry_id"])
        record = runtime.store.get(call.data["activity_id"])
        if record is None:
            raise ValueError("activity_missing")
        return {
            "activity_id": record.activity_id,
            "stage": record.stage.value,
            "classification": record.classification,
            "description": record.description,
            "confidence": record.confidence,
            "error_code": record.error_code,
            # Which provider answered, when the entry has one configured. `None`
            # before the activity is analysed -- deliberately not defaulted to
            # "primary", because a guess here would be indistinguishable from a
            # measurement and this field exists precisely to be trusted.
            "provider": (
                runtime.vision.provider_for(record.activity_id)
                if runtime.vision is not None
                else None
            ),
            "evidence_url": (
                None
                if record.evidence_expired_at is not None
                else record.evidence_media_url
            ),
            "evidence_expired": record.evidence_expired_at is not None,
            "review_ids": list(record.review_ids),
        }

    async def retry_failed(call: ServiceCall) -> None:
        await _require_admin(hass, call)
        entry, runtime = _runtime(hass, call.data["entry_id"])
        retry = await runtime.store.async_create_retry(
            call.data["activity_id"], now=time.time()
        )
        runtime.async_schedule_record(entry, retry)

    async def process_review(call: ServiceCall) -> None:
        await _require_admin(hass, call)
        entry, runtime = _runtime(hass, call.data["entry_id"])
        if runtime.frigate_client is None or runtime.correlation is None:
            raise ValueError("frigate_not_configured")
        review_id = call.data["review_id"]
        if any(review_id in record.review_ids for record in runtime.store.all()):
            raise ValueError("review_already_processed")
        camera = runtime.correlation.camera
        payload = await runtime.frigate_client.async_get_review(review_id, camera)
        data = payload["data"]
        if "person" not in data["objects"]:
            raise ValueError("review_not_person")
        analyze_all_person_reviews = bool(
            entry.options.get("analyze_all_far_reviews", True)
        )
        zones = tuple(zone for zone in data["zones"] if zone in runtime.allowed_zones)
        if not zones and not analyze_all_person_reviews:
            raise ValueError("review_zone_missing")
        runtime.queue.enqueue(
            IngressMessage(
                kind=IngressKind.FRIGATE_REVIEW,
                entry_id=call.data["entry_id"],
                source_id=review_id,
                review_id=review_id,
                started_at=float(payload["start_time"]),
                occurred_at=float(payload["end_time"]),
                camera=camera,
                current_zones=zones,
                detection_ids=tuple(data["detections"]),
                manual=True,
            )
        )

    async def ack_delivery(call: ServiceCall) -> None:
        await _require_admin(hass, call)
        _, runtime = _runtime(hass, call.data["entry_id"])
        if runtime.delivery is None:
            raise ValueError("delivery_not_configured")
        await runtime.delivery.async_ack(
            call.data["activity_id"], call.data["delivery_attempt_id"]
        )

    common = vol.Schema({vol.Required("entry_id"): str})
    hass.services.async_register(
        DOMAIN,
        "get_activity",
        get_activity,
        schema=common.extend({vol.Required("activity_id"): str}),
        supports_response=SupportsResponse.ONLY,
    )
    hass.services.async_register(
        DOMAIN,
        "retry_failed",
        retry_failed,
        schema=common.extend({vol.Required("activity_id"): str}),
    )
    hass.services.async_register(
        DOMAIN,
        "process_review",
        process_review,
        schema=common.extend({vol.Required("review_id"): str}),
    )
    hass.services.async_register(
        DOMAIN,
        "ack_delivery",
        ack_delivery,
        schema=common.extend(
            {
                vol.Required("activity_id"): str,
                vol.Required("delivery_attempt_id"): str,
            }
        ),
    )
