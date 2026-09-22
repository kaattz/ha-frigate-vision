"""Strict one-shot delivery protocol."""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from urllib.parse import urlencode

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant

from .clip_proxy import ClipUrlError, hls_path_for, signed_clip_url_for
from .models import ActivityRecord, ActivityStage
from .repairs import async_set_issue
from .store import ActivityStore

_LOGGER = logging.getLogger(__name__)

EVENT_ACTIVITY = "frigate_vision_activity"


class DeliveryManager:
    def __init__(
        self,
        hass: HomeAssistant,
        entry: ConfigEntry,
        store: ActivityStore,
        *,
        ack_timeout: float = 60,
    ) -> None:
        self._hass = hass
        self._entry = entry
        self._store = store
        self._ack_timeout = ack_timeout
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stopping = False

    async def async_start(self, activity_id: str) -> ActivityRecord:
        if self._stopping:
            raise RuntimeError("delivery_stopping")
        attempt_id = uuid.uuid4().hex
        record = await self._store.async_start_delivery(
            activity_id, attempt_id=attempt_id, updated_at=time.time()
        )
        frigate = self._entry.data.get("frigate", {})
        base_url = (
            str(frigate.get("base_url", "")).rstrip("/")
            if isinstance(frigate, dict)
            else ""
        )
        review_url = (
            f"{base_url}/review?{urlencode({'id': record.review_ids[0]})}"
            if base_url and record.review_ids
            else None
        )
        # The clip is served through Home Assistant, not Frigate, so the link
        # works away from home. Frigate's own URL would only resolve on the home
        # network, which is how the previous video-based automation behaved.
        #
        # Signed rather than bare: the link is opened from a notification, often
        # on a phone with no session in that browser, and a bare `/api/` URL is
        # refused with 401 there -- measured on this deployment, every tap was
        # rejected as `invalid authentication`.
        clip_url = signed_clip_url_for(
            self._hass,
            str(self._hass.config.external_url or ""),
            record.entry_id,
            record,
        )
        # The in-app player cannot use `clip_url`: every MP4 endpoint here
        # answers without `Accept-Ranges`, and a mobile browser will not stream
        # a video it cannot range-request. HLS is segmented, so this is the
        # playable one.
        #
        # Degraded rather than raised: delivery is the product and the play
        # link is an enhancement, so a camera name the URL builder rejects must
        # not cost the user their notification.
        try:
            hls_url: str | None = hls_path_for(record.camera, record)
        except ClipUrlError:
            hls_url = None
            _LOGGER.warning(
                "No playable URL for %s: camera %r is not a valid path segment",
                record.activity_id,
                record.camera,
            )
        self._hass.bus.async_fire(
            EVENT_ACTIVITY,
            {
                "entry_id": record.entry_id,
                "activity_id": record.activity_id,
                "delivery_attempt_id": attempt_id,
                "classification": record.classification,
                "description": record.description,
                "confidence": record.confidence,
                "evidence_url": record.evidence_media_url,
                "clip_url": clip_url,
                "hls_url": hls_url,
                "frigate_review_url": review_url,
                "review_ids": list(record.review_ids),
                "occurred_at": record.created_at,
            },
        )
        task = self._entry.async_create_background_task(
            self._hass,
            self._timeout(record.activity_id, attempt_id),
            f"{self._entry.domain}-{record.activity_id}-delivery-timeout",
        )
        self._tasks[record.activity_id] = task
        task.add_done_callback(lambda _: self._tasks.pop(record.activity_id, None))
        return record

    async def async_ack(self, activity_id: str, attempt_id: str) -> ActivityRecord:
        record = await self._store.async_ack_delivery(
            activity_id, attempt_id=attempt_id, updated_at=time.time()
        )
        task = self._tasks.pop(activity_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return record

    async def _timeout(self, activity_id: str, attempt_id: str) -> None:
        await asyncio.sleep(self._ack_timeout)
        current = self._store.get(activity_id)
        if current is None or current.stage is not ActivityStage.DELIVERY_STARTED:
            return
        if current.delivery_attempt_id != attempt_id:
            return
        while not self._stopping:
            try:
                await self._store.async_transition(
                    activity_id,
                    ActivityStage.DELIVERY_STARTED,
                    ActivityStage.FAILED,
                    updated_at=time.time(),
                    error_code="delivery_outcome_unknown",
                )
                async_set_issue(
                    self._hass,
                    self._entry.entry_id,
                    "delivery_outcome_unknown",
                )
                return
            except OSError:
                await asyncio.sleep(1)
            except Exception:
                return

    async def async_stop(self) -> None:
        self._stopping = True
        tasks = tuple(self._tasks.values())
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
