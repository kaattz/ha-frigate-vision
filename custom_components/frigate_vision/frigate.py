"""Frigate HTTP client and MQTT ingress parsing."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import quote

from aiohttp import ClientError, ClientSession, ClientTimeout, CookieJar
from homeassistant.components.mqtt.client import async_subscribe
from homeassistant.components.mqtt.models import ReceiveMessage
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.aiohttp_client import (
    async_create_clientsession,
    async_get_clientsession,
)

from .models import SAFE_ID, IngressKind, IngressMessage, ModelValidationError


class FrigateApiError(RuntimeError):
    """A Frigate API response violated the public contract."""


class FrigatePayloadError(ValueError):
    """A Frigate MQTT payload is invalid."""


class FrigateClient:
    """Strict subset of the Frigate HTTP API."""

    def __init__(
        self,
        session: ClientSession,
        base_url: str,
        *,
        request_timeout: float = 10,
        native_credentials: tuple[str, str] | None = None,
    ) -> None:
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("invalid_request_timeout")
        self._session = session
        self._base_url = base_url.rstrip("/")
        self._owned_session = False
        self._timeout = ClientTimeout(total=request_timeout)
        self._native_credentials = native_credentials
        self._last_auth_check = 0.0

    @classmethod
    async def async_create(
        cls, hass: HomeAssistant, config: Mapping[str, Any]
    ) -> FrigateClient:
        base_url = str(config["base_url"])
        if config.get("auth_mode") != "native":
            return cls(async_get_clientsession(hass), base_url)
        session = async_create_clientsession(
            hass,
            auto_cleanup=False,
            cookie_jar=CookieJar(unsafe=True),
        )
        client = cls(
            session,
            base_url,
            native_credentials=(
                str(config.get("username", "")),
                str(config.get("password", "")),
            ),
        )
        client._owned_session = True
        try:
            await client._async_login()
            await client._async_validate_auth()
        except Exception:
            session.detach()
            raise
        return client

    async def async_close(self) -> None:
        if self._owned_session:
            self._session.detach()

    async def async_get_version(self) -> str:
        _, data = await self._request_bytes("GET", f"{self._base_url}/api/version")
        version = data.decode("utf-8", errors="strict").strip()
        if not version:
            raise FrigateApiError("invalid_version")
        return version

    async def async_get_review(self, review_id: str, camera: str) -> dict[str, Any]:
        payload = await self._get_json(f"review/{quote(review_id, safe='')}")
        if payload.get("id") != review_id or payload.get("camera") != camera:
            raise FrigateApiError("review_identity_mismatch")
        self._validate_time_window(payload, "invalid_review_time")
        data = payload.get("data")
        if not isinstance(data, dict):
            raise FrigateApiError("invalid_review")
        detections = data.get("detections")
        objects = data.get("objects")
        zones = data.get("zones")
        if (
            not isinstance(detections, list)
            or not detections
            or any(
                not isinstance(item, str) or not SAFE_ID.fullmatch(item)
                for item in detections
            )
            or not isinstance(objects, list)
            or any(not isinstance(item, str) for item in objects)
            or not isinstance(zones, list)
            or any(not isinstance(item, str) for item in zones)
        ):
            raise FrigateApiError("invalid_review")
        return payload

    async def async_get_event(self, event_id: str, camera: str) -> dict[str, Any]:
        payload = await self._get_json(f"events/{quote(event_id, safe='')}")
        if payload.get("id") != event_id or payload.get("camera") != camera:
            raise FrigateApiError("event_identity_mismatch")
        if payload.get("label") != "person":
            raise FrigateApiError("event_not_person")
        if payload.get("end_time") is None:
            raise FrigateApiError("event_in_progress")
        self._validate_time_window(payload, "invalid_event_time")
        return payload

    async def async_get_recordings(
        self, camera: str, after: float, before: float
    ) -> list[dict[str, Any]]:
        if (
            not math.isfinite(after)
            or not math.isfinite(before)
            or after < 0
            or before <= after
        ):
            raise FrigateApiError("invalid_recording_range")
        payload = await self._request_json(
            "GET",
            f"{self._base_url}/api/{quote(camera, safe='')}/recordings",
            params={"after": str(after), "before": str(before)},
        )
        if not isinstance(payload, list) or not all(
            isinstance(item, dict) for item in payload
        ):
            raise FrigateApiError("invalid_recordings")
        for item in payload:
            self._validate_time_window(item, "invalid_recordings")
            if item["end_time"] < after or item["start_time"] > before:
                raise FrigateApiError("invalid_recordings")
        return payload

    async def async_get_snapshot(
        self, camera: str, timestamp: float, height: int
    ) -> bytes:
        if not math.isfinite(timestamp) or timestamp < 0 or height < 1 or height > 4320:
            raise FrigateApiError("invalid_snapshot_request")
        content_type, data = await self._request_bytes(
            "GET",
            f"{self._base_url}/api/{quote(camera, safe='')}/recordings/"
            f"{timestamp:.6f}/snapshot.jpg",
            params={"height": str(height)},
        )
        if content_type != "image/jpeg":
            raise FrigateApiError("unexpected_content_type")
        if not data:
            raise FrigateApiError("empty_snapshot")
        return data

    async def _get_json(self, path: str) -> dict[str, Any]:
        payload = await self._request_json("GET", f"{self._base_url}/api/{path}")
        if not isinstance(payload, dict):
            raise FrigateApiError("invalid_json_object")
        return payload

    async def _request_json(self, method: str, url: str, **kwargs: Any) -> Any:
        content_type, data = await self._request_bytes(method, url, **kwargs)
        if content_type != "application/json":
            raise FrigateApiError("unexpected_content_type")
        try:
            return json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
            raise FrigateApiError("invalid_json") from exc

    async def _request_bytes(
        self, method: str, url: str, **kwargs: Any
    ) -> tuple[str, bytes]:
        await self._async_ensure_auth()
        for attempt in range(2):
            try:
                async with self._session.request(
                    method, url, timeout=self._timeout, **kwargs
                ) as response:
                    if (
                        response.status in {401, 403}
                        and self._native_credentials is not None
                        and attempt == 0
                    ):
                        retry_auth = True
                    else:
                        retry_auth = False
                        self._check_status(response.status)
                        return response.content_type, await response.read()
            except TimeoutError as exc:
                raise FrigateApiError("request_timeout") from exc
            except ClientError as exc:
                raise FrigateApiError("frigate_unavailable") from exc
            if retry_auth:
                await self._async_restore_auth()
        raise FrigateApiError("authentication_failed")

    async def _async_ensure_auth(self) -> None:
        if (
            self._native_credentials is not None
            and time.monotonic() - self._last_auth_check >= 60
        ):
            await self._async_restore_auth()

    async def _async_restore_auth(self) -> None:
        try:
            await self._async_validate_auth()
        except FrigateApiError as exc:
            if str(exc) != "authentication_failed":
                raise
            await self._async_login()
            await self._async_validate_auth()

    async def _async_login(self) -> None:
        if self._native_credentials is None:
            raise FrigateApiError("authentication_failed")
        username, password = self._native_credentials
        try:
            async with self._session.post(
                f"{self._base_url}/api/login",
                json={"user": username, "password": password},
                timeout=self._timeout,
            ) as response:
                self._check_status(response.status)
        except TimeoutError as exc:
            raise FrigateApiError("request_timeout") from exc
        except ClientError as exc:
            raise FrigateApiError("frigate_unavailable") from exc

    async def _async_validate_auth(self) -> None:
        try:
            async with self._session.get(
                f"{self._base_url}/auth", timeout=self._timeout
            ) as response:
                self._check_status(response.status)
        except TimeoutError as exc:
            raise FrigateApiError("request_timeout") from exc
        except ClientError as exc:
            raise FrigateApiError("frigate_unavailable") from exc
        self._last_auth_check = time.monotonic()

    @staticmethod
    def _validate_time_window(payload: Mapping[str, Any], code: str) -> None:
        start = payload.get("start_time")
        end = payload.get("end_time")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(float(start))
            or not math.isfinite(float(end))
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise FrigateApiError(code)

    @staticmethod
    def _check_status(status: int) -> None:
        if status == 401 or status == 403:
            raise FrigateApiError("authentication_failed")
        if status < 200 or status >= 300:
            raise FrigateApiError(f"http_{status}")


def _load_payload(payload: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise FrigatePayloadError("invalid_json") from exc
    if not isinstance(value, dict):
        raise FrigatePayloadError("invalid_payload")
    return value


def parse_event_payload(
    payload: str, *, entry_id: str, camera: str, allowed_zones: set[str]
) -> IngressMessage | None:
    value = _load_payload(payload)
    after = value.get("after")
    if value.get("type") not in {"new", "update", "end"} or not isinstance(after, dict):
        raise FrigatePayloadError("invalid_event_payload")
    if after.get("camera") != camera or after.get("label") != "person":
        return None
    try:
        current_zones = after["current_zones"]
        entered_zones = after["entered_zones"]
        if not isinstance(current_zones, list) or not isinstance(entered_zones, list):
            raise TypeError
        current = tuple(zone for zone in current_zones if zone in allowed_zones)
        entered = tuple(zone for zone in entered_zones if zone in allowed_zones)
        return IngressMessage(
            kind=IngressKind.FRIGATE_EVENT,
            entry_id=entry_id,
            source_id=str(after["id"]),
            event_id=str(after["id"]),
            event_type=str(value["type"]),
            occurred_at=float(after["frame_time"]),
            camera=camera,
            current_zones=current,
            entered_zones=entered,
        )
    except (KeyError, TypeError, ValueError, ModelValidationError) as exc:
        raise FrigatePayloadError("invalid_event_payload") from exc


def parse_review_payload(
    payload: str,
    *,
    entry_id: str,
    camera: str,
    allowed_zones: set[str],
    analyze_all_person_reviews: bool = False,
) -> IngressMessage | None:
    value = _load_payload(payload)
    after = value.get("after")
    if value.get("type") != "end" or not isinstance(after, dict):
        return None
    data = after.get("data")
    if after.get("camera") != camera or not isinstance(data, dict):
        return None
    objects = data.get("objects")
    review_zones = data.get("zones")
    detections_value = data.get("detections")
    if (
        not isinstance(objects, list)
        or not isinstance(review_zones, list)
        or not isinstance(detections_value, list)
        or any(not isinstance(item, str) for item in objects)
        or any(
            not isinstance(item, str) or not SAFE_ID.fullmatch(item)
            for item in review_zones
        )
        or any(
            not isinstance(item, str) or not SAFE_ID.fullmatch(item)
            for item in detections_value
        )
    ):
        raise FrigatePayloadError("invalid_review_payload")
    if "person" not in objects:
        return None
    zones = tuple(zone for zone in review_zones if zone in allowed_zones)
    if not zones and not analyze_all_person_reviews:
        return None
    try:
        detections = tuple(str(item) for item in detections_value)
        if not detections:
            raise ValueError
        return IngressMessage(
            kind=IngressKind.FRIGATE_REVIEW,
            entry_id=entry_id,
            source_id=str(after["id"]),
            review_id=str(after["id"]),
            started_at=float(after["start_time"]),
            occurred_at=float(after["end_time"]),
            camera=camera,
            current_zones=zones,
            detection_ids=detections,
        )
    except (KeyError, TypeError, ValueError, ModelValidationError) as exc:
        raise FrigatePayloadError("invalid_review_payload") from exc


async def async_subscribe_frigate(
    hass: HomeAssistant,
    topic_prefix: str,
    event_callback: Callable[[str], None],
    review_callback: Callable[[str], None],
) -> CALLBACK_TYPE:
    def payload_text(message: ReceiveMessage) -> str:
        payload = message.payload
        if isinstance(payload, str):
            return payload
        try:
            return bytes(payload).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FrigatePayloadError("invalid_utf8") from exc

    @callback
    def event_message(message: ReceiveMessage) -> None:
        event_callback(payload_text(message))

    @callback
    def review_message(message: ReceiveMessage) -> None:
        review_callback(payload_text(message))

    unsubscribe_event = await async_subscribe(
        hass, f"{topic_prefix}/events", event_message
    )
    try:
        unsubscribe_review = await async_subscribe(
            hass, f"{topic_prefix}/reviews", review_message
        )
    except Exception:
        unsubscribe_event()
        raise

    def unsubscribe() -> None:
        unsubscribe_event()
        unsubscribe_review()

    return unsubscribe
