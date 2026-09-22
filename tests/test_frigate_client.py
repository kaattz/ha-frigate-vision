from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from aiohttp import web
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import (
    async_create_clientsession,
    async_get_clientsession,
)

from custom_components.frigate_vision.frigate import (
    FrigateApiError,
    FrigateClient,
)


async def test_frigate_client_validates_json_and_jpeg(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def version(request: web.Request) -> web.Response:
        return web.Response(text="0.17.2")

    async def review(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "review_1",
                "camera": "front",
                "start_time": 1,
                "end_time": 2,
                "data": {
                    "detections": ["event_1"],
                    "objects": ["person"],
                    "zones": ["near"],
                },
            }
        )

    async def event(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "event_1",
                "camera": "front",
                "label": "person",
                "start_time": 1,
                "end_time": 2,
            }
        )

    async def recordings(request: web.Request) -> web.Response:
        return web.json_response([{"start_time": 1, "end_time": 2}])

    async def snapshot(request: web.Request) -> web.Response:
        return web.Response(body=b"jpeg", content_type="image/jpeg")

    app.router.add_get("/api/version", version)
    app.router.add_get("/api/review/review_1", review)
    app.router.add_get("/api/events/event_1", event)
    app.router.add_get("/api/front/recordings", recordings)
    app.router.add_get("/api/front/recordings/1.500000/snapshot.jpg", snapshot)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))
    assert await client.async_get_version() == "0.17.2"
    assert (await client.async_get_review("review_1", "front"))["id"] == "review_1"
    assert (await client.async_get_event("event_1", "front"))["label"] == "person"
    assert len(await client.async_get_recordings("front", 1, 2)) == 1
    assert await client.async_get_snapshot("front", 1.5, 360) == b"jpeg"


async def test_frigate_client_rejects_wrong_content_type(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def invalid_snapshot(request: web.Request) -> web.Response:
        return web.Response(text="not jpeg", content_type="text/plain")

    app.router.add_get("/api/front/recordings/1.500000/snapshot.jpg", invalid_snapshot)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))
    with pytest.raises(FrigateApiError, match="unexpected_content_type"):
        await client.async_get_snapshot("front", 1.5, 360)


async def test_frigate_client_rejects_identity_mismatch(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def wrong_review(request: web.Request) -> web.Response:
        return web.json_response({"id": "other", "camera": "front"})

    app.router.add_get("/api/review/review_1", wrong_review)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))
    with pytest.raises(FrigateApiError, match="review_identity_mismatch"):
        await client.async_get_review("review_1", "front")


async def test_frigate_client_rejects_malformed_review_data(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def malformed_review(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "review_1",
                "camera": "front",
                "start_time": 1,
                "end_time": 2,
                "data": {
                    "detections": "event_1",
                    "objects": ["person"],
                    "zones": ["near"],
                },
            }
        )

    app.router.add_get("/api/review/review_1", malformed_review)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))
    with pytest.raises(FrigateApiError, match="invalid_review"):
        await client.async_get_review("review_1", "front")


async def test_frigate_client_rejects_invalid_review_event_and_recording_times(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def review(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "review_1",
                "camera": "front",
                "start_time": 3,
                "end_time": 2,
                "data": {"detections": ["event_1"]},
            }
        )

    async def event(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "id": "event_1",
                "camera": "front",
                "label": "person",
                "start_time": "bad",
                "end_time": 2,
            }
        )

    async def recordings(request: web.Request) -> web.Response:
        return web.json_response([{"start_time": 4, "end_time": 3}])

    app.router.add_get("/api/review/review_1", review)
    app.router.add_get("/api/events/event_1", event)
    app.router.add_get("/api/front/recordings", recordings)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))
    with pytest.raises(FrigateApiError, match="invalid_review_time"):
        await client.async_get_review("review_1", "front")
    with pytest.raises(FrigateApiError, match="invalid_event_time"):
        await client.async_get_event("event_1", "front")
    with pytest.raises(FrigateApiError, match="invalid_recordings"):
        await client.async_get_recordings("front", 1, 5)


async def test_frigate_client_uses_explicit_timeout(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def slow_version(request: web.Request) -> web.Response:
        await asyncio.sleep(0.05)
        return web.Response(text="0.17.2")

    app.router.add_get("/api/version", slow_version)
    server = await aiohttp_server(app)
    client = FrigateClient(
        async_get_clientsession(hass),
        str(server.make_url("/")),
        request_timeout=0.01,
    )
    with pytest.raises(FrigateApiError, match="request_timeout"):
        await client.async_get_version()


async def test_native_client_validates_auth_after_login(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    calls: list[str] = []
    app = web.Application()

    async def login(request: web.Request) -> web.Response:
        calls.append("login")
        response = web.json_response({"success": True})
        response.set_cookie("frigate_token", "token")
        return response

    async def auth(request: web.Request) -> web.Response:
        calls.append("auth")
        if request.cookies.get("frigate_token") != "token":
            raise web.HTTPUnauthorized
        return web.Response(text="ok")

    app.router.add_post("/api/login", login)
    app.router.add_get("/auth", auth)
    server = await aiohttp_server(app)
    with patch(
        "custom_components.frigate_vision.frigate.async_create_clientsession",
        wraps=async_create_clientsession,
    ) as create_session:
        client = await FrigateClient.async_create(
            hass,
            {
                "base_url": str(server.make_url("/")),
                "auth_mode": "native",
                "username": "user",
                "password": "pass",
            },
        )
    create_session.assert_called_once()
    assert calls == ["login", "auth"]
    await client.async_close()


async def test_frigate_client_maps_http_and_non_json_errors(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    app = web.Application()

    async def missing(request: web.Request) -> web.Response:
        raise web.HTTPNotFound

    async def non_json(request: web.Request) -> web.Response:
        return web.Response(text="no", content_type="text/plain")

    app.router.add_get("/api/review/missing", missing)
    app.router.add_get("/api/events/event_1", non_json)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))
    with pytest.raises(FrigateApiError, match="http_404"):
        await client.async_get_review("missing", "front")
    with pytest.raises(FrigateApiError, match="unexpected_content_type"):
        await client.async_get_event("event_1", "front")


async def test_native_client_reauthenticates_once_after_expired_session(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    login_count = 0
    version_count = 0
    app = web.Application()

    async def login(request: web.Request) -> web.Response:
        nonlocal login_count
        login_count += 1
        response = web.json_response({"success": True})
        response.set_cookie("frigate_token", f"token_{login_count}")
        return response

    async def auth(request: web.Request) -> web.Response:
        if request.cookies.get("frigate_token") != f"token_{login_count}":
            raise web.HTTPUnauthorized
        return web.Response(text="ok")

    async def version(request: web.Request) -> web.Response:
        nonlocal version_count
        version_count += 1
        if version_count == 1:
            response = web.Response(status=401)
            response.del_cookie("frigate_token")
            return response
        return web.Response(text="0.17.2")

    app.router.add_post("/api/login", login)
    app.router.add_get("/auth", auth)
    app.router.add_get("/api/version", version)
    server = await aiohttp_server(app)
    client = await FrigateClient.async_create(
        hass,
        {
            "base_url": str(server.make_url("/")),
            "auth_mode": "native",
            "username": "user",
            "password": "pass",
        },
    )
    assert await client.async_get_version() == "0.17.2"
    assert login_count == 2
    assert version_count == 2
    await client.async_close()
