from __future__ import annotations

import asyncio
from io import BytesIO
from unittest.mock import patch

import pytest
from aiohttp import web
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import (
    async_create_clientsession,
    async_get_clientsession,
)
from PIL import Image

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


async def test_a_404_names_the_call_that_failed(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """A failed request must say WHICH request failed.

    Diagnosed from a real deployment: an activity died with the bare code
    `http_404` -- no URL, no operation -- while every Frigate endpoint answered
    200 when probed by hand, so the one fact needed to locate the fault was the
    one fact the error discarded.

    `str(exc)` must stay exactly the code: callers compare it against
    `TRANSIENT_MEDIA_ERRORS`, pass it to `record_error`, and store it as an
    activity's `error_code`, so decorating the message would silently
    reclassify every failure.
    """
    app = web.Application()

    async def missing(request: web.Request) -> web.Response:
        return web.Response(status=404, text="not found")

    app.router.add_get("/api/front/recordings/1.500000/snapshot.jpg", missing)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    with pytest.raises(FrigateApiError) as caught:
        await client.async_get_snapshot("front", 1.5, 360)

    error = caught.value
    assert str(error) == "http_404", (
        "错误码必须保持原样：调用方拿它和 TRANSIENT_MEDIA_ERRORS 比较、"
        "写进 record_error、存成活动的 error_code"
    )
    assert error.url is not None, "错误必须带上失败的那个 URL，否则无法定位"
    assert "/api/front/recordings/1.500000/snapshot.jpg" in error.url


def _webp_bytes() -> bytes:
    """A real 320x180 WebP, as Frigate's preview route serves it."""
    buffer = BytesIO()
    Image.new("RGB", (320, 180), (40, 90, 160)).save(buffer, "WEBP")
    return buffer.getvalue()


async def test_preview_frames_returns_the_names_frigate_listed(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """The list endpoint answers with filenames, not URLs or timestamps."""
    app = web.Application()
    names = [
        "preview_front-1790870400.123521.webp",
        "preview_front-1790870401.5.webp",
    ]

    async def frames(request: web.Request) -> web.Response:
        # The route is literal, so there is no match_info to assert against;
        # the path itself is what proves the client built the right URL.
        assert request.path == "/api/preview/front/start/100/end/200/frames"
        return web.json_response(names)

    app.router.add_get("/api/preview/front/start/100/end/200/frames", frames)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    assert await client.async_get_preview_frames("front", 100, 200) == tuple(names)


async def test_preview_frames_empty_list_is_a_normal_answer(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """An empty array means "nothing cached here", not a failure.

    Frigate drops an hour's frames when the hour rolls over, and it answers an
    unknown camera the same way -- so emptiness must not raise, or the caller's
    fallback would look like an error on every window that crosses the hour.
    """
    app = web.Application()

    async def frames(request: web.Request) -> web.Response:
        return web.json_response([])

    app.router.add_get("/api/preview/front/start/100/end/200/frames", frames)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    assert await client.async_get_preview_frames("front", 100, 200) == ()


async def test_preview_frames_rejects_a_non_list_payload(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """A JSON object is not a frame list, and must not be iterated as one."""
    app = web.Application()

    async def frames(request: web.Request) -> web.Response:
        return web.json_response({"success": False, "message": "No previews found."})

    app.router.add_get("/api/preview/front/start/100/end/200/frames", frames)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    with pytest.raises(FrigateApiError, match="invalid_preview_frames"):
        await client.async_get_preview_frames("front", 100, 200)


async def test_preview_thumbnail_accepts_webp_behind_a_jpg_url(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """The route ends in `.jpg` but serves WebP; the body must survive intact.

    This is the trap the client has to avoid: `async_get_snapshot` rejects
    anything whose declared type is not `image/jpeg`, and reusing that check here
    would reject every preview frame -- silently, because the preview path falls
    back by design and a rejection looks exactly like an empty window.
    """
    app = web.Application()
    body = _webp_bytes()

    async def thumbnail(request: web.Request) -> web.Response:
        return web.Response(body=body, content_type="image/webp")

    app.router.add_get(
        "/api/preview/preview_front-1790870400.123521.webp/thumbnail.jpg", thumbnail
    )
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    returned = await client.async_get_preview_thumbnail(
        "preview_front-1790870400.123521.webp"
    )
    assert returned == body, "WebP 实体必须原样返回，不能被 content-type 校验拒掉"
    assert returned[:4] == b"RIFF" and returned[8:12] == b"WEBP"


async def test_preview_thumbnail_still_accepts_jpeg(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """Both formats are live, so both have to pass.

    `recordings/.../snapshot.jpg` really is JPEG while preview frames are WebP;
    a check written for one alone would break the other.
    """
    app = web.Application()

    async def thumbnail(request: web.Request) -> web.Response:
        return web.Response(body=b"\xff\xd8\xff\xe0jpeg", content_type="image/jpeg")

    app.router.add_get("/api/preview/frame.webp/thumbnail.jpg", thumbnail)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    assert (
        await client.async_get_preview_thumbnail("frame.webp")
        == b"\xff\xd8\xff\xe0jpeg"
    )


async def test_preview_thumbnail_rejects_an_error_page(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """A 200 carrying HTML is not a frame.

    Without a signature check this would be handed to Pillow, which would raise
    somewhere far from the cause -- and on the preview path that failure reads as
    "no frames here", hiding a broken endpoint indefinitely.
    """
    app = web.Application()

    async def thumbnail(request: web.Request) -> web.Response:
        return web.Response(body=b"<html>login</html>", content_type="image/jpeg")

    app.router.add_get("/api/preview/frame.webp/thumbnail.jpg", thumbnail)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    with pytest.raises(FrigateApiError, match="preview_decode_failed"):
        await client.async_get_preview_thumbnail("frame.webp")


async def test_preview_thumbnail_rejects_an_empty_body(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """A 200 with no bytes is not a frame either."""
    app = web.Application()

    async def thumbnail(request: web.Request) -> web.Response:
        return web.Response(body=b"", content_type="image/webp")

    app.router.add_get("/api/preview/frame.webp/thumbnail.jpg", thumbnail)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    with pytest.raises(FrigateApiError, match="empty_preview_frame"):
        await client.async_get_preview_thumbnail("frame.webp")


async def test_preview_frames_rejects_an_inverted_range(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """`end <= start` cannot name a window, so it is refused before any request."""
    app = web.Application()
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    with pytest.raises(FrigateApiError, match="invalid_preview_range"):
        await client.async_get_preview_frames("front", 200, 100)


# The two zones this deployment actually has, copied from `GET /api/config`.
# They are polygons with diagonal edges, not rectangles, which is the property
# the parser has to preserve.
_REAL_HOME_DOOR = [
    0.181, 0.175, 0.407, 0.125, 0.442, 0.677,
    0.456, 0.738, 0.533, 0.981, 0.286, 1.0,
]
_REAL_ELEVATOR = [
    0.411, 0.083, 0.68, 0.048, 0.662, 0.553, 0.713, 0.768,
    0.516, 0.909, 0.471, 0.766, 0.445, 0.672,
]


async def test_camera_zones_parses_real_frigate_polygons(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """Zones come back as vertex lists, not bounding boxes.

    The distinction is load-bearing: Frigate stores `x1,y1,x2,y2,...` with
    diagonal edges, and treating the numbers as a rectangle would crop in a good
    deal of wall for both zones on this deployment.
    """
    app = web.Application()

    async def config(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "cameras": {
                    "front": {
                        "zones": {
                            "home_door": {"coordinates": _REAL_HOME_DOOR},
                            "elevator": {"coordinates": _REAL_ELEVATOR},
                        }
                    }
                }
            }
        )

    app.router.add_get("/api/config", config)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    zones = await client.async_get_camera_zones("front")
    assert set(zones) == {"home_door", "elevator"}
    assert len(zones["home_door"]) == 6
    assert len(zones["elevator"]) == 7
    assert zones["home_door"][0] == (0.181, 0.175)
    assert zones["elevator"][-1] == (0.445, 0.672)


async def test_camera_zones_skips_a_malformed_zone_not_the_camera(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """One bad polygon must not cost the user every other region.

    A zone whose coordinates are odd-length or out of range cannot be used, but
    the zones beside it can -- so the good ones are returned and the bad one is
    dropped, rather than failing the call and leaving the form empty.
    """
    app = web.Application()

    async def config(request: web.Request) -> web.Response:
        return web.json_response(
            {
                "cameras": {
                    "front": {
                        "zones": {
                            "good": {"coordinates": _REAL_HOME_DOOR},
                            "odd_length": {"coordinates": [0.1, 0.2, 0.3]},
                            "two_vertices": {"coordinates": [0.1, 0.2, 0.3, 0.4]},
                            "out_of_frame": {
                                "coordinates": [0.1, 0.2, 2.0, 0.4, 0.5, 0.6]
                            },
                            "not_a_list": {"coordinates": "nope"},
                        }
                    }
                }
            }
        )

    app.router.add_get("/api/config", config)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    zones = await client.async_get_camera_zones("front")
    assert set(zones) == {"good"}, "畸形 zone 必须被跳过，而不是让整次调用失败"


async def test_camera_zones_empty_for_a_camera_without_any(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """"This camera has no zones" is an empty dict, not an error.

    The caller distinguishes it from "Frigate did not answer", which raises.
    """
    app = web.Application()

    async def config(request: web.Request) -> web.Response:
        return web.json_response({"cameras": {"front": {}}})

    app.router.add_get("/api/config", config)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    assert await client.async_get_camera_zones("front") == {}


async def test_camera_zones_raises_for_an_unknown_camera(
    hass: HomeAssistant, aiohttp_server, socket_enabled
) -> None:
    """A camera Frigate does not have is a configuration error, not emptiness."""
    app = web.Application()

    async def config(request: web.Request) -> web.Response:
        return web.json_response({"cameras": {"other": {}}})

    app.router.add_get("/api/config", config)
    server = await aiohttp_server(app)
    client = FrigateClient(async_get_clientsession(hass), str(server.make_url("/")))

    with pytest.raises(FrigateApiError, match="unknown_camera"):
        await client.async_get_camera_zones("front")
