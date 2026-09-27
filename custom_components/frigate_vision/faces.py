"""Ask an optional OpenCV service which frames show a face.

The person close-up picks the largest detection box, which answers "where is most
of the person" and not "which frame shows their face". On the activity that
prompted this, it picked a frame of the person's back while a clear frontal frame
sat four seconds earlier in the same clip.

This module talks to a separate container that runs OpenCV. It is separate because
it *has* to be: the Home Assistant container is Alpine (musl libc) and every
OpenCV wheel on PyPI is built for glibc, so `pip install opencv-python-headless`
there fails with "no matching distribution" -- verified. A glibc image installs
the same wheel in seconds.

Everything here is optional by construction. The answer is a plain mapping, an
unreachable service yields an empty one, and the caller's existing largest-box
rule stands when it does.
"""

from __future__ import annotations

import base64
import io
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from PIL import Image

from .const import FACE_SERVICE_CANDIDATES, FACE_SERVICE_TIMEOUT


class FaceServiceError(RuntimeError):
    """The service could not be asked. Never fatal -- the caller has a fallback."""


def normalise_service_url(url: str) -> str:
    """Return the `/face` endpoint for a configured base, or "" when unset.

    Accepts what a person would type: a bare host, a host with a port, either with
    or without a scheme, and either with or without the `/face` path. Every one of
    those is a reasonable thing to enter, and rejecting three of them would be a
    configuration trap rather than a safeguard.
    """
    text = (url or "").strip()
    if not text:
        return ""
    if "://" not in text:
        text = "http://" + text
    text = text.rstrip("/")
    if text.endswith("/face"):
        return text
    return text + "/face"


def _encode_crop(image_bytes: bytes, box: Sequence[float]) -> bytes | None:
    """Crop the person out of a frame and re-encode it as JPEG.

    The crop is what gets sent, not the whole frame, and that is a measured choice:
    on a 640x360 detect stream a head is about 30 px across and the model missed
    every one of them, while the same frames cropped to the person and sent at the
    same cost were all found. Fewer pixels, more face.

    Returns None when the frame cannot be read or the box is unusable, which the
    caller treats as "unknown" rather than as "no face".
    """
    try:
        x, y, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source.load()
            frame = source.convert("RGB")
    except (OSError, ValueError):
        return None
    frame_width, frame_height = frame.size
    left = max(0, int(x * frame_width))
    top = max(0, int(y * frame_height))
    right = min(frame_width, int((x + width) * frame_width))
    bottom = min(frame_height, int((y + height) * frame_height))
    if right - left < 8 or bottom - top < 8:
        return None
    buffer = io.BytesIO()
    frame.crop((left, top, right, bottom)).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


async def _ask_one(
    session: aiohttp.ClientSession,
    endpoint: str,
    crop: bytes,
    wait_seconds: float,
) -> bool:
    """One face query. Raises FaceServiceError on anything other than a clear answer."""
    payload = json.dumps({"image": base64.b64encode(crop).decode()})
    try:
        async with session.post(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            timeout=aiohttp.ClientTimeout(total=wait_seconds),
        ) as response:
            if response.status != 200:
                raise FaceServiceError(f"http_{response.status}")
            body: Any = json.loads(await response.text())
    except (TimeoutError, aiohttp.ClientError, ValueError) as exc:
        raise FaceServiceError("face_service_unreachable") from exc
    if not isinstance(body, Mapping):
        raise FaceServiceError("face_service_bad_response")
    return bool(body.get("has_face"))


async def faces_for_frames(
    hass: HomeAssistant,
    url: str,
    frames: Sequence[tuple[str, bytes, Sequence[float]]],
    *,
    wait_seconds: float = FACE_SERVICE_TIMEOUT,
) -> Mapping[str, bool]:
    """Ask which of `frames` show a face, keyed by id.

    `frames` is `(id, jpeg_bytes, normalised_box)`. At most `FACE_SERVICE_CANDIDATES`
    are sent: these are HTTP round trips to another host and the close-up is built
    once per activity, so the list is bounded rather than exhaustive.

    Returns {} when the service is unset, unreachable, or answers nothing usable.
    An empty mapping is an ordinary result -- every caller already has a
    largest-box choice to fall back on -- so nothing here ever raises for a
    network failure. A failure to reach one frame's check does not discard the
    others: a partial answer still improves the choice.
    """
    endpoint = normalise_service_url(url)
    if not endpoint or not frames:
        return {}
    session = async_get_clientsession(hass)
    answers: dict[str, bool] = {}
    for frame_id, image_bytes, box in frames[:FACE_SERVICE_CANDIDATES]:
        crop = _encode_crop(image_bytes, box)
        if crop is None:
            continue
        try:
            answers[frame_id] = await _ask_one(session, endpoint, crop, wait_seconds)
        except FaceServiceError:
            continue
    return answers


def candidate_frames(
    frames: Iterable[tuple[str, bytes, Sequence[float]]],
    limit: int = FACE_SERVICE_CANDIDATES,
) -> list[tuple[str, bytes, Sequence[float]]]:
    """The frames worth asking about, largest box first.

    Ordered by box area so that the cap keeps the most promising candidates: if
    only some can be checked, the ones with the most person in them are the ones
    whose pose matters.
    """
    ordered = sorted(
        frames,
        key=lambda item: float(item[2][2]) * float(item[2][3]),
        reverse=True,
    )
    return ordered[:limit]
