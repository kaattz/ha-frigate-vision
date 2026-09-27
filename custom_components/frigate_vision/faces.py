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

from .const import (
    FACE_SERVICE_CANDIDATES,
    FACE_SERVICE_TIMEOUT,
)


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
    every one of them, while the same frames cropped to the person were all found.
    Fewer pixels, more face.

    The padding comes from `crop_person_box`, the same function the close-up itself
    uses, and that is load-bearing rather than tidy. Cutting to the box's exact
    edges clips the top of the head, and the model then finds nothing: measured on
    one activity, the unpadded 137x180 crop detected 0 of 6 frames while the padded
    247x261 crop detected 2, the frontal ones. A tightly framed face is a face with
    no forehead and no hairline, which is much of what the detector looks for.

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
    try:
        # Imported here rather than at module scope: `media` imports this module,
        # so a top-level import of `media` would be circular. The padding comes from
        # the same constant the close-up uses, which is what keeps the two crops
        # framed identically.
        from .media import PERSON_CROP_PADDING, crop_person_box

        region = crop_person_box(
            (x, y, width, height), frame_size=frame.size, padding=PERSON_CROP_PADDING
        )
    except ValueError:
        return None
    cropped = frame.crop(region)
    if cropped.size[0] < 8 or cropped.size[1] < 8:
        return None
    buffer = io.BytesIO()
    cropped.save(buffer, "JPEG", quality=90)
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
    frames: Iterable[tuple[str, Any]],
    limit: int = FACE_SERVICE_CANDIDATES,
) -> list[tuple[str, Any]]:
    """The frames worth asking about, spread across the activity.

    Ordered by timestamp and then thinned evenly, rather than simply truncated.
    The candidates are the sheet's own sample frames, which already span the
    activity in order, and taking the first N would check only its opening -- the
    part where the person is furthest away and least likely to be facing the
    camera. Spreading keeps candidates from across the clip, which is where a
    frontal frame appears.

    The element type is opaque here on purpose: the caller passes `(id, path)` and
    reads the paths back. Only the count and the order are this function's business.
    """
    ordered = sorted(frames, key=lambda item: item[0])
    if len(ordered) <= limit or limit < 2:
        return ordered[:limit]
    # Evenly spaced with both endpoints included: the last frame is as much a
    # candidate as the first, and truncating the index would never look at it.
    step = (len(ordered) - 1) / (limit - 1)
    return [ordered[round(index * step)] for index in range(limit)]
