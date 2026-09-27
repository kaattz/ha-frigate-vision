"""Answer one question: does this crop contain a face?

The integration's person close-up picks the detection with the largest box, which
answers "where is most of the person" and not "which frame shows their face" -- on
the activity that prompted this, it picked a frame of the person's back while a
clear frontal frame sat four seconds earlier in the same clip.

Face detection is the signal that separates those two, and it is offered as an
improvement rather than a requirement: if this service is absent, slow, or wrong,
the integration keeps its existing largest-box choice. Nothing here may become
load-bearing.

Why a service and not a library: the Home Assistant container is Alpine (musl
libc), and every OpenCV wheel on PyPI targets glibc, so it cannot be pip-installed
there -- verified, pip reports no matching distribution for either opencv or
onnxruntime. Running here instead keeps OpenCV out of the HA image entirely.

The detector is YuNet, which OpenCV ships a wrapper for. It was chosen over the
older Haar cascade for a measured reason: on the real frames from this camera the
cascade is unavailable in OpenCV 5 and weaker in 4, while YuNet found the face in
all six frontal frames and never once reported one in the six back-facing frames.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

MODEL_PATH = os.environ.get("FACE_MODEL", "/app/yunet.onnx")
PORT = int(os.environ.get("FACE_PORT", "8788"))
# Below this a detection is not trusted. YuNet's own examples use 0.9, which on
# this camera's crops rejected real faces scoring 0.688; the model's default is too
# strict for a 640x360 detect stream. 0.6 is where the measured distributions
# separate: every true frontal frame scored 0.688-0.893, and the single back-of-head
# false positive scored 0.523. Chosen from those numbers, not from the docs.
SCORE_THRESHOLD = float(os.environ.get("FACE_SCORE_THRESHOLD", "0.6"))
# Faces smaller than this fraction of the crop are not worth reporting. Measured
# against this camera's frames rather than guessed: a real, correctly-detected face
# in a 637x360 frame came out 26x34 px, which is 0.39% of the image. An earlier
# value of 2% -- five times that -- silently discarded every true detection while
# looking like a sensible safeguard, and the service returned "no face" for frames
# where the detector had found one at 0.815 confidence.
MIN_FACE_FRACTION = float(os.environ.get("FACE_MIN_FRACTION", "0.0005"))
# The shorter side a crop is enlarged to before detection. 320 is the size YuNet's
# own examples use; smaller left faces undetected on this camera's crops.
UPSCALE_TO = float(os.environ.get("FACE_UPSCALE_TO", "320"))

# Detectors are size-specific, so one is cached per input size. Guarded because the
# server is threaded: two requests can arrive for a new size at once.
_detectors: dict[tuple[int, int], cv2.FaceDetectorYN] = {}
_lock = threading.Lock()


def detector_for(width: int, height: int) -> cv2.FaceDetectorYN:
    key = (width, height)
    with _lock:
        detector = _detectors.get(key)
        if detector is None:
            detector = cv2.FaceDetectorYN.create(
                MODEL_PATH, "", (width, height), score_threshold=SCORE_THRESHOLD
            )
            _detectors[key] = detector
        return detector


def detect(image_bytes: bytes) -> dict:
    """Return the best face in the image, or a negative answer.

    Never raises for bad input: a caller that gets an error has to decide what to
    do about it, and the whole point of this service is that its absence is
    survivable. A malformed image is simply "no face".

    The caller sends the *person crop*, not the whole frame. Measured: on a 637x360
    frame the person's head is about 30 px across, which the model resolves only
    sometimes, and the caller has already cut the person out anyway. A crop puts
    more pixels on the face for the same bandwidth.
    """
    buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        return {"has_face": False, "score": 0.0, "box": None, "reason": "undecodable"}

    height, width = image.shape[:2]
    if width < 16 or height < 16:
        return {"has_face": False, "score": 0.0, "box": None, "reason": "too_small"}

    # Upscale so the shorter side reaches UPSCALE_TO. A head in a person crop is
    # still only ~40 px, which is around the model's limit; enlarging first was
    # measured to turn misses into detections. Only ever enlarges -- shrinking a
    # large crop would throw away the detail this exists to use.
    shorter = min(width, height)
    scale = max(1.0, UPSCALE_TO / shorter)
    if scale > 1.0:
        image = cv2.resize(
            image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        height, width = image.shape[:2]

    _, faces = detector_for(width, height).detect(image)
    if faces is None or len(faces) == 0:
        return {"has_face": False, "score": 0.0, "box": None}

    best = max(faces, key=lambda face: float(face[-1]))
    face_width, face_height = float(best[2]), float(best[3])
    if (face_width * face_height) < MIN_FACE_FRACTION * (width * height):
        return {"has_face": False, "score": 0.0, "box": None, "reason": "too_small"}

    # Reported in the ORIGINAL image's coordinates, so a caller can draw on the
    # image it sent rather than the upscaled one used internally.
    return {
        "has_face": True,
        "score": float(best[-1]),
        "box": [
            float(best[0]) / scale,
            float(best[1]) / scale,
            face_width / scale,
            face_height / scale,
        ],
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # noqa: ANN002
        """Silence per-request logging; the caller logs anything that matters."""

    def _reply(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        """A health check, so the integration can probe without sending an image."""
        if self.path.rstrip("/") in ("/health", ""):
            self._reply(
                {"status": "ok", "model": os.path.basename(MODEL_PATH),
                 "threshold": SCORE_THRESHOLD}
            )
            return
        self._reply({"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/face":
            self._reply({"error": "not_found"}, status=404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._reply({"error": "bad_length"}, status=400)
            return
        if length <= 0 or length > 32 * 1024 * 1024:
            self._reply({"error": "bad_length"}, status=400)
            return
        try:
            request = json.loads(self.rfile.read(length) or b"{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._reply({"error": "bad_json"}, status=400)
            return
        encoded = request.get("image")
        if not isinstance(encoded, str) or not encoded:
            self._reply({"error": "no_image"}, status=400)
            return
        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            self._reply({"error": "bad_base64"}, status=400)
            return
        self._reply(detect(image_bytes))


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"face model missing: {MODEL_PATH}")
    # 0.0.0.0 so the container is reachable from the HA host; this service is meant
    # for a trusted LAN and has no authentication of its own.
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
