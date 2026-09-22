"""Self-contained OpenAI-compatible vision client.

This integration used to delegate analysis to the `llmvision` component. That
worked, but blocked the one parameter that dominates cost: the reasoning toggle.
Measured on this deployment's own contact sheets, the provider spent 400-1900
tokens thinking before emitting a single character, and `max_tokens` was
exhausted by that reasoning often enough to return an empty answer with no
error. A Custom OpenAI provider in llmvision maps to its LocalAI class, which
never emits the toggle (only its Anthropic class does), and llmvision exposes no
passthrough for extra body fields.

So the request is built here instead. The scope is deliberately one provider
family -- OpenAI-compatible chat completions -- which is what both configured
providers speak.

Measured against api.deepseek.com:
  * `{"thinking": {"type": "disabled"}}` at the body top level is accepted and
    reduces a call from ~2866 to ~1118 tokens on identical input.
  * `response_format: {"type": "json_schema", ...}` is rejected with HTTP 400
    ("This response_format type is unavailable now"), so the response contract
    travels in the prompt and is validated locally.
  * `temperature` has no effect in thinking mode, and `top_p` is clamped, so
    neither is exposed.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from PIL import Image, UnidentifiedImageError

from .models import ActivityRecord, ActivityStage, analysis_key
from .scenes import SCENES, SceneRequest, scene_for
from .store import ActivityStore

_LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 180.0

# Which classifications each evidence mode may produce. Kept here because the
# response validator enforces it: the provider has no working structured-output
# mode on this endpoint, so an out-of-set value must be refused locally.
#
# Derived from the scene registry rather than listed again, so a scene's allowed
# answers cannot drift between what the prompt offers and what the validator
# accepts.
MODE_CLASSIFICATIONS: dict[str, set[str]] = {
    mode: set(scene.classifications) for mode, scene in SCENES.items()
}

# Text-mode replies can carry the object bare, inside a fenced block, or wrapped
# in prose; both patterns are bounded to a single object.
_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

THINKING_MODES = ("default", "disabled")

# Reasoning effort, as the provider documents it. "default" leaves the field
# out entirely so the provider's own default applies.
#
# Effort is the safer cost control than disabling thinking outright. Measured on
# a task needing multi-step reasoning: disabled returned unrelated output on
# every attempt, while `low` answered correctly using ~60% fewer reasoning
# tokens than the default. So this knob trades cost without giving up the
# ability to reason at all.
REASONING_EFFORTS = ("default", "low", "high", "max")

# The provider expects a data URL; JPEG keeps the payload small for a photo
# contact sheet with no visible loss at this size.
_JPEG_QUALITY = 85

# Raised when the process cannot reach the provider at all, as opposed to the
# provider answering with an error.
_CONNECTION_ERRORS = (aiohttp.ClientConnectionError, asyncio.TimeoutError)


class VisionError(RuntimeError):
    """The vision request or its response did not satisfy the contract."""


@dataclass(frozen=True)
class VisionConfig:
    """Provider settings, supplied by the config entry."""

    base_url: str
    api_key: str
    model: str
    thinking: str = "default"
    reasoning_effort: str = "default"
    max_tokens: int = 4000
    target_width: int = 768
    language: str = "zh-CN"

    def endpoint(self) -> str:
        """Return the chat-completions URL for this base URL.

        Accepts a bare host, a `/v1` root, or the full completions path, because
        the value is typed by hand and every form is a reasonable thing to
        enter.
        """
        base = self.base_url.strip().rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"


def resize_for_provider(path: Path, target_width: int) -> bytes:
    """Return JPEG bytes no wider than target_width.

    Images are resized rather than sent at native resolution: the sheet is
    1920x720 and the provider bills by pixel area, so sending it whole costs
    several times more for no extra legibility.
    """
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise VisionError("invalid_evidence_image") from exc
    width, height = image.size
    if width > target_width:
        target_height = max(1, round(height * target_width / width))
        image = image.resize((target_width, target_height))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return buffer.getvalue()


def build_payload(
    config: VisionConfig,
    prompt: str,
    image_bytes: bytes,
) -> dict[str, Any]:
    """Assemble the chat-completions body, including the reasoning toggle."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                    },
                ],
            }
        ],
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    # Sent at the body top level. The provider's own docs show this through the
    # OpenAI SDK's `extra_body`, which simply merges keys into the body; a
    # non-OpenAI endpoint ignores the field rather than failing (verified: both
    # configured endpoints accept an unknown field without error, so acceptance
    # alone proves nothing -- the effect was measured separately).
    if config.thinking == "disabled":
        payload["thinking"] = {"type": "disabled"}
    elif config.thinking == "enabled":
        payload["thinking"] = {"type": "enabled"}
    if config.reasoning_effort and config.reasoning_effort != "default":
        payload["reasoning_effort"] = config.reasoning_effort
    return payload


def extract_json_object(raw: Any) -> Any:
    """Recover a JSON object from a text-mode reply."""
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    text = raw.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fenced = _JSON_FENCE.search(text)
    if fenced is not None:
        try:
            return json.loads(fenced.group(1))
        except ValueError:
            pass
    brace = _JSON_OBJECT.search(text)
    if brace is not None:
        try:
            return json.loads(brace.group(0))
        except ValueError:
            return None
    return None


def vision_config_from(
    data: Mapping[str, Any], options: Mapping[str, Any]
) -> VisionConfig:
    """Build a VisionConfig from the entry's data and options.

    Options are read first and fall back to data, so provider settings can be
    changed from the Options page at any time while a value stored during the
    initial flow keeps working.
    """

    def pick(key: str, default: str = "") -> str:
        value = options.get(key)
        if value in (None, ""):
            value = data.get(key)
        return str(value) if value not in (None, "") else default

    return VisionConfig(
        base_url=pick("llm_base_url"),
        api_key=pick("llm_api_key"),
        model=pick("llm_model"),
        thinking=pick("llm_thinking", "default"),
        reasoning_effort=pick("llm_reasoning_effort", "default"),
        max_tokens=int(options.get("max_tokens", 4000)),
        target_width=int(options.get("target_width", 768)),
        language=str(options.get("output_language", "zh-CN")),
    )


def vision_is_configured(data: Mapping[str, Any], options: Mapping[str, Any]) -> bool:
    """Return whether enough settings exist to attempt an analysis.

    Provider settings may live in either `data` (set during the initial flow) or
    `options` (set later from the Options page), so presence is judged on the
    merged view rather than on `data` alone. Checking only `data` left an entry
    configured entirely through the UI with no client at all, and every activity
    then failed with `vision_not_configured`. Each field the request needs is
    required: a URL alone cannot produce a call.
    """
    config = vision_config_from(data, options)
    return bool(config.base_url and config.api_key and config.model)


class VisionClient:
    """Analyse one stored contact sheet and record the outcome.

    Owns the store transitions so a failure leaves a visible, specific error
    code rather than a generic one, and so the analysis side effect cannot be
    started twice.
    """

    def __init__(
        self, hass: HomeAssistant, store: ActivityStore, config: VisionConfig
    ) -> None:
        self._hass = hass
        self._store = store
        self._config = config

    async def async_analyze(self, activity_id: str) -> ActivityRecord:
        record = self._store.get(activity_id)
        if record is None:
            raise VisionError("activity_missing")
        if record.stage is not ActivityStage.EVIDENCE_READY:
            raise VisionError("stage_conflict")
        if record.processing_mode.value == "observe":
            raise VisionError("observe_llm_forbidden")
        if record.evidence_path is None or record.evidence_mode is None:
            raise VisionError("evidence_incomplete")
        exists = await self._hass.async_add_executor_job(
            Path(record.evidence_path).is_file
        )
        if not exists:
            raise VisionError("evidence_incomplete")
        scene = scene_for(record.evidence_mode)
        if scene is None:
            raise VisionError("unsupported_evidence_mode")
        allowed = set(scene.classifications)

        key = analysis_key(activity_id, record.evidence_mode, scene.prompt_version)
        started = await self._store.async_start_side_effect(
            activity_id,
            key,
            ActivityStage.EVIDENCE_READY,
            ActivityStage.ANALYSIS_STARTED,
            updated_at=time.time(),
        )
        if not started:
            raise VisionError("analysis_already_started")

        session = async_get_clientsession(self._hass)
        try:
            (
                classification,
                description,
                confidence,
                prompt_version,
            ) = await async_analyze(
                session,
                self._config,
                evidence_path=record.evidence_path,
                evidence_mode=record.evidence_mode,
                allowed=allowed,
                door_remained_open=record.door_remained_open,
                opening_side=record.opening_side,
            )
        except VisionError as exc:
            await self._store.async_transition(
                activity_id,
                ActivityStage.ANALYSIS_STARTED,
                ActivityStage.FAILED,
                updated_at=time.time(),
                error_code=str(exc),
            )
            raise
        except Exception as exc:  # noqa: BLE001
            # The request may or may not have reached the provider, so the
            # outcome is unknown rather than failed.
            await self._store.async_transition(
                activity_id,
                ActivityStage.ANALYSIS_STARTED,
                ActivityStage.FAILED,
                updated_at=time.time(),
                error_code="analysis_outcome_unknown",
            )
            raise VisionError("analysis_outcome_unknown") from exc
        return await self._store.async_complete_analysis(
            activity_id,
            prompt_version=prompt_version,
            classification=classification,
            description=description,
            confidence=confidence,
            updated_at=time.time(),
        )


def validate_response(body: Any, allowed: set[str]) -> tuple[str, str, int]:
    """Return (classification, description, confidence) or raise.

    The provider cannot enforce the enum (no working structured-output mode), so
    an out-of-set value is rejected here rather than stored.
    """
    if not isinstance(body, Mapping):
        raise VisionError("invalid_provider_response")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionError("invalid_provider_response")
    first = choices[0]
    if not isinstance(first, Mapping):
        raise VisionError("invalid_provider_response")
    message = first.get("message")
    if not isinstance(message, Mapping):
        raise VisionError("invalid_provider_response")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        # Empty content is how a starved reasoning budget presents itself: the
        # provider reports finish_reason="length" with every token spent on
        # reasoning. Surface it distinctly so the fix is obvious.
        finish = first.get("finish_reason")
        usage = body.get("usage")
        reasoning = None
        if isinstance(usage, Mapping):
            details = usage.get("completion_tokens_details")
            if isinstance(details, Mapping):
                reasoning = details.get("reasoning_tokens")
        if finish == "length" and reasoning:
            raise VisionError("reasoning_exhausted_max_tokens")
        raise VisionError("empty_provider_response")
    value = extract_json_object(content)
    if not isinstance(value, Mapping) or set(value) != {
        "classification",
        "description",
        "confidence",
    }:
        raise VisionError("invalid_llm_response")
    classification = value["classification"]
    description = value["description"]
    confidence = value["confidence"]
    if isinstance(confidence, float) and confidence.is_integer():
        confidence = int(confidence)
    elif isinstance(confidence, str) and confidence.strip().isdigit():
        confidence = int(confidence.strip())
    if (
        not isinstance(classification, str)
        or classification not in allowed
        or not isinstance(description, str)
        or not description.strip()
        or len(description) > 500
        or isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 0 <= confidence <= 100
    ):
        raise VisionError("invalid_llm_response")
    return classification, description.strip(), confidence


async def async_request(
    session: aiohttp.ClientSession,
    config: VisionConfig,
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    """POST the payload and return the decoded body."""
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with session.post(
            config.endpoint(),
            json=dict(payload),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SECONDS),
        ) as response:
            text = await response.text()
            if response.status != 200:
                # Keep the provider's own message: it is what distinguishes a
                # bad model name from a bad credential from a rejected field.
                raise VisionError(f"provider_http_{response.status}")
            try:
                body = json.loads(text)
            except ValueError as exc:
                raise VisionError("invalid_provider_response") from exc
            if not isinstance(body, dict):
                raise VisionError("invalid_provider_response")
            return body
    except _CONNECTION_ERRORS as exc:
        raise VisionError("provider_unavailable") from exc


async def async_analyze(
    session: aiohttp.ClientSession,
    config: VisionConfig,
    *,
    evidence_path: str,
    evidence_mode: str,
    allowed: set[str],
    door_remained_open: bool | None = None,
    opening_side: str = "unknown",
) -> tuple[str, str, int, str]:
    """Analyse one contact sheet.

    Returns (classification, description, confidence, prompt_version).
    """
    scene = scene_for(evidence_mode)
    if scene is None:
        raise VisionError("unsupported_evidence_mode")
    prompt = scene.render(
        SceneRequest(
            language=config.language,
            allowed=allowed,
            # Only the signals this scene declared are passed on; the rest are
            # dropped here rather than filtered inside the scene, so an
            # undeclared input never reaches the renderer at all.
            signals={
                "door_remained_open": door_remained_open,
                "opening_side": opening_side,
            },
        )
    )
    image_bytes = await asyncio.get_running_loop().run_in_executor(
        None, resize_for_provider, Path(evidence_path), config.target_width
    )
    payload = build_payload(config, prompt, image_bytes)
    started = time.monotonic()
    body = await async_request(session, config, payload)
    elapsed = time.monotonic() - started
    classification, description, confidence = validate_response(body, allowed)
    usage = body.get("usage") if isinstance(body, Mapping) else None
    total = usage.get("total_tokens") if isinstance(usage, Mapping) else None
    # A single debug line per call: enough to see cost drift and the reasoning
    # toggle taking effect without re-instrumenting.
    _LOGGER.debug(
        "vision call model=%s thinking=%s tokens=%s seconds=%.1f classification=%s",
        config.model,
        config.thinking,
        total,
        elapsed,
        classification,
    )
    return classification, description, confidence, scene.prompt_version


@dataclass(frozen=True)
class ConnectionReport:
    """Outcome of a provider connectivity probe."""

    model: str
    seconds: float
    total_tokens: int | None


async def async_test_connection(
    session: aiohttp.ClientSession,
    config: VisionConfig,
) -> ConnectionReport:
    """Make one cheap text-only call to prove the settings work.

    Deliberately sends no image: the point is to verify the endpoint, key and
    model name, and an image would multiply the cost of a check a user may run
    repeatedly while editing the fields. Raises VisionError with a specific code
    so the form can say what is actually wrong.
    """
    if not config.base_url:
        raise VisionError("llm_missing_url")
    if not config.api_key:
        raise VisionError("llm_missing_key")
    if not config.model:
        raise VisionError("llm_missing_model")
    payload: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": "Reply with the single word OK."}],
        "max_tokens": 16,
        "stream": False,
    }
    if config.thinking == "disabled":
        payload["thinking"] = {"type": "disabled"}
    if config.reasoning_effort and config.reasoning_effort != "default":
        payload["reasoning_effort"] = config.reasoning_effort
    started = time.monotonic()
    body = await async_request(session, config, payload)
    elapsed = time.monotonic() - started
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionError("invalid_provider_response")
    usage = body.get("usage")
    total = usage.get("total_tokens") if isinstance(usage, Mapping) else None
    return ConnectionReport(
        model=config.model,
        seconds=elapsed,
        total_tokens=total if isinstance(total, int) else None,
    )
