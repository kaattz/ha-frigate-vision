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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp
from homeassistant.core import HomeAssistant
from homeassistant.helpers.aiohttp_client import async_get_clientsession
from PIL import Image, UnidentifiedImageError

from .const import (
    CONF_FACE_SERVICE_URL,
    CONF_FACE_SERVICE_URL_DEFAULT,
    CONF_PERSON_HIGHLIGHT,
    CONF_PERSON_HIGHLIGHT_DEFAULT,
    CONF_PROMPT_OVERRIDE,
    CONF_SCENE_DESCRIPTION,
    CONF_SCENE_LABELS,
    MAX_PROMPT_OVERRIDE_LENGTH,
    MAX_SCENE_DESCRIPTION_LENGTH,
    PERSON_HIGHLIGHT_WIDTH,
)
from .models import (
    AUTOMATIC_RETRY_STATUSES,
    ActivityRecord,
    ActivityStage,
    analysis_key,
    provider_status,
)
from .scenes import (
    SCENES,
    SceneRequest,
    effective_prompt_version,
    parse_scene_labels,
    scene_for,
)
from .store import ActivityStore

_LOGGER = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 180.0

# How many times a transient provider status is attempted in total, including
# the first try. Four attempts wait 2s + 6s + 18s = 26s between them, which is
# short enough that an activity is still delivered while the person is likely
# still nearby, and long enough to ride out the burst of 503s an overloaded
# endpoint emits.
#
# Bounded on purpose. An uncapped loop against a provider that is down for the
# night would never finish the activity and would keep issuing a billable request
# every backoff interval; the cap converts that into a recorded failure the user
# can see and replay by hand.
PROVIDER_RETRY_ATTEMPTS = 4
PROVIDER_RETRY_BACKOFF_SECONDS = 2.0
# x3 rather than the usual x2: 503s from an overloaded endpoint tend to arrive in
# a burst, so the useful signal is "has the burst passed", and the geometric
# schedule spends its waits where that is decided.
PROVIDER_RETRY_BACKOFF_FACTOR = 3.0

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
    # The deployment's own description of the camera's view. Empty by default,
    # and empty means the prompt is byte-for-byte what it was before this
    # option existed.
    scene_description: str = ""
    # 该 entry 自定义的标签与提示词覆盖，原样来自选项。空表示用场景内置的。
    scene_labels: str = ""
    prompt_override: str = ""
    # 证据图右侧是否附了一栏人物放大特写。默认关：关着的时候提示词与缓存键都
    # 逐字节不变，既有部署升级后行为与缓存完全不受影响。
    person_highlight: bool = CONF_PERSON_HIGHLIGHT_DEFAULT
    # Optional OpenCV face-detection service, used only to choose which frame the
    # close-up comes from. Empty means the largest-box rule decides alone, which
    # is the behaviour every existing deployment already has.
    face_service_url: str = CONF_FACE_SERVICE_URL_DEFAULT

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


def evidence_width_budget(config: VisionConfig) -> int:
    """The widest the evidence sheet may be before it must be scaled down.

    A budget rather than a boolean `already_sized`: the option says what *new*
    sheets look like, but a sheet can be reused from before the option was
    switched on, and that one is grid-only and up to 1920 wide. Deriving a
    boolean from the current setting would pass such a sheet through unscaled --
    6.3x the pixels, billed, for a close-up that is not in the image. A budget
    stays correct for both: a fresh composed sheet equals it and is left alone,
    while a stale grid-only one exceeds it and is still shrunk.
    """
    if config.person_highlight:
        return config.target_width + PERSON_HIGHLIGHT_WIDTH
    return config.target_width


def resize_for_provider(
    path: Path, target_width: int, *, already_sized: bool = False
) -> bytes:
    """Return JPEG bytes no wider than target_width.

    Images are resized rather than sent at native resolution: the sheet is
    1920x720 and the provider bills by pixel area, so sending it whole costs
    several times more for no extra legibility.

    `already_sized` is for a sheet that has already been scaled by
    `build_contact_sheet` and had a person close-up column appended. Such a sheet
    is wider than `target_width` on purpose -- the grid was scaled first and the
    column added afterwards -- so shrinking it here would undo the scaling
    decision and take the close-up down with it. The default is unchanged: any
    caller that does not opt out still gets an over-wide image scaled down.

    Prefer the budget form over this flag where the caller has the close-up width
    in hand: a boolean is derived from the *current* setting, so it also passes
    through artifacts built before the setting changed. See `evidence_width_budget`.
    """
    try:
        with Image.open(path) as source:
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise VisionError("invalid_evidence_image") from exc
    width, height = image.size
    if not already_sized and width > target_width:
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

    def pick_flag(key: str, default: bool) -> bool:
        """Read a boolean the way `pick` reads text: options first, then data.

        Deliberately not `bool(value)`, which is how the older
        `analyze_all_far_reviews` option is read elsewhere. A stored value can
        come back as the *string* "false" -- from the frontend, or from a JSON
        round trip -- and `bool("false")` is True. Here that would silently turn
        the close-up on for a user who switched it off, changing the image that
        gets billed and the cache key with it. Only a recognised true/false is
        honoured; anything else falls back to the default rather than to
        truthiness.
        """
        value = options.get(key)
        if value is None or value == "":
            value = data.get(key)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("true", "yes", "on", "1"):
                return True
            if text in ("false", "no", "off", "0"):
                return False
        return default

    return VisionConfig(
        base_url=pick("llm_base_url"),
        api_key=pick("llm_api_key"),
        model=pick("llm_model"),
        thinking=pick("llm_thinking", "default"),
        reasoning_effort=pick("llm_reasoning_effort", "default"),
        max_tokens=int(options.get("max_tokens", 4000)),
        target_width=int(options.get("target_width", 768)),
        language=str(options.get("output_language", "zh-CN")),
        scene_description=pick(CONF_SCENE_DESCRIPTION)[
            :MAX_SCENE_DESCRIPTION_LENGTH
        ],
        scene_labels=pick(CONF_SCENE_LABELS, ""),
        prompt_override=pick(CONF_PROMPT_OVERRIDE, "")[
            :MAX_PROMPT_OVERRIDE_LENGTH
        ],
        person_highlight=pick_flag(
            CONF_PERSON_HIGHLIGHT, CONF_PERSON_HIGHLIGHT_DEFAULT
        ),
        face_service_url=pick(CONF_FACE_SERVICE_URL, CONF_FACE_SERVICE_URL_DEFAULT),
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
        # 解析一次，同一份有序列表同时喂给提示词渲染和缓存键。两者若拿到
        # 不同的顺序，键会漂移（只是多一次缓存未命中，不会拿到过期答案），
        # 但没必要冒这个风险。
        try:
            custom_labels = parse_scene_labels(self._config.scene_labels)
        except ValueError as exc:
            # 标签格式是用户在选项里填的，写错了要给一个能看懂的错误码，
            # 而不是落到通用分支变成 analysis_outcome_unknown。这里在 claim
            # 与 provider 调用之前，所以不会产生费用。
            raise VisionError(str(exc)) from exc
        # 契约枚举与答案校验必须用同一个集合：改了一边而另一边不认，答案会被
        # invalid_llm_response 静默丢弃，用户只看到「没有通知」。
        allowed = (
            {name for name, _ in custom_labels}
            if custom_labels
            else set(scene.classifications)
        )

        key = analysis_key(
            activity_id,
            record.evidence_mode,
            effective_prompt_version(
                record.evidence_mode,
                self._config.scene_description,
                scene_labels=custom_labels,
                prompt_override=self._config.prompt_override,
                has_person_highlight=self._config.person_highlight,
            )
            or scene.prompt_version,
        )
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
            # The scene is part of the claimed key, so it must be passed back
            # rather than left to be re-derived from an ambiguous version string.
            scene_mode=record.evidence_mode,
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
    """POST the payload and return the decoded body.

    One attempt. Retrying lives in `async_request_with_retry` so that the
    connection probe -- which a user runs by hand and which must report a status
    immediately -- keeps answering on the first try.
    """
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
                # The provider's own message is the only thing that separates an
                # overloaded endpoint from a bad model name from a rejected
                # field, so it is logged here and reported alongside the status.
                # It cannot travel in the error code: that is persisted as the
                # record's `error_code` and validation restricts it to `SAFE_ID`
                # characters, so prose there would fail the write and downgrade a
                # clear provider failure to `analysis_outcome_unknown`.
                _LOGGER.warning(
                    "Provider answered HTTP %s: %s",
                    response.status,
                    _summarise_provider_message(text),
                )
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


def _summarise_provider_message(text: str, limit: int = 300) -> str:
    """Trim a provider error body to one log-friendly line.

    Providers answer errors with JSON, HTML error pages, or nothing at all.
    Whitespace is collapsed so a multi-line HTML page cannot flood the log, and
    the result is bounded because the body is attacker-influenced only in the
    sense that it comes from a remote host -- but it is remote text being written
    to a log, so it is capped rather than trusted.
    """
    collapsed = " ".join(text.split())
    if not collapsed:
        return "(empty body)"
    if len(collapsed) <= limit:
        return collapsed
    return f"{collapsed[:limit]}..."


async def async_request_with_retry(
    session: aiohttp.ClientSession,
    config: VisionConfig,
    payload: Mapping[str, Any],
    *,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> Mapping[str, Any]:
    """POST the payload, retrying the transient provider statuses.

    Only 502/503/504 are retried: each one means the provider did not process the
    request, so a second attempt cannot double-bill or duplicate a side effect.
    A 500 is left alone -- the provider admits a fault but says nothing about
    whether the request was handled, and that is the uncertainty that makes a
    replay unsafe.

    The attempt count is capped. An unbounded loop against a provider that is
    down for the night would never finish the activity, and would keep billing a
    request every backoff interval while it waited; a bounded one turns that into
    a recorded failure the user can see and replay by hand.

    `sleep` is injected so tests can assert the backoff schedule without serving
    it. It defaults to `asyncio.sleep`, and every production caller omits it.
    """
    attempts = 0
    while True:
        try:
            return await async_request(session, config, payload)
        except VisionError as exc:
            status = provider_status(str(exc))
            if status not in AUTOMATIC_RETRY_STATUSES:
                raise
            attempts += 1
            if attempts >= PROVIDER_RETRY_ATTEMPTS:
                _LOGGER.warning(
                    "Giving up on the vision provider after %s attempts (HTTP %s)",
                    attempts,
                    status,
                )
                raise
            delay = PROVIDER_RETRY_BACKOFF_SECONDS * (
                PROVIDER_RETRY_BACKOFF_FACTOR ** (attempts - 1)
            )
            _LOGGER.warning(
                "Vision provider answered HTTP %s; retrying in %.1fs (attempt %s/%s)",
                status,
                delay,
                attempts + 1,
                PROVIDER_RETRY_ATTEMPTS,
            )
            await sleep(delay)


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

    The returned version is the effective one -- the base version with the
    deployment's scene description folded in -- not the bare
    `scene.prompt_version`. The caller stores it, and the store derives the
    claimed side-effect key from it, so returning the base version while the
    claim used the effective one would fail the analysis *after* the provider
    had already been billed, with `side_effect_key_mismatch`.
    """
    scene = scene_for(evidence_mode)
    if scene is None:
        raise VisionError("unsupported_evidence_mode")
    # 模块级函数是独立入口（测试与 e2e 会直接调它），所以这里也要转换，
    # 否则直接调用它的人拿到的仍是裸 ValueError。
    try:
        custom_labels = parse_scene_labels(config.scene_labels)
    except ValueError as exc:
        raise VisionError(str(exc)) from exc
    prompt = scene.render(
        SceneRequest(
            language=config.language,
            # 这里的 allowed 由调用方派生，与答案校验用的是同一个集合。
            allowed=allowed,
            # Only the signals this scene declared are passed on; the rest are
            # dropped here rather than filtered inside the scene, so an
            # undeclared input never reaches the renderer at all.
            signals={
                "door_remained_open": door_remained_open,
                "opening_side": opening_side,
            },
            scene_description=config.scene_description,
            scene_labels=custom_labels,
            prompt_override=config.prompt_override,
            has_person_highlight=config.person_highlight,
        )
    )
    image_bytes = await asyncio.get_running_loop().run_in_executor(
        None,
        resize_for_provider,
        Path(evidence_path),
        evidence_width_budget(config),
    )
    payload = build_payload(config, prompt, image_bytes)
    started = time.monotonic()
    body = await async_request_with_retry(session, config, payload)
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
    # The effective version, not the bare scene one: the caller stores this and
    # the store rebuilds the claimed key from it. See the docstring. It must come
    # from the same three inputs -- and the same label order -- as the key the
    # caller claimed, or the analysis fails with `side_effect_key_mismatch`
    # after the provider has already been billed.
    version = (
        effective_prompt_version(
            evidence_mode,
            config.scene_description,
            scene_labels=custom_labels,
            prompt_override=config.prompt_override,
            has_person_highlight=config.person_highlight,
        )
        or scene.prompt_version
    )
    return classification, description, confidence, version


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
