"""Unit tests for the self-contained vision client."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import pytest
from PIL import Image

from custom_components.frigate_vision.vision import (
    VisionConfig,
    VisionError,
    build_payload,
    extract_json_object,
    resize_for_provider,
    validate_response,
)

ALLOWED = {"cleaning", "visitor", "unknown_activity", "unable_to_confirm"}


def test_review_mode_allows_arrival_and_departure() -> None:
    """The review path must be able to express coming home and leaving.

    Without these values the model can only reach them via `visitor` or
    `unknown_activity`, so a correct observation would be recorded as the wrong
    label regardless of what the prompt asks for.
    """
    from custom_components.frigate_vision.vision import (
        MODE_CLASSIFICATIONS,
    )

    review = MODE_CLASSIFICATIONS["review_six"]
    assert "home_arrival" in review
    assert "home_departure" in review
    # The other review outcomes must not have been displaced.
    assert {"package_delivery", "cleaning", "unknown_activity"} <= review


def test_a_direction_is_accepted_for_a_review_analysis() -> None:
    from custom_components.frigate_vision.vision import (
        MODE_CLASSIFICATIONS,
    )

    review = MODE_CLASSIFICATIONS["review_six"]
    classification, _description, confidence = validate_response(
        _body(
            '{"classification":"home_arrival",'
            '"description":"人物从门外走近并进入门内。","confidence":74}'
        ),
        review,
    )
    assert classification == "home_arrival"
    assert confidence == 74


def test_settings_entered_only_in_options_still_count_as_configured() -> None:
    """A UI-configured entry must produce a client.

    Provider settings entered from the Options page live in `options`, while the
    initial flow writes them to `data`. Judging readiness on `data` alone left a
    fully configured entry with no client at all, so every activity failed with
    `vision_not_configured` despite a valid model being set.
    """
    from custom_components.frigate_vision.vision import (
        vision_config_from,
        vision_is_configured,
    )

    options = {
        "llm_base_url": "http://provider.test:7864/v1",
        "llm_api_key": "secret",
        "llm_model": "deepseek-v4.1-flash",
        "llm_thinking": "default",
        "llm_reasoning_effort": "low",
    }
    assert vision_is_configured({}, options) is True
    config = vision_config_from({}, options)
    assert config.model == "deepseek-v4.1-flash"
    assert config.base_url == "http://provider.test:7864/v1"

    # And settings from the initial flow still work on their own.
    data = {
        "llm_base_url": "https://api.example.com/v1",
        "llm_api_key": "k",
        "llm_model": "m",
    }
    assert vision_is_configured(data, {}) is True


@pytest.mark.parametrize(
    "missing",
    ["llm_base_url", "llm_api_key", "llm_model"],
)
def test_partial_settings_are_not_configured(missing: str) -> None:
    """A URL with no key cannot produce a call, so it must not count."""
    from custom_components.frigate_vision.vision import (
        vision_is_configured,
    )

    options = {
        "llm_base_url": "https://api.example.com/v1",
        "llm_api_key": "secret",
        "llm_model": "model",
    }
    options[missing] = ""
    assert vision_is_configured({}, options) is False


def test_options_override_data_for_the_same_field() -> None:
    """The Options page is the live control, so it must win over the flow value."""
    from custom_components.frigate_vision.vision import (
        vision_config_from,
    )

    data = {
        "llm_model": "from-flow",
        "llm_base_url": "https://a/v1",
        "llm_api_key": "k",
    }
    options = {"llm_model": "from-options"}
    assert vision_config_from(data, options).model == "from-options"


def _config(**overrides: Any) -> VisionConfig:
    base = {
        "base_url": "https://api.example.com/v1",
        "api_key": "secret",
        "model": "vision-model",
        "thinking": "disabled",
        "max_tokens": 4000,
        "target_width": 768,
        "language": "zh-CN",
    }
    base.update(overrides)
    return VisionConfig(**base)


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_endpoint_accepts_every_reasonable_base_url() -> None:
    """The field is typed by hand, so all three forms must work.

    A bare host, a `/v1` root, and the full completions path are all things a
    user may reasonably enter; silently requesting a wrong path would fail with
    an opaque 404.
    """
    for value, expected in (
        ("https://api.deepseek.com", "https://api.deepseek.com/chat/completions"),
        ("https://api.deepseek.com/", "https://api.deepseek.com/chat/completions"),
        ("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
        (
            "https://api.deepseek.com/v1/chat/completions",
            "https://api.deepseek.com/v1/chat/completions",
        ),
        (
            "  https://api.deepseek.com/v1  ",
            "https://api.deepseek.com/v1/chat/completions",
        ),
    ):
        assert _config(base_url=value).endpoint() == expected


def test_thinking_disabled_is_sent_at_the_body_top_level() -> None:
    """The toggle must be a top-level body key.

    The provider documents it through the OpenAI SDK's `extra_body`, which only
    merges keys into the request body. Sending it nested would be silently
    ignored and quietly restore the full reasoning cost.
    """
    payload = build_payload(_config(thinking="disabled"), "prompt", b"jpeg")
    assert payload["thinking"] == {"type": "disabled"}
    # Not nested under any other key.
    assert "extra_body" not in payload
    assert all(
        not (isinstance(value, dict) and "thinking" in value)
        for key, value in payload.items()
        if key != "thinking"
    )


def test_default_thinking_omits_the_field_entirely() -> None:
    """`default` must send nothing, so the provider's own default applies."""
    payload = build_payload(_config(thinking="default"), "prompt", b"jpeg")
    assert "thinking" not in payload
    assert "reasoning_effort" not in payload


def test_reasoning_effort_is_forwarded_when_set() -> None:
    payload = build_payload(
        _config(thinking="default", reasoning_effort="low"), "prompt", b"jpeg"
    )
    assert payload["reasoning_effort"] == "low"


def test_reasoning_effort_default_sends_nothing() -> None:
    """`default` must leave the field out so the provider's own default applies.

    Sending an explicit value would pin the cost profile even when the user
    never chose one.
    """
    payload = build_payload(_config(reasoning_effort="default"), "prompt", b"jpeg")
    assert "reasoning_effort" not in payload


def test_low_effort_keeps_reasoning_enabled() -> None:
    """Effort must not disable thinking: that combination defeats the purpose.

    Measured on a multi-step task, disabling thinking entirely returned
    unrelated output, while `low` answered correctly at roughly 60% of the
    default's reasoning tokens. So `low` is the safe cost control and must keep
    reasoning on.
    """
    payload = build_payload(
        _config(thinking="default", reasoning_effort="low"), "prompt", b"jpeg"
    )
    assert payload.get("thinking") != {"type": "disabled"}
    assert payload["reasoning_effort"] == "low"


def test_payload_carries_the_image_as_a_data_url() -> None:
    payload = build_payload(_config(), "the prompt", b"\x01\x02\x03")
    content = payload["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "the prompt"}
    url = content[1]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]) == b"\x01\x02\x03"


def test_no_schema_or_sampling_fields_are_sent() -> None:
    """json_schema is rejected outright and sampling is inert here.

    Sending `response_format` would make the provider return HTTP 400, and
    `temperature`/`top_p` have no effect in thinking mode -- exposing either
    would imply a control that does not exist.
    """
    payload = build_payload(_config(), "prompt", b"jpeg")
    assert "response_format" not in payload
    assert "temperature" not in payload
    assert "top_p" not in payload


def test_resize_shrinks_wide_images_and_keeps_aspect(tmp_path) -> None:
    path = tmp_path / "wide.png"
    path.write_bytes(_png(1920, 720))
    data = resize_for_provider(path, 768)
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (768, 288)
        assert image.format == "JPEG"


def test_resize_leaves_small_images_alone(tmp_path) -> None:
    path = tmp_path / "small.png"
    path.write_bytes(_png(320, 240))
    data = resize_for_provider(path, 768)
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == (320, 240)


def test_resize_rejects_a_non_image(tmp_path) -> None:
    path = tmp_path / "evidence.jpg"
    path.write_bytes(b"not an image")
    with pytest.raises(VisionError, match="invalid_evidence_image"):
        resize_for_provider(path, 768)


def test_extract_handles_all_three_reply_shapes() -> None:
    payload = '{"classification":"visitor","description":"x","confidence":10}'
    assert extract_json_object(payload)["classification"] == "visitor"
    assert (
        extract_json_object(f"```json\n{payload}\n```")["classification"] == "visitor"
    )
    assert (
        extract_json_object(f"结果如下：\n{payload}\n以上。")["classification"]
        == "visitor"
    )
    assert extract_json_object("没有对象") is None


def _body(content: str, *, finish: str = "stop", reasoning: int | None = None):
    body: dict[str, Any] = {
        "choices": [{"message": {"content": content}, "finish_reason": finish}]
    }
    if reasoning is not None:
        body["usage"] = {"completion_tokens_details": {"reasoning_tokens": reasoning}}
    return body


def test_validate_accepts_a_well_formed_reply() -> None:
    classification, description, confidence = validate_response(
        _body('{"classification":"cleaning","description":"拖地。","confidence":92}'),
        ALLOWED,
    )
    assert (classification, description, confidence) == ("cleaning", "拖地。", 92)


def test_validate_parses_a_fenced_reply_with_float_confidence() -> None:
    classification, _description, confidence = validate_response(
        _body(
            "```json\n"
            '{"classification":"visitor","description":"停留片刻。","confidence":80.0}'
            "\n```"
        ),
        ALLOWED,
    )
    assert (classification, confidence) == ("visitor", 80)


def test_validate_rejects_a_classification_outside_the_enum() -> None:
    """The enum cannot be enforced by the provider, so it is enforced here.

    Measured live: the model answered "mopping_floor", which is not an allowed
    value; storing it would put an unbounded string into the record.
    """
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(
            _body(
                '{"classification":"mopping_floor","description":"x","confidence":90}'
            ),
            ALLOWED,
        )


def test_empty_content_names_the_reasoning_culprit() -> None:
    """A starved reasoning budget must not look like a malformed reply.

    With max_tokens too low the provider spends the whole budget reasoning and
    returns empty content with finish_reason="length" -- and no error. Naming
    that case makes the fix obvious instead of looking like model disobedience.
    """
    with pytest.raises(VisionError, match="reasoning_exhausted_max_tokens"):
        validate_response(_body("", finish="length", reasoning=800), ALLOWED)


def test_empty_content_without_reasoning_is_a_plain_empty_response() -> None:
    with pytest.raises(VisionError, match="empty_provider_response"):
        validate_response(_body(""), ALLOWED)


def test_validate_rejects_prose_with_no_object() -> None:
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(_body("画面中似乎有一个人，但无法确认。"), ALLOWED)


def test_validate_rejects_extra_or_missing_keys() -> None:
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(
            _body(
                '{"classification":"visitor","description":"x","confidence":1,'
                '"extra":true}'
            ),
            ALLOWED,
        )
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(_body('{"classification":"visitor"}'), ALLOWED)


@pytest.mark.parametrize("confidence", [-1, 101, True, "high", None])
def test_validate_rejects_out_of_range_confidence(confidence: Any) -> None:
    body = _body(
        json.dumps(
            {
                "classification": "visitor",
                "description": "x",
                "confidence": confidence,
            }
        )
    )
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(body, ALLOWED)


def test_validate_rejects_a_missing_choices_array() -> None:
    with pytest.raises(VisionError, match="invalid_provider_response"):
        validate_response({"error": "nope"}, ALLOWED)


class _FakeResponse:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _FakeSession:
    """Minimal aiohttp.ClientSession stand-in capturing the posted payload."""

    def __init__(self, status: int = 200, body: str | None = None) -> None:
        self.status = status
        self.body = body or json.dumps({"choices": [{"message": {"content": "OK"}}]})
        self.payloads: list[dict[str, Any]] = []
        self.urls: list[str] = []

    def post(self, url: str, *, json: dict[str, Any], **_kwargs: Any):
        self.urls.append(url)
        self.payloads.append(json)
        return _FakeResponse(self.status, self.body)


async def test_connection_probe_is_text_only_and_cheap() -> None:
    """The check must not send an image.

    Users run it repeatedly while editing fields; attaching the contact sheet
    would multiply the cost of a diagnostic and make the token figure it reports
    unrepresentative of the call it is meant to verify.
    """
    from custom_components.frigate_vision.vision import (
        async_test_connection,
    )

    session = _FakeSession()
    report = await async_test_connection(session, _config())  # type: ignore[arg-type]
    payload = session.payloads[0]
    content = payload["messages"][0]["content"]
    assert isinstance(content, str), "the probe must send text, not image parts"
    assert "image_url" not in json.dumps(payload)
    assert payload["max_tokens"] == 16
    assert report.model == "vision-model"


async def test_connection_probe_reports_the_providers_own_status() -> None:
    """A rejected key must be distinguishable from an unreachable host.

    Collapsing every failure into one message would send the user looking at the
    wrong field.
    """
    from custom_components.frigate_vision.vision import (
        async_test_connection,
    )

    with pytest.raises(VisionError, match="provider_http_401"):
        await async_test_connection(  # type: ignore[arg-type]
            _FakeSession(status=401, body="{}"), _config()
        )
    with pytest.raises(VisionError, match="provider_http_404"):
        await async_test_connection(  # type: ignore[arg-type]
            _FakeSession(status=404, body="{}"), _config()
        )


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("base_url", "llm_missing_url"),
        ("api_key", "llm_missing_key"),
        ("model", "llm_missing_model"),
    ],
)
async def test_connection_probe_names_the_missing_field(
    field: str, expected: str
) -> None:
    from custom_components.frigate_vision.vision import (
        async_test_connection,
    )

    with pytest.raises(VisionError, match=expected):
        await async_test_connection(  # type: ignore[arg-type]
            _FakeSession(), _config(**{field: ""})
        )


async def test_connection_probe_sends_the_thinking_toggle() -> None:
    """The probe must exercise the same body shape as a real analysis."""
    from custom_components.frigate_vision.vision import (
        async_test_connection,
    )

    session = _FakeSession()
    await async_test_connection(session, _config(thinking="disabled"))  # type: ignore[arg-type]
    assert session.payloads[0]["thinking"] == {"type": "disabled"}

    session = _FakeSession()
    await async_test_connection(session, _config(thinking="default"))  # type: ignore[arg-type]
    assert "thinking" not in session.payloads[0]
