"""Unit tests for the self-contained vision client."""

from __future__ import annotations

import base64
import io
import json
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from PIL import Image

from custom_components.frigate_vision.models import analysis_key
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


def test_a_prose_answer_in_the_household_format_is_rejected() -> None:
    """A human-readable answer format cannot be stored, however good it reads.

    Recorded because it is the obvious thing to try: a prompt whose output section
    asks for `事件类型：回家` on one line and `事件描述：...` on the next, written in
    Chinese with no JSON. That is a perfectly clear instruction for a person, and
    the answers it produces are exactly the labels the integration wants -- but the
    reply is not the three-field object the rest of the pipeline reads, so it fails
    before any of that prose is looked at.

    The rejection is deliberate rather than a gap: `confidence` has no place in a
    prose format, and the notification, the stored record and the blueprint all key
    off the classification enum. What is worth keeping from such a prompt is its
    *content* -- the scene layout, the time order, the wording of the categories --
    which can travel in the existing structure.
    """
    from custom_components.frigate_vision.vision import (
        MODE_CLASSIFICATIONS,
    )

    review = MODE_CLASSIFICATIONS["review_six"]
    prose = (
        "事件类型：离家\n"
        "事件描述：一名男子从左侧入户门走出，随后走向中央电梯。"
    )
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(_body(prose), review)


def test_a_chinese_label_is_not_silently_mapped_to_an_enum() -> None:
    """The category words must not be accepted in place of the stored values.

    `回家` and `home_arrival` mean the same thing to a reader, and the prompt the
    household wrote offers the Chinese forms. Accepting them would need a
    translation table in the validator, and a wrong or missing entry there would
    file an activity under the wrong label -- a silent, stored error rather than a
    visible failure. So the mapping lives in the prompt (which tells the model
    which value to emit) and the validator stays strict.
    """
    from custom_components.frigate_vision.vision import (
        MODE_CLASSIFICATIONS,
    )

    review = MODE_CLASSIFICATIONS["review_six"]
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(
            _body(
                '{"classification":"回家",'
                '"description":"人物进入门内。","confidence":80}'
            ),
            review,
        )


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


async def test_client_completes_the_analysis_it_claimed(
    hass: HomeAssistant,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The full claim -> analyse -> complete cycle must reach ANALYSIS_DONE.

    This is the path that broke in production. `VisionClient` claims its side
    effect with a key that includes the scene, but the store rebuilt that key
    without it when completing, so the membership check could never pass. Every
    real activity stopped at `analysis_started` with `side_effect_key_mismatch`
    and the retry policy then refused to re-run it, because an analysis that may
    already have been billed is not a safe retry.

    The unit tests around the store and the response validator both passed while
    this was broken, because nothing exercised the two halves together.
    """
    from custom_components.frigate_vision.models import (
        ActivityRecord,
        ActivitySource,
        ActivityStage,
        ProcessingMode,
    )
    from custom_components.frigate_vision.store import ActivityStore
    from custom_components.frigate_vision.vision import VisionClient

    sheet = tmp_path / "activity.png"
    sheet.write_bytes(_png(640, 240))

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            processing_mode=ProcessingMode.SHADOW,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="review_six",
            evidence_path=str(sheet),
        )
    )

    reply = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "classification": "visitor",
                                "description": "一人在门前经过。",
                                "confidence": 71,
                            }
                        )
                    }
                }
            ]
        }
    )
    session = _FakeSession(body=reply)
    monkeypatch.setattr(
        "custom_components.frigate_vision.vision.async_get_clientsession",
        lambda _hass: session,
    )

    client = VisionClient(hass, store, _config())
    done = await client.async_analyze("activity_1")

    assert done.stage is ActivityStage.ANALYSIS_DONE
    assert done.classification == "visitor"
    assert done.confidence == 71
    # The scene-aware key must be what got recorded, not a shortened variant.
    assert (
        analysis_key("activity_1", "review_six", done.prompt_version)
        in done.claimed_side_effects
    )


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


def test_a_custom_label_is_accepted_and_a_builtin_one_is_not() -> None:
    """自定义标签必须被校验器接受，内置标签必须被拒绝。

    这是整个功能的核心不变量：契约渲染与答案校验用同一个 allowed 集合。
    若两者不一致，模型按自定义标签回答而校验器只认内置标签，答案会被
    invalid_llm_response 静默丢弃，用户只看到「没有通知」。
    """
    from custom_components.frigate_vision.vision import validate_response

    allowed = {"宠物", "无人"}
    got, _d, _c = validate_response(
        _body('{"classification":"宠物","description":"只有一只猫。","confidence":80}'),
        allowed,
    )
    assert got == "宠物"
    with pytest.raises(VisionError, match="invalid_llm_response"):
        validate_response(
            _body(
                '{"classification":"elevator_activity",'
                '"description":"x","confidence":80}'
            ),
            allowed,
        )


def test_the_config_carries_custom_labels_and_override() -> None:
    """配置对象必须能携带这两个新设置，否则接线无处可接。"""
    from custom_components.frigate_vision.vision import VisionConfig

    config = VisionConfig(
        base_url="https://api.example.com/v1",
        api_key="k",
        model="m",
        scene_labels="宠物: 只有宠物",
        prompt_override="自定义规则。",
    )
    assert config.scene_labels == "宠物: 只有宠物"
    assert config.prompt_override == "自定义规则。"


def test_vision_config_from_reads_both_new_options() -> None:
    """两个新选项必须从 entry 的 options/data 里读出来。"""
    from custom_components.frigate_vision.vision import vision_config_from

    config = vision_config_from(
        {},
        {
            "llm_base_url": "https://api.example.com/v1",
            "llm_api_key": "k",
            "llm_model": "m",
            "scene_labels": "宠物: 只有宠物",
            "prompt_override": "自定义规则。",
        },
    )
    assert config.scene_labels == "宠物: 只有宠物"
    assert config.prompt_override == "自定义规则。"


def test_vision_config_from_defaults_the_new_options_to_empty() -> None:
    """没配置时必须是空字符串——空意味着「用内置场景」，零回归。"""
    from custom_components.frigate_vision.vision import vision_config_from

    config = vision_config_from(
        {},
        {
            "llm_base_url": "https://api.example.com/v1",
            "llm_api_key": "k",
            "llm_model": "m",
        },
    )
    assert config.scene_labels == ""
    assert config.prompt_override == ""


def test_the_prompt_override_is_capped_at_the_configured_limit() -> None:
    """超长文本要在读入时截断，避免误粘一整篇文档把提示词撑爆。"""
    from custom_components.frigate_vision.const import MAX_PROMPT_OVERRIDE_LENGTH
    from custom_components.frigate_vision.vision import vision_config_from

    config = vision_config_from(
        {},
        {
            "llm_base_url": "https://api.example.com/v1",
            "llm_api_key": "k",
            "llm_model": "m",
            "prompt_override": "x" * (MAX_PROMPT_OVERRIDE_LENGTH + 500),
        },
    )
    assert len(config.prompt_override) == MAX_PROMPT_OVERRIDE_LENGTH


async def test_a_configured_entry_analyses_and_completes_its_claim(
    hass: HomeAssistant,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """配上自定义标签后，整条 claim -> analyse -> complete 必须仍然走通。

    这是接线的端到端护栏，也是三个各自独立的失效点合流的地方：

    1. 校验器：模型答的是自定义标签，所以 `allowed` 必须来自 entry 的配置。
       若仍用 `scene.classifications`，这一格答案会被 invalid_llm_response
       静默丢弃，用户只看到「没有通知」。
    2. 缓存键：`VisionClient` 认领副作用时把配置折进了键。
    3. 返回的版本：模块级函数必须用**同样三个输入**算出 effective version
       返回，否则 store 重建出的键与认领的键不符，分析在**已经计费之后**
       以 side_effect_key_mismatch 失败。

    这三处各自都有单测通过、合起来却全断的先例（见
    `test_client_completes_the_analysis_it_claimed` 的 docstring），所以必须
    用一次真实调用把它们串起来验证。
    """
    from custom_components.frigate_vision.models import (
        ActivityRecord,
        ActivitySource,
        ActivityStage,
        ProcessingMode,
    )
    from custom_components.frigate_vision.store import ActivityStore
    from custom_components.frigate_vision.vision import VisionClient

    sheet = tmp_path / "activity.png"
    sheet.write_bytes(_png(640, 240))

    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            processing_mode=ProcessingMode.SHADOW,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="review_six",
            evidence_path=str(sheet),
        )
    )

    reply = json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "classification": "猫",
                                "description": "画面里只有一只猫。",
                                "confidence": 71,
                            }
                        )
                    }
                }
            ]
        }
    )
    session = _FakeSession(body=reply)
    monkeypatch.setattr(
        "custom_components.frigate_vision.vision.async_get_clientsession",
        lambda _hass: session,
    )

    config = _config(scene_labels="猫: 只有一只猫", prompt_override="只看猫。")
    client = VisionClient(hass, store, config)
    done = await client.async_analyze("activity_1")

    assert done.stage is ActivityStage.ANALYSIS_DONE
    # 内置标签集里没有「猫」，所以这一格能到 ANALYSIS_DONE，就证明校验器
    # 用的确实是 entry 自己的 `allowed`。
    assert done.classification == "猫"
    assert done.confidence == 71
    # 提示词契约里枚举的必须就是模型答出来的那个标签。
    content = session.payloads[0]["messages"][0]["content"]
    prompt = content[0]["text"]
    assert "猫" in prompt
    assert "elevator_activity" not in prompt


# 每种畸形写法对应的精确错误码。底层解析器的契约就是抛 ValueError，其消息
# 即错误码；视觉调用路径必须把这个码原样带出去，而不是换成通用错误。
MALFORMED_LABELS = [
    ("宠物", "label_malformed"),
    ("宠物:", "label_definition_missing"),
    ("宠物: 只有猫\n宠物: 又来一只", "label_duplicate"),
]


async def _store_with_a_ready_activity(
    hass: HomeAssistant, sheet: Any, *, entry_id: str = "entry_1"
) -> Any:
    """建一个持有单个 EVIDENCE_READY 活动的 store，与运行时的状态一致。

    构造方式与 `test_a_configured_entry_analyses_and_completes_its_claim` 相同，
    这样畸形标签的用例走的是真实可认领的活动，而不是桩对象。

    `entry_id` 可换是为了让一个测试里跑两遍：HA 的 Store 按 entry_id 持久化，
    同一个 id 第二次 `async_load` 会读回上一次已完成的记录，直接 stage_conflict。
    """
    from custom_components.frigate_vision.models import (
        ActivityRecord,
        ActivitySource,
        ActivityStage,
        ProcessingMode,
    )
    from custom_components.frigate_vision.store import ActivityStore

    store = ActivityStore(hass, entry_id)
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id=entry_id,
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            processing_mode=ProcessingMode.SHADOW,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="review_six",
            evidence_path=str(sheet),
        )
    )
    return store


def test_person_highlight_is_off_by_default() -> None:
    """默认不启用 —— 既有部署的行为与缓存完全不变。

    这是一个**行为变更**（改变发给模型的图）并且会改变缓存键，所以默认必须是
    关的：没打开开关的部署，提示词逐字节不变、键也逐字节不变，缓存全部命中。
    """
    from custom_components.frigate_vision.const import (
        CONF_PERSON_HIGHLIGHT_DEFAULT,
    )
    from custom_components.frigate_vision.vision import vision_config_from

    assert CONF_PERSON_HIGHLIGHT_DEFAULT is False
    assert VisionConfig(
        base_url="https://api.example.com/v1", api_key="k", model="m"
    ).person_highlight is False
    # 没配置时（既有 entry 的 options 里根本没有这个键）也是关的。
    assert vision_config_from({}, {}).person_highlight is False


def test_enabling_the_highlight_changes_the_cache_key() -> None:
    """打开开关必须换键，否则已有活动不会重新分析，会返回旧提示词的结果。

    键与提示词必须同源：只改提示词不换键，用户打开开关后拿到的仍是**上一次
    提示词**产生的存储结果，改动看起来完全没生效。本项目已因这种形状丢过功能
    两次。
    """
    from custom_components.frigate_vision.const import CONF_PERSON_HIGHLIGHT
    from custom_components.frigate_vision.scenes import SCENES, effective_prompt_version
    from custom_components.frigate_vision.vision import vision_config_from

    base = SCENES["review_six"].prompt_version
    assert effective_prompt_version("review_six", "") == base
    assert (
        effective_prompt_version("review_six", "", has_person_highlight=False) == base
    )
    on = effective_prompt_version("review_six", "", has_person_highlight=True)
    assert on != base, "打开特写改变了提示词，缓存键必须跟着变"

    # 而且这个值确实是从 entry 的配置里读出来的，不是某处写死的常量。
    enabled = vision_config_from(
        {}, {CONF_PERSON_HIGHLIGHT: True, "llm_model": "m"}
    )
    assert enabled.person_highlight is True
    assert (
        effective_prompt_version(
            "review_six", "", has_person_highlight=enabled.person_highlight
        )
        == on
    )


async def test_the_claim_and_the_rendered_prompt_use_the_same_flag(
    hass: HomeAssistant,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """claim 键与 render 必须用同一个值。

    不一致会导致 claim 键 != return 键 -> side_effect_key_mismatch，
    而那发生在 provider 已经计费之后。

    这里不读源码、也不假定哪几行传参，而是让**真实的一次分析**跑完，然后检查
    两个可观测结果是否自洽：

      * ground truth —— 模型实际收到的提示词里有没有特写段落（render 传的值）；
      * store 的结果 —— 认领的副作用键是哪一个版本（claim 传的值）。

    分析能到 ANALYSIS_DONE，就要求 claim 键 == return 键 == store 重建的键。
    于是只要 claim 与 render 传了不同的值，三者中必有一处对不上，这个测试就会
    以 side_effect_key_mismatch（或 store 里没有那个键）失败。两个方向都跑一遍，
    所以恒 True / 恒 False 的写死同样会被抓住。
    """
    from custom_components.frigate_vision.models import ActivityStage
    from custom_components.frigate_vision.scenes import effective_prompt_version
    from custom_components.frigate_vision.vision import VisionClient

    # 提示词里只有特写段落用到「放大特写」这个词，所以它是 render 是否收到
    # True 的直接证据。
    marker = "放大特写"
    seen: list[tuple[bool, str]] = []

    for flag in (False, True):
        sheet = tmp_path / f"activity_{flag}.png"
        sheet.write_bytes(_png(640, 240))
        # 每轮一个独立的 entry_id：HA 的 Store 按 entry_id 持久化，复用会让
        # 第二次 async_load 读回上一轮已完成的记录。
        store = await _store_with_a_ready_activity(
            hass, sheet, entry_id=f"entry_{flag}"
        )
        reply = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "classification": "visitor",
                                    "description": "一人在门前经过。",
                                    "confidence": 71,
                                }
                            )
                        }
                    }
                ]
            }
        )
        session = _FakeSession(body=reply)
        monkeypatch.setattr(
            "custom_components.frigate_vision.vision.async_get_clientsession",
            lambda _hass, _session=session: _session,
        )

        client = VisionClient(hass, store, _config(person_highlight=flag))
        done = await client.async_analyze("activity_1")

        # render 实际收到了什么。
        rendered = session.payloads[0]["messages"][0]["content"][0]["text"]
        rendered_highlight = marker in rendered

        # claim 实际用了哪个版本：键必须就在认领集合里，且它就是 store 用来
        # 完成分析的那一个（能到 ANALYSIS_DONE 已证明这一点）。
        claimed = done.claimed_side_effects
        expected_version = effective_prompt_version(
            "review_six", "", has_person_highlight=flag
        )
        assert done.stage is ActivityStage.ANALYSIS_DONE, (
            f"person_highlight={flag} 时分析没有走完：{done.error_code!r}。"
            "claim 键与 return 键不同会在这里以 side_effect_key_mismatch 失败，"
            "而那发生在 provider 已经计费之后。"
        )
        assert analysis_key("activity_1", "review_six", expected_version) in claimed, (
            f"person_highlight={flag} 时认领的键不是 {expected_version!r}，"
            f"实际认领：{sorted(claimed)}"
        )
        # 反过来：另一个取值的键不能被认领，否则说明 flag 根本没进键。
        other_version = effective_prompt_version(
            "review_six", "", has_person_highlight=not flag
        )
        assert analysis_key("activity_1", "review_six", other_version) not in claimed

        seen.append((rendered_highlight, done.prompt_version or ""))

    # 两行必须自洽：render 收到什么，claim 就用了什么。任何一处写死/传错，
    # 两行之间就会出现「提示词不同而版本相同」或反之。
    assert seen[0][0] is False and seen[1][0] is True, (
        f"提示词没有按 person_highlight 变化：{seen}"
    )
    assert seen[0][1] != seen[1][1], (
        f"两个取值的 prompt_version 相同（{seen[0][1]!r}），说明键没有跟着开关走"
    )
    for flag, (rendered_highlight, version) in zip((False, True), seen, strict=True):
        assert version == effective_prompt_version(
            "review_six", "", has_person_highlight=flag
        ), (
            f"person_highlight={flag}：模型收到的图"
            f"{'有' if rendered_highlight else '没有'}特写，"
            f"但 claim 用的是版本 {version!r}，与 render 传的值不一致。"
            "两处不一致会让 claim 键 != return 键 -> side_effect_key_mismatch，"
            "而那发生在 provider 已经计费之后。"
        )


@pytest.mark.parametrize(("labels", "expected"), MALFORMED_LABELS)
async def test_malformed_labels_raise_a_vision_error_not_a_value_error(
    hass: HomeAssistant,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    labels: str,
    expected: str,
) -> None:
    """标签格式写错必须给出精确的错误码，而不是裸 ValueError。

    裸 ValueError 会落到 runtime.py 的通用 except 分支，最终表现为
    analysis_outcome_unknown——用户看到这个码无法知道是标签格式写错了。
    VisionError 的错误码会被原样记录到诊断传感器。

    这条路径在 claim 与 provider 调用之前，所以不产生费用；下面断言没有任何
    请求发出，就是这一点的直接证据。
    """
    from custom_components.frigate_vision.scenes import parse_scene_labels
    from custom_components.frigate_vision.vision import VisionClient

    # 底层解析器仍然抛 ValueError —— 那是它的契约（保存时校验会直接用它），
    # 不该为了调用方而改。
    with pytest.raises(ValueError, match=expected):
        parse_scene_labels(labels)

    sheet = tmp_path / "activity.png"
    sheet.write_bytes(_png(640, 240))
    store = await _store_with_a_ready_activity(hass, sheet)

    session = _FakeSession()
    monkeypatch.setattr(
        "custom_components.frigate_vision.vision.async_get_clientsession",
        lambda _hass: session,
    )

    client = VisionClient(hass, store, _config(scene_labels=labels))
    with pytest.raises(VisionError) as caught:
        await client.async_analyze("activity_1")

    # 精确到错误码本身：既证明不是裸 ValueError，也证明没有被替换成
    # analysis_outcome_unknown 之类的通用码。
    assert str(caught.value) == expected
    assert session.payloads == []


@pytest.mark.parametrize(("labels", "expected"), MALFORMED_LABELS)
async def test_the_module_level_analysis_rejects_malformed_labels_too(
    tmp_path,
    labels: str,
    expected: str,
) -> None:
    """模块级 `async_analyze` 是独立入口（测试与 e2e 会直接调它），同样要转换。

    只保护 `VisionClient` 那一处会留下同一个洞：直接调用模块级函数的人拿到的
    仍是裸 ValueError。
    """
    from custom_components.frigate_vision.vision import async_analyze

    sheet = tmp_path / "activity.png"
    sheet.write_bytes(_png(640, 240))
    session = _FakeSession()

    with pytest.raises(VisionError) as caught:
        await async_analyze(  # type: ignore[arg-type]
            session,
            _config(scene_labels=labels),
            evidence_path=str(sheet),
            evidence_mode="review_six",
            allowed={"visitor"},
        )

    assert str(caught.value) == expected
    assert session.payloads == []
