"""故障转移的错误分类与转向逻辑。"""

from __future__ import annotations

import asyncio
import io
import json
import logging
from typing import Any

import pytest
from homeassistant.core import HomeAssistant
from PIL import Image
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision import vision
from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
    is_failover_eligible,
    is_provider_side_failure,
    retry_is_safe,
    the_provider_is_the_suspect,
)
from custom_components.frigate_vision.runtime import EntryRuntime, IntegrationRuntime
from custom_components.frigate_vision.store import ActivityStore
from custom_components.frigate_vision.vision import (
    PROVIDER_RETRY_ATTEMPTS,
    VisionClient,
    VisionConfig,
    VisionError,
    fallback_config_from,
    fallback_is_configured,
    vision_config_from,
)

_ENTRY_ID = "entry_1"
_VISION_LOGGER = "custom_components.frigate_vision.vision"

# Full completions paths, so `VisionConfig.endpoint()` returns them unchanged and
# the fake session can route by URL without re-deriving anything.
_PRIMARY_URL = "https://primary.example.com/v1/chat/completions"
_FALLBACK_URL = "https://backup.example.com/v1/chat/completions"
_PRIMARY_MODEL = "primary-model"
_FALLBACK_MODEL = "backup-model"


class TestFailoverEligibility:
    """哪些错误值得换一组供应商再试。"""

    @pytest.mark.parametrize(
        "code",
        [
            "provider_http_500",
            "provider_http_502",
            "provider_http_503",
            "provider_http_504",
            "provider_http_429",
            "provider_unavailable",
        ],
    )
    def test_transport_and_server_failures_fail_over(self, code: str) -> None:
        assert is_failover_eligible(code)

    @pytest.mark.parametrize(
        "code",
        [
            "invalid_llm_response",
            "empty_provider_response",
            "reasoning_exhausted_max_tokens",
            "invalid_provider_response",
        ],
    )
    def test_contract_failures_fail_over(self, code: str) -> None:
        """答了但不合契约：换个模型可能答得对，所以也转移。"""
        assert is_failover_eligible(code)

    @pytest.mark.parametrize(
        "code",
        [
            "evidence_incomplete",
            "stage_conflict",
            "unsupported_evidence_mode",
            "analysis_already_started",
            "vision_not_configured",
            "label_malformed",
            # 结果未知：可能已计费，绝不对同一活动发起第二次调用。
            "analysis_outcome_unknown",
        ],
    )
    def test_local_and_unknown_errors_do_not_fail_over(self, code: str) -> None:
        assert not is_failover_eligible(code)


class TestBlameIsNotFailover:
    """两个问题，相反的偏置，必须由两个谓词回答。"""

    @pytest.mark.parametrize(
        "code",
        [
            # 我们自己的错误：store / media / frigate 层
            "terminal_activity",
            "identity_conflict",
            "entry_mismatch",
            "side_effect_key_mismatch",
            "history_capacity",
            "queue_full",
            "frame_decode_failed",
            "artifact_missing",
            "review_too_short",
            "invalid_path_data",
            "recording_gap",
            "authentication_failed",
            "request_timeout",
            "frigate_unavailable",
            "media_cleanup_failed",
            "storage_corrupt",
            "media_deadline_missing",
            "face_service_unreachable",
            # 分析路径上的本地错误
            "evidence_incomplete",
            "stage_conflict",
            "vision_not_configured",
            "analysis_outcome_unknown",
            "label_malformed",
            "too_many_labels",
            "ambiguous_review_ownership",
        ],
    )
    def test_our_own_faults_are_never_blamed_on_the_provider(self, code: str) -> None:
        assert not the_provider_is_the_suspect(code)

    @pytest.mark.parametrize(
        "code",
        [
            "provider_unavailable",
            "provider_http_500",
            "provider_http_502",
            "provider_http_503",
            "provider_http_504",
            "provider_http_429",
            "provider_http_418",
            "provider_http_599",
            "invalid_provider_response",
            "invalid_llm_response",
            "empty_provider_response",
            "reasoning_exhausted_max_tokens",
        ],
    )
    def test_genuine_provider_faults_are_blamed(self, code: str) -> None:
        assert the_provider_is_the_suspect(code)

    def test_a_code_we_raise_may_still_be_worth_failing_over(self) -> None:
        """`media_retry_exhausted` 不转移也不归咎；但两个谓词的偏置本来就不同。

        这条钉住二者的关系：转移是「未知即尝试」，归咎是「未知即不是 provider」。
        """
        assert not is_failover_eligible("media_retry_exhausted")
        assert not the_provider_is_the_suspect("media_retry_exhausted")
        # A code can be failover-eligible yet not blamed (unknown provider code
        # that we have never seen a status for is impossible; but a *known*
        # provider status is both).
        assert is_failover_eligible("provider_http_503")
        assert the_provider_is_the_suspect("provider_http_503")


class TestUnknownErrorsFailOver:
    """未知错误码必须转移 —— 这是「不丢活动」的兜底。

    供应商侧的失败是开放集合：网关可以发明一个我们从未见过的状态码。把它当成
    「本地代码的错」就会在第一次出现新故障时静默停止转移，而那正是本功能存在的
    理由。所以拒绝集合只列本地错误（我们自己产生的、可枚举的闭集）。
    """

    @pytest.mark.parametrize(
        "code",
        [
            "a_brand_new_provider_error",
            "provider_malformed_sse",
            "some_future_status_we_never_saw",
            "provider_http_599",
        ],
    )
    def test_unknown_codes_are_worth_trying_elsewhere(self, code: str) -> None:
        assert is_failover_eligible(code)


class TestEveryLocalErrorIsDenied:
    """本地错误必须完整枚举：漏掉一个就会白调一次第二组。"""

    @pytest.mark.parametrize(
        "code",
        [
            # 状态机 / 证据检查
            "activity_missing",
            "stage_conflict",
            "evidence_incomplete",
            "unsupported_evidence_mode",
            "analysis_already_started",
            # 证据图解码：两组用同一张图
            "invalid_evidence_image",
            # 标签解析：两组共享同一份 scene_labels
            "label_malformed",
            "label_name_too_long",
            "label_name_invalid",
            "label_definition_missing",
            "label_definition_too_long",
            "label_duplicate",
            "too_many_labels",
            # 结果未知：可能已计费
            "analysis_outcome_unknown",
        ],
    )
    def test_local_errors_never_fail_over(self, code: str) -> None:
        assert not is_failover_eligible(code)


class TestFallbackConfiguration:
    """第二组只在三个字段齐全时启用。"""

    def test_complete_fallback_is_configured(self) -> None:
        options = {
            "fallback_llm_base_url": "http://backup.local/v1",
            "fallback_llm_api_key": "k",
            "fallback_llm_model": "m",
        }
        assert fallback_is_configured({}, options)
        config = fallback_config_from({}, options)
        assert config is not None
        assert config.base_url == "http://backup.local/v1"
        assert config.model == "m"

    @pytest.mark.parametrize(
        "missing",
        ["fallback_llm_base_url", "fallback_llm_api_key", "fallback_llm_model"],
    )
    def test_incomplete_fallback_is_not_configured(self, missing: str) -> None:
        """半配置不启用：运行单组，而不是带着一半配置去调用。"""
        options = {
            "fallback_llm_base_url": "http://backup.local/v1",
            "fallback_llm_api_key": "k",
            "fallback_llm_model": "m",
        }
        options[missing] = ""
        assert not fallback_is_configured({}, options)
        assert fallback_config_from({}, options) is None

    def test_absent_fallback_is_not_configured(self) -> None:
        assert not fallback_is_configured({}, {})
        assert fallback_config_from({}, {}) is None

    def test_fallback_inherits_the_shared_prompt_settings(self) -> None:
        """共享设置（标签、描述、宽度）落到第二组，因为它决定「问什么」。"""
        options = {
            "fallback_llm_base_url": "http://backup.local/v1",
            "fallback_llm_api_key": "k",
            "fallback_llm_model": "m",
            "fallback_llm_thinking": "disabled",
            "target_width": 1024,
            "scene_description": "左侧是入户门",
            "output_language": "zh-CN",
        }
        config = fallback_config_from({}, options)
        assert config is not None
        assert config.thinking == "disabled"
        assert config.target_width == 1024
        assert config.scene_description == "左侧是入户门"

    def test_fallback_own_reasoning_settings_do_not_leak_to_primary(self) -> None:
        options = {
            "llm_base_url": "http://primary.local/v1",
            "llm_api_key": "p",
            "llm_model": "pm",
            "llm_thinking": "default",
            "fallback_llm_base_url": "http://backup.local/v1",
            "fallback_llm_api_key": "k",
            "fallback_llm_model": "m",
            "fallback_llm_thinking": "disabled",
        }
        primary = vision_config_from({}, options)
        fallback = fallback_config_from({}, options)
        assert primary.thinking == "default", "第二组的 thinking 不能污染主组"
        assert fallback is not None and fallback.thinking == "disabled"

    def test_shared_settings_are_identical_for_both_providers(self) -> None:
        """提示词相关设置必须完全相同 —— 换供应商不能改变问题本身。"""
        options = {
            "llm_base_url": "http://primary.local/v1",
            "llm_api_key": "p",
            "llm_model": "pm",
            "fallback_llm_base_url": "http://backup.local/v1",
            "fallback_llm_api_key": "k",
            "fallback_llm_model": "m",
            "scene_labels": "visitor: 访客",
            "prompt_override": "自定义规则",
            "max_tokens": 1234,
            "target_width": 999,
            "output_language": "en",
            "person_highlight": True,
        }
        primary = vision_config_from({}, options)
        fallback = fallback_config_from({}, options)
        assert fallback is not None
        for field in (
            "scene_labels",
            "prompt_override",
            "max_tokens",
            "target_width",
            "language",
            "person_highlight",
        ):
            assert getattr(primary, field) == getattr(fallback, field), field

    def test_a_stored_false_string_still_means_false_for_the_fallback(self) -> None:
        """字符串 "false" 不是真值 —— 这是已经踩过的缺陷，重构不能弄丢它。"""
        options = {
            "fallback_llm_base_url": "http://backup.local/v1",
            "fallback_llm_api_key": "k",
            "fallback_llm_model": "m",
            "person_highlight": "false",
        }
        config = fallback_config_from({}, options)
        assert config is not None
        assert config.person_highlight is False

    @pytest.mark.parametrize("stored", ["true", True, "on"])
    def test_a_stored_true_is_true_for_the_fallback_too(self, stored: Any) -> None:
        """它的孪生断言：省略字段不能冒充「解析正确」。

        `person_highlight` 的默认值就是 `False`，所以上面那条测试在字段**完全
        没被读取**时也会通过——`assert config.person_highlight is False` 无法区分
        「读到了 false」和「根本没读」。这一条用真值来钉住同一个字段：只有真的
        走了 `pick_flag` 才可能得到 `True`。

        只列 `bool` 与字符串形态：`pick_flag` 的契约是「只认明确的是/否，其余一律
        回落到默认值」，所以整数 `1` 有意**不**当作真值（见其 docstring）。
        """
        options = {
            "fallback_llm_base_url": "http://backup.local/v1",
            "fallback_llm_api_key": "k",
            "fallback_llm_model": "m",
            "person_highlight": stored,
        }
        config = fallback_config_from({}, options)
        assert config is not None
        assert config.person_highlight is True


# --------------------------------------------------------------------------- #
# The failover loop itself.
# --------------------------------------------------------------------------- #


def _ok_body(description: str, classification: str = "visitor") -> str:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "classification": classification,
                                "description": description,
                                "confidence": 71,
                            }
                        )
                    }
                }
            ]
        }
    )


class _RoutingSession:
    """Answer per-URL, so the two providers are distinguishable in one call.

    Routing by URL rather than by call order matters: an ordering-based fake
    would silently swap which provider it was answering as soon as the loop's
    attempt count changed, and every assertion about "the fallback answered"
    would then be testing the fake instead of the loop.
    """

    def __init__(self, by_url: dict[str, list[tuple[int, str]]]) -> None:
        self._by_url = by_url
        self.calls: list[str] = []

    def post(self, url: str, *, json: dict[str, Any], **_kwargs: Any):
        self.calls.append(url)
        script = self._by_url[url]
        index = min(
            sum(1 for seen in self.calls if seen == url) - 1,
            len(script) - 1,
        )
        status, body = script[index]
        return _Response(status, body)

    def calls_to(self, url: str) -> int:
        return sum(1 for seen in self.calls if seen == url)


class _Response:
    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


def _png(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def _provider_config(url: str, model: str) -> VisionConfig:
    return VisionConfig(
        base_url=url,
        api_key="secret",
        model=model,
        thinking="disabled",
        language="zh-CN",
    )


async def _ready_store(hass: HomeAssistant, sheet: Any) -> ActivityStore:
    store = ActivityStore(hass, _ENTRY_ID)
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id=_ENTRY_ID,
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.EVIDENCE_READY,
            created_at=1,
            updated_at=1,
            camera="front",
            evidence_mode="review_six",
            evidence_path=str(sheet),
        )
    )
    return store


async def _wire(
    hass: HomeAssistant,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_script: list[tuple[int, str]],
    fallback_script: list[tuple[int, str]] | None = None,
    with_fallback: bool = True,
) -> tuple[ActivityStore, VisionClient, _RoutingSession]:
    """Build a real store, client and routing session for one analysis.

    The backoff constant is zeroed for every case that reaches the transport
    retry: the 2s/6s/18s schedule is a policy this change must not touch, and it
    is pinned as literals by `test_provider_retry.py`, so serving it here would
    only cost 26 seconds per test. The attempt *count* is untouched.
    """
    monkeypatch.setattr(vision, "PROVIDER_RETRY_BACKOFF_SECONDS", 0.0)

    sheet = tmp_path / "activity.png"
    sheet.write_bytes(_png(640, 240))
    store = await _ready_store(hass, sheet)

    by_url: dict[str, list[tuple[int, str]]] = {_PRIMARY_URL: primary_script}
    if with_fallback:
        by_url[_FALLBACK_URL] = fallback_script or [(200, _ok_body("备用组的描述。"))]
    session = _RoutingSession(by_url)
    monkeypatch.setattr(
        "custom_components.frigate_vision.vision.async_get_clientsession",
        lambda _hass: session,
    )
    client = VisionClient(
        hass,
        store,
        _provider_config(_PRIMARY_URL, _PRIMARY_MODEL),
        _provider_config(_FALLBACK_URL, _FALLBACK_MODEL) if with_fallback else None,
    )
    return store, client, session


class TestTheFailoverLoop:
    """一次认领之内，主组失败后转向备用组。"""

    async def test_the_primary_answering_alone_makes_exactly_one_call(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 1：主组成功时行为不变，且只发一次 HTTP。"""
        store, client, session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(200, _ok_body("主组的描述。"))],
        )

        done = await client.async_analyze("activity_1")

        assert done.stage is ActivityStage.ANALYSIS_DONE
        assert done.description == "主组的描述。"
        assert session.calls_to(_PRIMARY_URL) == 1
        assert session.calls_to(_FALLBACK_URL) == 0, "备用组不能被白调"
        assert client.provider_for("activity_1") == "primary"
        assert store.get("activity_1") is not None

    async def test_the_fallback_answer_completes_the_activity(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 2：主组可转移地失败、备用组成答，活动以**备用组的答案**完成。"""
        store, client, _session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "overloaded")],
            fallback_script=[(200, _ok_body("备用组的描述。"))],
        )

        done = await client.async_analyze("activity_1")

        assert done.stage is ActivityStage.ANALYSIS_DONE
        assert done.description == "备用组的描述。"
        assert done.error_code is None
        fallback_record = store.get("activity_1")
        assert fallback_record is not None
        assert fallback_record.error_code is None
        assert client.provider_for("activity_1") == "fallback"

    async def test_a_local_error_never_calls_the_fallback(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 3：本地错误（不可转移）绝不能调第二组。

        用一张损坏的证据图产生真正的本地错误 `invalid_evidence_image`——两组拿到
        的是同一张图，第二组必然撞同一堵墙，所以不该白调。
        """
        sheet = tmp_path / "activity.png"
        sheet.write_bytes(b"not an image at all")
        store = await _ready_store(hass, sheet)
        session = _RoutingSession(
            {
                _PRIMARY_URL: [(200, _ok_body("never"))],
                _FALLBACK_URL: [(200, _ok_body("never"))],
            }
        )
        monkeypatch.setattr(
            "custom_components.frigate_vision.vision.async_get_clientsession",
            lambda _hass: session,
        )
        client = VisionClient(
            hass,
            store,
            _provider_config(_PRIMARY_URL, _PRIMARY_MODEL),
            _provider_config(_FALLBACK_URL, _FALLBACK_MODEL),
        )

        with pytest.raises(VisionError, match="invalid_evidence_image"):
            await client.async_analyze("activity_1")

        assert session.calls == [], "本地错误在构造请求之前就该结束"
        local_record = store.get("activity_1")
        assert local_record is not None
        assert local_record.error_code == "invalid_evidence_image"

    async def test_a_local_error_is_refused_even_when_the_fallback_could_answer(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 3 的判别版：备用组**本来能答对**，也必须不去调它。

        上一条测试里两组共用同一张损坏的图，所以「备用组一次都没被调用」在
        「拒绝本地错误」和「调了但同样失败」两种实现下都成立 —— 它能证明真实
        代码路径会产生 `invalid_evidence_image`，却不能证明那条 `not
        is_failover_eligible(code)` 判断存在。把备用组换成会成功的桩，两者才分得开。

        判据不是「本地错误不该转移」这句话本身，而是它的理由：本地错误意味着
        第二组撞的是同一堵墙。用一个**不会**撞墙的备用组，就把「因为本地所以
        不试」和「试了但碰巧也失败」彻底分开。
        """
        sheet = tmp_path / "activity.png"
        sheet.write_bytes(_png(640, 240))
        store = await _ready_store(hass, sheet)

        seen: list[str] = []

        async def scripted(session: Any, config: VisionConfig, **_kwargs: Any) -> Any:
            seen.append(config.model)
            if config.model == _PRIMARY_MODEL:
                # After the claim, inside the loop: exactly where the real
                # `resize_for_provider` raises for an unreadable sheet.
                raise VisionError("invalid_evidence_image")
            return ("visitor", "备用组本可以答对。", 71, "prompt_6")

        monkeypatch.setattr(
            "custom_components.frigate_vision.vision.async_analyze", scripted
        )
        client = VisionClient(
            hass,
            store,
            _provider_config(_PRIMARY_URL, _PRIMARY_MODEL),
            _provider_config(_FALLBACK_URL, _FALLBACK_MODEL),
        )

        with pytest.raises(VisionError, match="invalid_evidence_image"):
            await client.async_analyze("activity_1")

        assert seen == [_PRIMARY_MODEL], (
            "本地错误不该去问第二组，哪怕它答得出来"
        )
        refused = store.get("activity_1")
        assert refused is not None
        assert refused.stage is ActivityStage.FAILED
        assert refused.error_code == "invalid_evidence_image"

    async def test_an_unexpected_exception_never_calls_the_fallback(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 4：意外异常（结果未知）绝不转移——可能已经计费。"""
        sheet = tmp_path / "activity.png"
        sheet.write_bytes(_png(640, 240))
        store = await _ready_store(hass, sheet)

        calls: list[str] = []

        async def explode(*_args: Any, **_kwargs: Any) -> Any:
            calls.append("primary")
            raise RuntimeError("socket layer blew up")

        monkeypatch.setattr(
            "custom_components.frigate_vision.vision.async_request_with_retry",
            explode,
        )
        client = VisionClient(
            hass,
            store,
            _provider_config(_PRIMARY_URL, _PRIMARY_MODEL),
            _provider_config(_FALLBACK_URL, _FALLBACK_MODEL),
        )

        with pytest.raises(VisionError, match="analysis_outcome_unknown"):
            await client.async_analyze("activity_1")

        assert calls == ["primary"], "意外异常之后不能再碰第二组"
        unknown_record = store.get("activity_1")
        assert unknown_record is not None
        assert unknown_record.error_code == "analysis_outcome_unknown"

    async def test_both_failing_records_the_last_providers_error(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 5：两组都失败时，记录**最后一个**供应商的错误码。"""
        store, client, _session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "primary down")],
            fallback_script=[(500, "backup broke")],
        )

        with pytest.raises(VisionError, match="provider_http_500"):
            await client.async_analyze("activity_1")

        record = store.get("activity_1")
        assert record is not None
        assert record.stage is ActivityStage.FAILED
        assert record.error_code == "provider_http_500", (
            "记录最后一家说的话：那才是用户重放时要面对的端点"
        )
        assert client.provider_for("activity_1") is None

    async def test_the_fallback_does_not_hit_analysis_already_started(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 6：整个循环只认领一次，备用组不能撞 `analysis_already_started`。

        认领一旦落进循环里，第二次迭代就会撞上自己刚写下的 key，于是备用组根本
        不会被调用——转移静默失效。所以这条断言的是「备用组真的被调过」。
        """
        _store, client, session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "down")],
            fallback_script=[(200, _ok_body("备用组补上了。"))],
        )

        done = await client.async_analyze("activity_1")

        assert session.calls_to(_FALLBACK_URL) == 1
        assert done.stage is ActivityStage.ANALYSIS_DONE
        claimed = [
            key for key in done.claimed_side_effects if key.startswith("analysis:")
        ]
        assert len(claimed) == 1, f"只能认领一次，实际 {claimed}"

    async def test_a_contract_failure_does_not_spend_the_retry_budget(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 7（也是 Part C2）：契约类失败主组只被调**一次**。

        `validate_response` 在 `async_request_with_retry` **之外**运行，所以越界
        答案天然不会被重试。这条把那个涌现行为钉住：HTTP 200 配不合契约的答案，
        主组必须只调一次就交给备用组。
        """
        bad = _ok_body("闲聊", classification="not_a_real_label")
        _store, client, session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(200, bad)],
            fallback_script=[(200, _ok_body("备用组答对了。"))],
        )

        done = await client.async_analyze("activity_1")

        assert session.calls_to(_PRIMARY_URL) == 1, (
            "契约失败在重试层之外抛出，所以不该花掉重试预算"
        )
        assert done.description == "备用组答对了。"
        assert client.provider_for("activity_1") == "fallback"

    async def test_a_malformed_body_is_not_retried_either(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Part C2：HTTP 200 配不合契约的 JSON 结构，主组同样只调一次。

        与上一条是同一个机制的两个入口：`invalid_provider_response` 由
        `async_request` 抛出、`invalid_llm_response` 由 `validate_response` 抛出，
        两者都在重试层之外，所以都不得消耗重试预算。
        """
        _store, client, session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(200, json.dumps({"choices": []}))],
            fallback_script=[(200, _ok_body("备用组答对了。"))],
        )

        done = await client.async_analyze("activity_1")

        assert session.calls_to(_PRIMARY_URL) == 1
        assert done.description == "备用组答对了。"

    async def test_a_transport_failure_spends_the_budget_before_failing_over(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 8：传输类失败先花完主组的重试预算，才轮到备用组。"""
        _store, client, session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "down")],
            fallback_script=[(200, _ok_body("备用组接手。"))],
        )

        done = await client.async_analyze("activity_1")

        assert session.calls_to(_PRIMARY_URL) == PROVIDER_RETRY_ATTEMPTS
        assert session.calls_to(_FALLBACK_URL) == 1
        assert done.description == "备用组接手。"

    async def test_without_a_fallback_the_behaviour_is_unchanged(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """要求 9：没配第二组时，逐字节就是今天的行为。"""
        store, client, session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "down")],
            with_fallback=False,
        )

        with pytest.raises(VisionError, match="provider_http_503"):
            await client.async_analyze("activity_1")

        assert session.calls_to(_PRIMARY_URL) == PROVIDER_RETRY_ATTEMPTS
        assert session.calls == [_PRIMARY_URL] * PROVIDER_RETRY_ATTEMPTS
        record = store.get("activity_1")
        assert record is not None
        assert record.error_code == "provider_http_503"
        assert client.provider_for("activity_1") is None

    async def test_the_fallback_success_logs_the_primarys_error(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """转移必须留痕：日志里要写清**为什么**换了供应商。"""
        _store, client, _session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "down")],
            fallback_script=[(200, _ok_body("备用组的描述。"))],
        )

        with caplog.at_level(logging.WARNING, logger=_VISION_LOGGER):
            await client.async_analyze("activity_1")

        messages = [failed.getMessage() for failed in caplog.records]
        assert any(
            "provider_http_503" in message and "fallback" in message
            for message in messages
        ), f"the failover must name the primary's error, got {messages}"

    async def test_both_failing_logs_both_errors(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """两组都失败时，两条错误都要出现在日志里。"""
        _store, client, _session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "primary down")],
            fallback_script=[(500, "backup broke")],
        )

        with caplog.at_level(logging.WARNING, logger=_VISION_LOGGER):
            with pytest.raises(VisionError):
                await client.async_analyze("activity_1")

        messages = [failed.getMessage() for failed in caplog.records]
        assert any(
            "provider_http_503" in message and "provider_http_500" in message
            for message in messages
        ), f"both errors must be named, got {messages}"

    async def test_the_fallback_answer_is_what_gets_stored(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """备用组的答案（分类、置信度、描述）整体落库。"""
        store, client, _session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(502, "bad gateway")],
            fallback_script=[
                (200, _ok_body("备用组看到有人经过。", classification="visitor"))
            ],
        )

        done = await client.async_analyze("activity_1")

        record = store.get("activity_1")
        assert record is not None
        assert record.classification == "visitor"
        assert record.confidence == 71
        assert record.description == "备用组看到有人经过。"
        assert done.stage is ActivityStage.ANALYSIS_DONE

    def test_provider_for_is_none_before_any_analysis(self) -> None:
        """未分析过的活动必须返回 None，而不是冒充 'primary'。"""
        client = VisionClient.__new__(VisionClient)
        client._last_provider = {}
        assert client.provider_for("activity_never_seen") is None
# --------------------------------------------------------------------------- #
# The reporting gap: one predicate for both decisions.
# --------------------------------------------------------------------------- #


async def _runtime(hass: HomeAssistant) -> IntegrationRuntime:
    """A runtime with an entry_id, so `record_error` can raise a repair."""
    store = ActivityStore(hass, _ENTRY_ID)
    await store.async_load()

    async def handler(message: object) -> None:
        return None

    return IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        entry_id=_ENTRY_ID,
    )


def _issues(hass: HomeAssistant) -> Any:
    import homeassistant.helpers.issue_registry as ir

    return ir.async_get(hass).issues


class TestTheProviderAlertCannotDrift:
    """报修与转移由**两个**判断回答，各自的未知偏置相反。

    `record_error` 原本用 `is_provider_side_failure`（只认 5xx + 429）决定是否
    报修，而转移用 `is_failover_eligible`（只排除本地错误的**允许清单**）。两者
    的差集就是「会转移、但两家都失败后一声不响」的那批错误码。

    同一个谓词不能同时回答两个问题：转移要「未知即尝试」（不试就丢活动），
    归咎要「未知即不是 provider」（`record_error` 的输入域是**任意**异常字符串，
    把库存冲突或 Frigate 鉴权失败算到视觉模型头上会误导用户）。所以这里钉的是
    两个谓词各自覆盖哪些码，以及那个相反偏置本身。
    """

    #: 会转移、且确实来自供应商的错误码。
    DRIFTING_CODES = [
        "invalid_llm_response",
        "empty_provider_response",
        "reasoning_exhausted_max_tokens",
        "invalid_provider_response",
    ]

    #: 从未见过的供应商错误码：值得转移（未知即尝试），但**不归咎**
    #: （未知即不是 provider）。这两条把两个谓词相反的偏置钉在一起。
    UNSEEN_PROVIDER_CODES = [
        "provider_malformed_sse",
        "a_brand_new_provider_error",
    ]

    @pytest.mark.parametrize("code", UNSEEN_PROVIDER_CODES)
    def test_an_unseen_provider_code_is_worth_trying_but_not_blamed(
        self, code: str
    ) -> None:
        assert is_failover_eligible(code), "未知故障必须值得换一家再试"
        assert not the_provider_is_the_suspect(code), (
            "但归咎是闭集：我们没见过它，就不能断定是 provider 的错"
        )

    @pytest.mark.parametrize("code", DRIFTING_CODES)
    def test_a_drifting_code_is_the_providers_fault(self, code: str) -> None:
        assert the_provider_is_the_suspect(code)

    @pytest.mark.parametrize("code", DRIFTING_CODES)
    async def test_a_drifting_code_raises_the_shared_provider_alert(
        self, hass: HomeAssistant, code: str
    ) -> None:
        """两家都失败后落到这里的每一个码，都必须能在「修复」里被看见。"""
        runtime = await _runtime(hass)

        runtime.record_error(code)

        assert (
            "frigate_vision",
            f"{_ENTRY_ID}_provider_error",
        ) in _issues(hass), f"{code} 失败了却没有任何告警"
        assert runtime.last_error == code

    @pytest.mark.parametrize(
        "code",
        [
            "invalid_evidence_image",
            "label_malformed",
            "analysis_outcome_unknown",
            "stage_conflict",
            "evidence_incomplete",
            # 我们自己的媒体构建用尽了重试次数：换一家视觉供应商毫无帮助。
            "media_retry_exhausted",
        ],
    )
    async def test_a_purely_local_code_raises_no_provider_alert(
        self, hass: HomeAssistant, code: str
    ) -> None:
        """本地错误不能让用户去查供应商 —— 那是误导。"""
        runtime = await _runtime(hass)

        runtime.record_error(code)

        assert (
            "frigate_vision",
            f"{_ENTRY_ID}_provider_error",
        ) not in _issues(hass), f"{code} 不是供应商的问题，不该报 provider_error"

    @pytest.mark.parametrize(
        "code",
        [
            # Frigate 离线与媒体清理失败各有自己的修复项，不能被并进供应商告警。
            "frigate_unavailable",
            "media_cleanup_failed",
        ],
    )
    async def test_other_transient_failures_keep_their_own_repair(
        self, hass: HomeAssistant, code: str
    ) -> None:
        """`record_error` 的输入域比分析路径宽得多，不能一并算作供应商故障。

        这两个码来自 Frigate 客户端与媒体清理，跟视觉供应商无关。把它们并进
        `provider_error` 会让「供应商在报错」这条修复在 Frigate 掉线时也亮起来。
        """
        runtime = await _runtime(hass)

        runtime.record_error(code)

        assert ("frigate_vision", f"{_ENTRY_ID}_{code}") in _issues(hass)
        assert (
            "frigate_vision",
            f"{_ENTRY_ID}_provider_error",
        ) not in _issues(hass), f"{code} 不该冒充供应商故障"

    @pytest.mark.parametrize(
        "code",
        [
            # 关联引擎的错误经由 entry worker -> handle_error -> record_error 到达，
            # 全部由我们自己的代码产生，跟供应商无关。
            "ambiguous_review_ownership",
            "ingress_identity_mismatch",
            "event_id_missing",
        ],
    )
    async def test_a_correlation_fault_raises_no_provider_alert(
        self, hass: HomeAssistant, code: str
    ) -> None:
        """worker 的兜底路径会把任意异常喂给 `record_error`。

        这些码不在「转移拒绝清单」里，却也从不出现在转移循环中——它们是关联
        引擎在**分析之前**抛的。仅用「非本地错误」来判定供应商故障，会让一条
        「供应商在报错」的修复因为一次 review 归属歧义而亮起。
        """
        runtime = await _runtime(hass)

        runtime.record_error(code)

        assert (
            "frigate_vision",
            f"{_ENTRY_ID}_provider_error",
        ) not in _issues(hass), f"{code} 是关联引擎的问题，不是供应商的"

    async def test_clearing_retires_a_drifting_code_too(
        self, hass: HomeAssistant
    ) -> None:
        """能报出来就必须能收回去，否则修复项会永远挂着。"""
        runtime = await _runtime(hass)
        runtime.record_error("invalid_llm_response")
        assert runtime.last_error == "invalid_llm_response"

        runtime.clear_error("provider_error")

        assert runtime.last_error is None
        assert (
            "frigate_vision",
            f"{_ENTRY_ID}_provider_error",
        ) not in _issues(hass)

    async def test_the_alert_still_fires_for_the_statuses_it_always_did(
        self, hass: HomeAssistant
    ) -> None:
        """放宽判断不能弄丢原有覆盖：5xx 与 429 依旧报修。"""
        for code in ("provider_http_503", "provider_http_500", "provider_http_429"):
            runtime = await _runtime(hass)
            runtime.record_error(code)
            assert (
                "frigate_vision",
                f"{_ENTRY_ID}_provider_error",
            ) in _issues(hass), code

    @pytest.mark.parametrize(
        "code",
        ["provider_http_500", "provider_http_503", "provider_http_429"],
    )
    def test_the_narrow_predicate_is_still_the_narrower_one(self, code: str) -> None:
        """两个判断的关系被钉住：窄的那个只回答「确定没做这件事」。

        它保留下来是因为重放决策要问的正是这个更窄的问题 —— 契约失败是供应商的
        错，却说明不了有没有被计费。这里断言包含关系，防止有人图省事把它也换成
        宽判断（那会让 `invalid_llm_response` 被当成「可以安全重放」）。
        """
        assert is_provider_side_failure(code)
        assert the_provider_is_the_suspect(code)

    @pytest.mark.parametrize(
        "code",
        [
            "invalid_llm_response",
            "empty_provider_response",
            "invalid_provider_response",
        ],
    )
    def test_a_contract_failure_is_the_providers_fault_but_not_safely_replayable(
        self, code: str
    ) -> None:
        """宽判断为真而窄判断为假，正是两者必须并存的原因。"""
        assert the_provider_is_the_suspect(code), "答坏了是供应商的问题"
        assert not is_provider_side_failure(code), (
            "但它答了，可能已经计费 —— 不能当成「确定没做」"
        )

    async def test_both_providers_failing_reaches_the_alert_through_the_real_path(
        self,
        hass: HomeAssistant,
        tmp_path: Any,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """端到端：两家都失败 -> 活动 FAILED -> 修复项亮起。

        单元测试断言 `record_error` 的行为，这一条把 `_schedule_analysis` 的接线
        也串起来。分开测过、合起来断掉的先例在本项目已经有过。
        """
        store, client, _session = await _wire(
            hass,
            tmp_path,
            monkeypatch,
            primary_script=[(503, "primary down")],
            fallback_script=[(500, "backup broke")],
        )

        entry = MockConfigEntry(
            domain="frigate_vision",
            title="Front Door",
            data={},
            options={},
            entry_id=_ENTRY_ID,
        )
        entry.add_to_hass(hass)

        async def handler(message: object) -> None:
            return None

        runtime = IntegrationRuntime(
            hass=hass,
            store=store,
            queue=EntryRuntime(queue_size=1, handler=handler),
            analysis_tasks={},
            entry_id=_ENTRY_ID,
        )
        runtime.vision = client

        record = store.get("activity_1")
        assert record is not None
        runtime._schedule_analysis(entry, record)
        await asyncio.gather(*(runtime.analysis_tasks or {}).values())

        assert (
            "frigate_vision",
            f"{_ENTRY_ID}_provider_error",
        ) in _issues(hass), "两家供应商都失败，用户必须看到告警"
        assert runtime.last_error == "provider_http_500"
        failed = store.get("activity_1")
        assert failed is not None
        assert failed.stage is ActivityStage.FAILED
        assert failed.error_code == "provider_http_500"


class TestMediaRetryExhaustedIsNotProvidersFault:
    """`media_retry_exhausted` 属于本地，不属于供应商。

    它在简报的漂移清单里，但语义上不是供应商失败：它表示**我们自己的**媒体构建
    （Frigate 拉流 / 录像 / 文件写入）用尽了重试。此时视觉供应商根本还没被调用，
    所以：

    * 不该并进 `provider_error` 告警（否则 Frigate 掉线会让用户去查模型）；
    * 也不该让转移循环去试第二组 —— 第二组拿到的是同一堵墙。
    """

    def test_it_is_not_the_providers_fault(self) -> None:
        assert not the_provider_is_the_suspect("media_retry_exhausted")

    def test_it_is_not_failover_eligible(self) -> None:
        assert not is_failover_eligible("media_retry_exhausted")

    def test_it_stays_replayable_by_hand(self) -> None:
        """改判不能伤到人工重放：它仍然是用户可安全重试的。"""
        assert retry_is_safe("media_retry_exhausted")
