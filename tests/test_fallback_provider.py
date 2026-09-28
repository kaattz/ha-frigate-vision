"""故障转移的错误分类与转向逻辑。"""

from __future__ import annotations

import pytest

from custom_components.frigate_vision.models import (
    IMMEDIATE_FAILOVER_ERRORS,  # noqa: F401 - imported to fail loudly if the name disappears
    is_failover_eligible,
    is_failover_immediate,
)
from custom_components.frigate_vision.vision import (
    fallback_config_from,
    fallback_is_configured,
    vision_config_from,
)


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


class TestFailoverImmediacy:
    """契约类失败不重试：同一提示词重试大概率得到同样的越界答案。"""

    @pytest.mark.parametrize(
        "code",
        [
            "invalid_llm_response",
            "empty_provider_response",
            "reasoning_exhausted_max_tokens",
            "invalid_provider_response",
        ],
    )
    def test_contract_failures_skip_the_retry_budget(self, code: str) -> None:
        assert is_failover_immediate(code)

    @pytest.mark.parametrize(
        "code",
        [
            "provider_http_503",
            "provider_http_429",
            "provider_unavailable",
        ],
    )
    def test_transport_failures_use_the_retry_budget_first(self, code: str) -> None:
        assert not is_failover_immediate(code)


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
