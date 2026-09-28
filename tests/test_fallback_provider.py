"""故障转移的错误分类与转向逻辑。"""

from __future__ import annotations

import pytest

from custom_components.frigate_vision.models import (
    IMMEDIATE_FAILOVER_ERRORS,  # noqa: F401 - imported to fail loudly if the name disappears
    is_failover_eligible,
    is_failover_immediate,
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
