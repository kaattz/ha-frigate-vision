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
