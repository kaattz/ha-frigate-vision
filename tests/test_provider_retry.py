"""Policy for transient provider failures.

A provider that answers 5xx has not processed the request, so trying again is
safe -- and it is the difference between a recovered activity and a lost one.

Measured on this deployment: `gemini-3.8-flash` behind Google's
OpenAI-compatible endpoint answered `503` for the 20:35 activity, and had
already answered `503` on three earlier replays of the previous one. Every one
of those activities was discarded with no log line, no repair and no
notification, so the user's only evidence was a notification that never came.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest

from custom_components.frigate_vision.vision import (
    PROVIDER_RETRY_ATTEMPTS,
    VisionConfig,
    VisionError,
    async_request_with_retry,
)

_LOGGER_NAME = "custom_components.frigate_vision.vision"


def _config() -> VisionConfig:
    return VisionConfig(
        base_url="https://api.example.com/v1",
        api_key="secret",
        model="vision-model",
        thinking="disabled",
    )


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


class _ScriptedSession:
    """Answer a scripted list of (status, body), then repeat the last one.

    Repeating the last entry is what lets a test describe a provider that is
    *persistently* down without having to know the retry cap in advance.
    """

    def __init__(self, responses: list[tuple[int, str]]) -> None:
        self._responses = responses
        self.calls = 0

    def post(self, url: str, *, json: dict[str, Any], **_kwargs: Any) -> _Response:
        index = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        status, body = self._responses[index]
        return _Response(status, body)


class _Sleeper:
    """Record the backoff waits instead of serving them."""

    def __init__(self) -> None:
        self.waits: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.waits.append(seconds)


def _ok() -> tuple[int, str]:
    return 200, json.dumps({"choices": [{"message": {"content": "OK"}}]})


async def test_a_transient_5xx_is_retried_until_the_provider_answers() -> None:
    """A 503 must be tried again rather than abandoning the activity.

    503 means the provider was busy and did not process the request, which is
    exactly the case a second attempt fixes.
    """
    session = _ScriptedSession(
        [(503, "overloaded"), (503, "overloaded"), _ok()],
    )
    sleeper = _Sleeper()

    body = await async_request_with_retry(  # type: ignore[arg-type]
        session, _config(), {"model": "vision-model"}, sleep=sleeper
    )

    assert body["choices"], "the successful attempt's body must be returned"
    assert session.calls == 3, "two 503s then success must take three requests"


async def test_a_persistent_5xx_gives_up_after_a_bounded_number_of_attempts() -> None:
    """The retry must be capped.

    An uncapped loop would hang the analysis forever on a provider that is down
    for the night, and would bill a request every backoff interval while doing
    it. The cap is the whole point of the bound.
    """
    session = _ScriptedSession([(503, "down")])

    with pytest.raises(VisionError, match="provider_http_503"):
        await async_request_with_retry(  # type: ignore[arg-type]
            session, _config(), {"model": "vision-model"}, sleep=_Sleeper()
        )

    assert session.calls == PROVIDER_RETRY_ATTEMPTS


async def test_the_backoff_waits_grow_exponentially() -> None:
    """Waiting is what gives an overloaded provider time to recover.

    Pinned as literals rather than recomputed from the constants: the schedule
    is a policy decision (how long a user waits before an activity is declared
    lost), so changing it should require changing this test on purpose.
    """
    session = _ScriptedSession([(503, "down")])
    sleeper = _Sleeper()

    with pytest.raises(VisionError):
        await async_request_with_retry(  # type: ignore[arg-type]
            session, _config(), {"model": "vision-model"}, sleep=sleeper
        )

    assert sleeper.waits == [2.0, 6.0, 18.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
async def test_a_permanent_status_is_not_retried(status: int) -> None:
    """A rejected key or a wrong model name is not fixed by trying again.

    Retrying these would triple the latency of a failure the user has to fix by
    hand, and would make the reported status harder to trust.
    """
    session = _ScriptedSession([(status, "nope")])

    with pytest.raises(VisionError, match=f"provider_http_{status}"):
        await async_request_with_retry(  # type: ignore[arg-type]
            session, _config(), {"model": "vision-model"}, sleep=_Sleeper()
        )

    assert session.calls == 1


async def test_the_retry_logs_the_status_and_the_providers_own_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The failure has to be visible, and it has to say what the provider said.

    The provider's prose is what separates an overloaded endpoint from a bad
    model name from a rejected field; it was previously read and thrown away, so
    a production failure left no trace at all.
    """
    session = _ScriptedSession([(503, "model is overloaded, try later")])

    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        with pytest.raises(VisionError):
            await async_request_with_retry(  # type: ignore[arg-type]
                session, _config(), {"model": "vision-model"}, sleep=_Sleeper()
            )

    messages = [record.getMessage() for record in caplog.records]
    assert any("503" in message for message in messages), messages
    assert any(
        "model is overloaded" in message for message in messages
    ), f"the provider's own message must be logged, got {messages}"


async def test_the_giving_up_error_code_stays_safe_to_persist() -> None:
    """The persisted `error_code` must remain `provider_http_<status>`.

    It is validated against `SAFE_ID` when written to the record, so the
    provider's prose cannot travel in it -- it would fail the write and turn a
    clear provider failure into `analysis_outcome_unknown`.
    """
    session = _ScriptedSession([(503, "理由：模型过载")])

    with pytest.raises(VisionError) as caught:
        await async_request_with_retry(  # type: ignore[arg-type]
            session, _config(), {"model": "vision-model"}, sleep=_Sleeper()
        )

    assert str(caught.value) == "provider_http_503"
