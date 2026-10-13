"""The outage push: once per outage, and never at the cost of the activity.

On 2026-10-13 the Repair card for a provider outage was raised at 15:08 and went
unnoticed until 18:00, with five activities lost in between. The card was working
as designed; it simply was not where the owner was looking. These tests pin the
two properties that make the replacement push trustworthy:

* one message per outage, because a burst of identical alerts is how a channel
  gets muted -- which restores the original problem with the notification still
  technically working;
* a push that fails must not fail the analysis, because the Repair card and the
  recorded error are the real product.
"""

from __future__ import annotations

from typing import Any

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.outage import (
    OUTAGE_NOTIFY_SERVICE,
    OUTAGE_TITLE,
    OutageNotifier,
)


class _Recorder:
    """Capture service calls instead of sending them."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def register(self, hass: HomeAssistant, name: str = OUTAGE_NOTIFY_SERVICE) -> None:
        domain, service = name.split(".", 1)

        async def handler(call: Any) -> None:
            self.calls.append((domain, service, dict(call.data)))

        hass.services.async_register(domain, service, handler)


async def test_the_first_failure_sends_exactly_one_message(
    hass: HomeAssistant,
) -> None:
    recorder = _Recorder()
    recorder.register(hass)
    notifier = OutageNotifier(hass)

    assert await notifier.async_notify_failure("provider_http_503") is True
    await hass.async_block_till_done()

    assert len(recorder.calls) == 1
    _, _, data = recorder.calls[0]
    assert data["title"] == OUTAGE_TITLE
    assert "provider_http_503" in data["message"], "the status must travel"


async def test_later_failures_in_the_same_outage_are_silent(
    hass: HomeAssistant,
) -> None:
    """Six failing activities must not mean six messages.

    This is the guarded property: without it the owner gets one alert per lost
    activity -- exactly the flood that trains a channel to be ignored -- so the
    count is asserted rather than the message.
    """
    recorder = _Recorder()
    recorder.register(hass)
    notifier = OutageNotifier(hass)

    for _ in range(6):
        await notifier.async_notify_failure("provider_http_503")
    await hass.async_block_till_done()

    assert len(recorder.calls) == 1, "one outage, one message"


async def test_a_recovered_provider_rearms_the_notification(
    hass: HomeAssistant,
) -> None:
    """Success is what ends an outage, so it is also what re-arms the push.

    Driven by success rather than a timer: a timer would either re-notify in the
    middle of one long outage or stay silent through a second, separate one.
    """
    recorder = _Recorder()
    recorder.register(hass)
    notifier = OutageNotifier(hass)

    await notifier.async_notify_failure("provider_http_503")
    await hass.async_block_till_done()
    notifier.reset()
    await notifier.async_notify_failure("provider_http_429")
    await hass.async_block_till_done()

    assert len(recorder.calls) == 2, "a second outage must be announced"


async def test_a_missing_notify_service_is_survivable(
    hass: HomeAssistant,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Renaming or removing the script must not break the analysis.

    The push is a side channel. Logged at warning rather than swallowed, because
    silence is the exact failure this module exists to fix -- but it must never
    raise into the caller.
    """
    notifier = OutageNotifier(hass)

    assert await notifier.async_notify_failure("provider_http_503") is True
    await hass.async_block_till_done()

    assert "not registered" in caplog.text


async def test_a_failing_service_call_does_not_propagate(
    hass: HomeAssistant,
) -> None:
    """A raising notify service must be logged, not re-raised."""
    domain, service = OUTAGE_NOTIFY_SERVICE.split(".", 1)

    async def explode(_call: Any) -> None:
        raise RuntimeError("channel down")

    hass.services.async_register(domain, service, explode)
    notifier = OutageNotifier(hass)

    assert await notifier.async_notify_failure("provider_http_503") is True
    await hass.async_block_till_done()
    assert notifier.notified is True


async def test_the_flag_is_set_before_any_await(hass: HomeAssistant) -> None:
    """Two activities failing at once must not both claim the notification.

    `async_analyze` runs one task per activity, so a second failure can be
    sitting at the same line. The flag is set synchronously for that reason, and
    this asserts the property that makes it work rather than the mechanism.
    """
    recorder = _Recorder()
    recorder.register(hass)
    notifier = OutageNotifier(hass)

    # Concurrent, not sequential: the await never gets a chance to interleave
    # before both calls decide.
    import asyncio

    results = await asyncio.gather(
        notifier.async_notify_failure("provider_http_503"),
        notifier.async_notify_failure("provider_http_503"),
    )
    await hass.async_block_till_done()

    assert results.count(True) == 1, "exactly one caller may win the claim"
    assert len(recorder.calls) == 1
