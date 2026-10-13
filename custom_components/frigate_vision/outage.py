"""Alert the owner when every vision provider has failed.

A provider outage used to be visible only as a Repair card, and that turned out
not to be visible enough: on 2026-10-13 the card appeared at 15:08 and the owner
did not notice until 18:00, by which point five activities had been lost. A card
sits in Settings waiting to be looked at; the failure it describes is "the thing
you rely on has stopped working", which deserves to arrive on its own.

So this module pushes to the same channel the activity notifications already use.
Two properties matter more than the wording:

* **Once per outage, not once per failure.** Every failing activity would
  otherwise send its own message -- six on the day of the incident -- and a burst
  of identical alerts is how a channel gets muted, which restores the original
  problem with the notification still technically working. The reset is driven by
  success: the next outage is free to notify again.
* **It can never break the analysis.** The push is a side channel. It is sent
  from a background task, the service name is resolved defensively, and every
  failure is logged rather than raised -- a notification that cannot be delivered
  must not also cost the user their activity.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from homeassistant.core import HomeAssistant

_LOGGER = logging.getLogger(__name__)

# The deployment's own notification script. It wraps a phone push plus the
# persisted list this owner already reads, so the outage arrives wherever the
# activity notifications do rather than inventing a second channel.
#
# A constant rather than an option: it is one line to change if the script is
# renamed, and a form field for it would be another row on a twenty-field options
# page to configure something that has exactly one sensible value here. Absence is
# tolerated rather than assumed -- see `async_notify_failure`.
OUTAGE_NOTIFY_SERVICE = "script.alert_notify"

OUTAGE_TITLE = "Frigate Vision 分析中断"


class OutageNotifier:
    """Send at most one notification per continuous provider outage.

    Not thread-safe by design: it is only ever touched from the event loop, and
    the flag is set without an intervening await, so two activities failing at the
    same moment cannot both claim the notification.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        *,
        service: str = OUTAGE_NOTIFY_SERVICE,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._hass = hass
        self._service = service
        self._clock = clock
        self._notified = False
        self._outage_started_at: float | None = None

    @property
    def notified(self) -> bool:
        """Whether the current outage has already been announced."""
        return self._notified

    async def async_notify_failure(self, code: str, *, camera: str = "") -> bool:
        """Announce a provider outage, at most once until `reset`.

        Returns True when this call was the one that sent the notification, which
        is what a test asserts against; production ignores it.
        """
        if self._notified:
            return False
        # Set before the first await. `async_analyze` runs one task per activity,
        # so a second failure can be sitting at this exact line; claiming the flag
        # synchronously is what keeps the pair from both sending.
        self._notified = True
        self._outage_started_at = self._clock()
        self._hass.async_create_task(
            self._async_send(code, camera=camera),
            f"frigate_vision-outage-{code}",
        )
        return True

    def reset(self) -> None:
        """Clear the outage, so the next one notifies again.

        Called when a provider answers. Driven by success rather than by a timer
        because only success proves the outage is over: a timer would either
        re-notify during a long outage or stay silent through a second one.
        """
        self._notified = False
        self._outage_started_at = None

    async def _async_send(self, code: str, *, camera: str) -> None:
        """Deliver the notification, swallowing everything that can go wrong."""
        if not self._hass.services.has_service(*self._service.split(".", 1)):
            # Not configured, or renamed. Logged at warning because the whole
            # point of this module is that the failure was invisible, and silent
            # absence would recreate it.
            _LOGGER.warning(
                "Cannot send the provider-outage notification: service %s is not "
                "registered. The Repair card is still raised.",
                self._service,
            )
            return
        domain, service = self._service.split(".", 1)
        try:
            await self._hass.services.async_call(
                domain,
                service,
                {
                    "title": OUTAGE_TITLE,
                    "message": self._message(code, camera=camera),
                    "level": "critical",
                },
                blocking=True,
            )
        except Exception:  # noqa: BLE001
            # Deliberately broad: this is the last step of a failure path whose
            # real product -- the Repair card and the recorded error -- has
            # already been produced. An exception here would replace a visible
            # outage with an invisible crash.
            _LOGGER.exception(
                "Failed to send the provider-outage notification via %s",
                self._service,
            )

    def _message(self, code: str, *, camera: str) -> str:
        lines = [
            "视觉分析已连续失败，所有 provider 都无法作答，"
            "这段时间的活动不会产生通知。",
            f"错误：{code}",
        ]
        if camera:
            lines.append(f"摄像头：{camera}")
        lines.append(
            "已在「设置 → 系统 → 修复」留下卡片；"
            "provider 恢复后可用 frigate_vision.retry_failed 重放已丢失的活动。"
        )
        return "\n".join(lines)
