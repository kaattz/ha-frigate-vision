"""A provider failure has to be visible and recoverable.

Two halves of the same gap:

* **Visible.** A 5xx was recorded on a sensor and nowhere else. The user's only
  evidence was a notification that never arrived, so there was nothing to
  investigate -- no log line, no repair, no record of what the provider said.
* **Recoverable.** The status was not in `retry_is_safe`, so the explicit
  `frigate_vision.retry_failed` service refused it too. Once the automatic retry
  gave up, the activity was unreachable by any route.
"""

from __future__ import annotations

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from custom_components.frigate_vision.models import (
    is_automatically_retryable,
    is_server_error,
    retry_is_safe,
)
from custom_components.frigate_vision.repairs import async_set_issue
from custom_components.frigate_vision.runtime import EntryRuntime, IntegrationRuntime
from custom_components.frigate_vision.store import ActivityStore

ENTRY_ID = "entry_1"


@pytest.mark.parametrize("status", [500, 502, 503, 504])
def test_a_server_error_is_safe_to_replay_by_hand(status: int) -> None:
    """A person asking for one more attempt must be allowed to have it.

    The manual service exists precisely for the case where the automatic policy
    declined: the user has weighed the cost, and the worst case is one wasted
    call. Refusing here left a provider outage with no recovery path at all.
    """
    assert retry_is_safe(f"provider_http_{status}")


@pytest.mark.parametrize("status", [502, 503, 504])
def test_only_the_gateway_statuses_are_retried_automatically(status: int) -> None:
    """Automatic retries must stay narrower than the manual ones.

    Each of these means the provider did not process the request, so trying
    again cannot double-bill.
    """
    assert is_automatically_retryable(f"provider_http_{status}")


def test_a_bare_500_is_not_retried_automatically() -> None:
    """A 500 admits a fault but says nothing about whether the request ran.

    That is the same uncertainty that makes `analysis_outcome_unknown` unsafe to
    replay, so it is recorded and left to the explicit, human-initiated retry.
    """
    assert not is_automatically_retryable("provider_http_500")
    assert retry_is_safe("provider_http_500"), "but a person may still ask"


@pytest.mark.parametrize(
    "code",
    ["analysis_outcome_unknown", "delivery_outcome_unknown", "invented"],
)
def test_the_uncertain_stages_stay_unsafe(code: str) -> None:
    """Replaying an uncertain side effect must remain refused.

    The model may already have been billed and the notification may already have
    been sent; a replay would send it twice.
    """
    assert not retry_is_safe(code)


def test_a_connection_failure_is_now_replayable() -> None:
    """`provider_unavailable` was the one transient error with no recovery path.

    The provider was never reached, so nothing can have been billed -- yet it
    was absent from the safe set, so an outage could not be replayed by hand.
    """
    assert retry_is_safe("provider_unavailable")


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("provider_http_503", True),
        ("provider_http_500", True),
        ("provider_http_401", False),
        ("provider_unavailable", False),
        ("frigate_unavailable", False),
        ("analysis_outcome_unknown", False),
    ],
)
def test_only_5xx_counts_as_a_server_error(code: str, expected: bool) -> None:
    """`is_server_error` must not mistake other codes for a status.

    It gates the repair, so a false positive would raise an alert about the wrong
    thing -- and `invented` has no digits to read at all.
    """
    assert is_server_error(code) is expected


async def test_a_5xx_raises_a_repair_the_user_can_see(hass: HomeAssistant) -> None:
    """The failure has to reach Settings > Repairs.

    Recorded on a memory-only sensor it was invisible: the sensor already held
    the same value from an earlier failure, so even its `last_changed` did not
    move and nothing on the dashboard changed.
    """
    store = ActivityStore(hass, ENTRY_ID)
    await store.async_load()

    async def handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        entry_id=ENTRY_ID,
    )
    runtime.record_error("provider_http_503")

    registry = ir.async_get(hass)
    assert (
        "frigate_vision",
        f"{ENTRY_ID}_provider_error",
    ) in registry.issues
    # The raw status stays on the sensor: it is the specific diagnosis, and the
    # repair is the actionable summary keyed to a translation.
    assert runtime.last_error == "provider_http_503"


async def test_the_repair_clears_once_an_analysis_succeeds(
    hass: HomeAssistant,
) -> None:
    """A stale repair would train the user to ignore the list.

    The next success proves the provider is answering again.
    """
    store = ActivityStore(hass, ENTRY_ID)
    await store.async_load()

    async def handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        entry_id=ENTRY_ID,
    )
    runtime.record_error("provider_http_503")
    registry = ir.async_get(hass)
    assert ("frigate_vision", f"{ENTRY_ID}_provider_error") in registry.issues

    runtime.clear_error("provider_error")

    assert ("frigate_vision", f"{ENTRY_ID}_provider_error") not in registry.issues
    assert runtime.last_error is None


async def test_the_repair_is_translated_in_every_shipped_language() -> None:
    """A repair with no translation shows its raw key in the UI.

    The translation files are the only place a user reads the explanation, so a
    missing entry makes the alert unreadable rather than helpful.
    """
    from custom_components.frigate_vision.repairs import ISSUES

    assert "provider_error" in ISSUES
    for name in ("strings.json", "translations/en.json", "translations/zh-Hans.json"):
        import json
        from pathlib import Path

        path = Path("custom_components/frigate_vision") / name
        issues = json.loads(path.read_text(encoding="utf-8"))["issues"]
        assert "provider_error" in issues, f"{name} has no provider_error issue"
        assert issues["provider_error"]["title"], f"{name} title is empty"
        assert issues["provider_error"]["description"], f"{name} description is empty"


async def test_an_unknown_repair_code_is_still_rejected(
    hass: HomeAssistant,
) -> None:
    """The allowlist must keep rejecting codes that have no translation."""
    with pytest.raises(ValueError, match="unknown_repair_issue"):
        async_set_issue(hass, ENTRY_ID, "provider_http_503")


async def test_a_lost_503_activity_can_actually_be_replayed(
    hass: HomeAssistant,
) -> None:
    """The end-to-end promise, not just the predicate.

    `retry_is_safe` returning True is only half of it: the store also has to
    accept the record, and the resulting attempt has to land where a re-analysis
    can pick it up. Asserting the predicate alone would pass even if the store
    refused the handoff -- which is exactly the kind of gap that let the original
    bug hide behind green unit tests.

    The attempt must land on EVIDENCE_READY, not SEALED: the sheet still exists,
    so the retry re-runs the model call and skips re-collecting media. Proving
    that also proves the evidence survived, which is what makes the replay worth
    offering at the user's request hours later.
    """
    from custom_components.frigate_vision.models import (
        ActivityRecord,
        ActivitySource,
        ActivityStage,
        ProcessingMode,
    )
    from custom_components.frigate_vision.store import ActivityStore

    store = ActivityStore(hass, ENTRY_ID)
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="review_lost",
            entry_id=ENTRY_ID,
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.FAILED,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            evidence_mode="review_six",
            evidence_path="/tmp/sheet.png",
            error_code="provider_http_503",
        )
    )

    retry = await store.async_create_retry("review_lost", now=3)

    assert retry.stage is ActivityStage.EVIDENCE_READY
    assert retry.error_code is None, "the retry must not inherit the old failure"
    assert retry.activity_id != "review_lost"
    assert retry.classification is None


async def test_a_lost_503_activity_without_evidence_is_recollected(
    hass: HomeAssistant,
) -> None:
    """A replay whose sheet was already cleaned up must rebuild it.

    Evidence expires on `media_retention_days` (7 by default). A user replaying a
    week-old failure would otherwise hand the model a path that no longer exists.
    """
    from custom_components.frigate_vision.models import (
        ActivityRecord,
        ActivitySource,
        ActivityStage,
        ProcessingMode,
    )
    from custom_components.frigate_vision.store import ActivityStore

    store = ActivityStore(hass, ENTRY_ID)
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="review_expired",
            entry_id=ENTRY_ID,
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.FAILED,
            processing_mode=ProcessingMode.LIVE,
            created_at=1,
            updated_at=2,
            camera="front",
            evidence_mode="review_six",
            evidence_path="/tmp/gone.png",
            evidence_expired_at=50,
            error_code="provider_http_503",
        )
    )

    retry = await store.async_create_retry("review_expired", now=3)

    assert retry.stage is ActivityStage.SEALED, "media must be rebuilt first"


def test_a_quota_error_is_replayable_but_not_auto_retried() -> None:
    """A 429 is the same *failure* class as the 503, with a different remedy.

    Measured on this deployment's own provider while verifying the capability fix:

        "Quota exceeded for metric:
         generativelanguage.googleapis.com/generate_content_free_tier_requests,
         limit: 20, model: gemini-3.8-flash
         ... Please retry in 39.152574657s."

    A *daily* free-tier cap of 20 requests. Two consequences, and both are the bug
    this file exists for:

    * It is neither 5xx nor in `retry_is_safe`, so `retry_failed` refuses it and no
      repair is raised. An exhausted quota therefore reproduced the original
      symptom exactly -- Frigate triggers, nothing is reported, nothing says why --
      while the integration looked healthy.
    * Automatic retrying is deliberately *not* added. The cap is per day, so a
      26-second backoff cannot outlast it; retrying would spend the next window's
      budget rather than recover. Manual replay once the window resets is the
      honest remedy, and that is the route that was missing.
    """
    assert not is_server_error("provider_http_429"), (
        "429 is not a 5xx; if this changes, the repair keyed on is_server_error "
        "would silently start or stop covering quota errors"
    )
    assert not is_automatically_retryable("provider_http_429"), (
        "a daily quota cannot be waited out by a 26-second backoff"
    )
    assert retry_is_safe("provider_http_429"), (
        "but a person must be able to replay it once the window resets"
    )


async def test_a_quota_error_raises_a_repair(hass: HomeAssistant) -> None:
    """The user has to be told, because the fix is not something HA can do.

    Recovering from a quota error needs a human decision -- wait for the window, or
    move to a paid plan. Without a repair the integration would quietly stop
    analysing activities, which is indistinguishable from a quiet night.
    """
    store = ActivityStore(hass, ENTRY_ID)
    await store.async_load()

    async def handler(message: object) -> None:
        return None

    runtime = IntegrationRuntime(
        hass=hass,
        store=store,
        queue=EntryRuntime(queue_size=1, handler=handler),
        entry_id=ENTRY_ID,
    )
    runtime.record_error("provider_http_429")

    registry = ir.async_get(hass)
    assert ("frigate_vision", f"{ENTRY_ID}_provider_error") in registry.issues
