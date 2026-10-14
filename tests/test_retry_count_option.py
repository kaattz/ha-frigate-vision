"""The retry count is a deployment setting, and reading it must never misfire.

`vision_retry_count` decides how many requests one failed activity may cost, and
it arrives from a JSON round trip that the form does not fully control. The cases
below are the shapes that value can actually come back in.
"""

from __future__ import annotations

import pytest

from custom_components.frigate_vision.const import (
    CONF_VISION_RETRY_COUNT,
    CONF_VISION_RETRY_COUNT_DEFAULT,
    CONF_VISION_RETRY_COUNT_MAX,
)
from custom_components.frigate_vision.vision import (
    fallback_config_from,
    pick_retry_count,
    vision_config_from,
)

_BASE_DATA = {
    "llm_base_url": "https://api.example.com/v1",
    "llm_api_key": "secret",
    "llm_model": "vision-model",
}


def test_the_default_is_no_retry() -> None:
    """An unset option means "fail over immediately".

    This is the incident's conclusion, so it is pinned: the old hardcoded 4 made
    one failure cost 8 requests, and 8 alone exceeded the ~7 requests of headroom
    a typical day left against the 20/day free tier.
    """
    assert CONF_VISION_RETRY_COUNT_DEFAULT == 0
    assert pick_retry_count({}, {}) == 0


@pytest.mark.parametrize("value", [0, 1, 2, 5])
def test_a_stored_int_is_read_back(value: int) -> None:
    assert pick_retry_count({CONF_VISION_RETRY_COUNT: value}, {}) == value


def test_options_win_over_data() -> None:
    """Matches every other provider setting: the Options page is authoritative."""
    assert (
        pick_retry_count(
            {CONF_VISION_RETRY_COUNT: 3}, {CONF_VISION_RETRY_COUNT: 1}
        )
        == 3
    )


@pytest.mark.parametrize("value", ["2", " 2 ", 2.0])
def test_a_stringified_or_float_value_is_still_honoured(value: object) -> None:
    """The value crosses a JSON round trip, so its type is not guaranteed."""
    assert pick_retry_count({CONF_VISION_RETRY_COUNT: value}, {}) == 2


@pytest.mark.parametrize("value", ["", None, "abc", "two", [], {}, "1e3"])
def test_an_unreadable_value_falls_back_to_the_default(value: object) -> None:
    """A value that cannot be read must not be able to mean "retry forever"."""
    assert pick_retry_count({CONF_VISION_RETRY_COUNT: value}, {}) == 0


def test_a_bool_is_refused_rather_than_meaning_one() -> None:
    """`True` is an `int` in Python, so it needs its own branch.

    Without it, a form or an automation that stored `true` would silently become
    "retry once", which is a different budget from either the default or an
    explicit number.
    """
    assert pick_retry_count({CONF_VISION_RETRY_COUNT: True}, {}) == 0
    assert pick_retry_count({CONF_VISION_RETRY_COUNT: False}, {}) == 0


def test_the_value_is_clamped_to_the_supported_range() -> None:
    """An out-of-range value is clamped, not trusted.

    A negative cannot reach the request builder, and an absurdly large one cannot
    turn one activity into an unbounded spend.
    """
    assert pick_retry_count({CONF_VISION_RETRY_COUNT: -5}, {}) == 0
    assert (
        pick_retry_count({CONF_VISION_RETRY_COUNT: 999}, {})
        == CONF_VISION_RETRY_COUNT_MAX
    )


def test_the_count_reaches_the_vision_config() -> None:
    """The option must actually travel to the analyser, via `data` too.

    `vision_config_from` reads options first and data second, and the retry count
    has to follow that rule like every other field -- otherwise an entry
    configured during the initial flow would silently keep the default.
    """
    from_options = vision_config_from(_BASE_DATA, {CONF_VISION_RETRY_COUNT: 3})
    assert from_options.retry_count == 3

    from_data = vision_config_from({**_BASE_DATA, CONF_VISION_RETRY_COUNT: 2}, {})
    assert from_data.retry_count == 2

    unset = vision_config_from(_BASE_DATA, {})
    assert unset.retry_count == CONF_VISION_RETRY_COUNT_DEFAULT


@pytest.mark.parametrize(
    ("data", "options", "expected"),
    [
        ({}, {}, 0),
        ({CONF_VISION_RETRY_COUNT: 2}, {}, 2),
        ({CONF_VISION_RETRY_COUNT: 2}, {CONF_VISION_RETRY_COUNT: 3}, 3),
    ],
)
def test_retry_budget_reaches_both_provider_configs(
    data: dict, options: dict, expected: int
) -> None:
    configured = {
        **_BASE_DATA,
        "fallback_llm_base_url": "https://backup.example.com/v1",
        "fallback_llm_api_key": "test-key",
        "fallback_llm_model": "backup-vision",
        **data,
    }
    primary = vision_config_from(configured, options)
    fallback = fallback_config_from(configured, options)
    assert fallback is not None
    assert primary.retry_count == fallback.retry_count == expected
