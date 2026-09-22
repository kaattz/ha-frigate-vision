from __future__ import annotations

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from homeassistant.core import HomeAssistant

from custom_components.frigate_vision.frigate import (
    FrigatePayloadError,
    async_subscribe_frigate,
    parse_event_payload,
    parse_review_payload,
)
from custom_components.frigate_vision.models import IngressKind


def test_event_and_review_payloads_are_whitelisted() -> None:
    event = parse_event_payload(
        json.dumps(
            {
                "type": "update",
                "after": {
                    "id": "event_1",
                    "camera": "front",
                    "label": "person",
                    "frame_time": 10,
                    "current_zones": ["near", "ignored"],
                    "entered_zones": ["near"],
                },
            }
        ),
        entry_id="entry_1",
        camera="front",
        allowed_zones={"near", "far"},
    )
    assert event is not None
    assert event.kind is IngressKind.FRIGATE_EVENT
    assert event.current_zones == ("near",)
    review = parse_review_payload(
        json.dumps(
            {
                "type": "end",
                "after": {
                    "id": "review_1",
                    "camera": "front",
                    "start_time": 15,
                    "end_time": 20,
                    "severity": "alert",
                    "data": {
                        "detections": ["event_1"],
                        "objects": ["person"],
                        "zones": ["far"],
                    },
                },
            }
        ),
        entry_id="entry_1",
        camera="front",
        allowed_zones={"near", "far"},
    )
    assert review is not None
    assert review.detection_ids == ("event_1",)
    assert review.started_at == 15


def test_non_person_and_unrelated_review_are_skipped() -> None:
    assert (
        parse_event_payload(
            json.dumps(
                {
                    "type": "end",
                    "after": {
                        "id": "event_1",
                        "camera": "front",
                        "label": "car",
                        "frame_time": 10,
                        "current_zones": [],
                        "entered_zones": [],
                    },
                }
            ),
            entry_id="entry_1",
            camera="front",
            allowed_zones={"near"},
        )
        is None
    )
    assert (
        parse_review_payload(
            json.dumps(
                {
                    "type": "end",
                    "after": {
                        "id": "review_1",
                        "camera": "front",
                        "end_time": 20,
                        "severity": "alert",
                        "data": {
                            "detections": ["event_1"],
                            "objects": ["person"],
                            "zones": ["other"],
                        },
                    },
                }
            ),
            entry_id="entry_1",
            camera="front",
            allowed_zones={"near"},
        )
        is None
    )


def test_invalid_json_fails_explicitly() -> None:
    with pytest.raises(FrigatePayloadError, match="invalid_json"):
        parse_event_payload(
            "not-json", entry_id="entry_1", camera="front", allowed_zones={"near"}
        )


def test_mqtt_array_fields_reject_strings_instead_of_iterating_them() -> None:
    with pytest.raises(FrigatePayloadError, match="invalid_event_payload"):
        parse_event_payload(
            json.dumps(
                {
                    "type": "new",
                    "after": {
                        "id": "event_1",
                        "camera": "front",
                        "label": "person",
                        "frame_time": 10,
                        "current_zones": "near",
                        "entered_zones": [],
                    },
                }
            ),
            entry_id="entry_1",
            camera="front",
            allowed_zones={"near"},
        )
    with pytest.raises(FrigatePayloadError, match="invalid_review_payload"):
        parse_review_payload(
            json.dumps(
                {
                    "type": "end",
                    "after": {
                        "id": "review_1",
                        "camera": "front",
                        "start_time": 1,
                        "end_time": 2,
                        "data": {
                            "detections": "event_1",
                            "objects": ["person"],
                            "zones": ["near"],
                        },
                    },
                }
            ),
            entry_id="entry_1",
            camera="front",
            allowed_zones={"near"},
        )


def test_review_missing_start_time_fails_explicitly() -> None:
    with pytest.raises(FrigatePayloadError, match="invalid_review_payload"):
        parse_review_payload(
            json.dumps(
                {
                    "type": "end",
                    "after": {
                        "id": "review_1",
                        "camera": "front",
                        "end_time": 2,
                        "data": {
                            "detections": ["event_1"],
                            "objects": ["person"],
                            "zones": ["near"],
                        },
                    },
                }
            ),
            entry_id="entry_1",
            camera="front",
            allowed_zones={"near"},
        )


def test_review_rejects_non_string_array_members() -> None:
    with pytest.raises(FrigatePayloadError, match="invalid_review_payload"):
        parse_review_payload(
            json.dumps(
                {
                    "type": "end",
                    "after": {
                        "id": "review_1",
                        "camera": "front",
                        "start_time": 1,
                        "end_time": 2,
                        "data": {
                            "detections": ["event_1"],
                            "objects": ["person"],
                            "zones": [["near"]],
                        },
                    },
                }
            ),
            entry_id="entry_1",
            camera="front",
            allowed_zones={"near"},
        )


async def test_mqtt_subscription_lifecycle(hass: HomeAssistant) -> None:
    def unsub_events() -> None:
        return None

    def unsub_reviews() -> None:
        return None

    subscribe = AsyncMock(side_effect=[unsub_events, unsub_reviews])
    with patch(
        "custom_components.frigate_vision.frigate.async_subscribe",
        subscribe,
    ):
        unsubscribe = await async_subscribe_frigate(
            hass, "frigate", lambda payload: None, lambda payload: None
        )
    assert [call.args[1] for call in subscribe.await_args_list] == [
        "frigate/events",
        "frigate/reviews",
    ]
    unsubscribe()


async def test_second_mqtt_subscription_failure_rolls_back_first(
    hass: HomeAssistant,
) -> None:
    unsubscribe_events = Mock()
    subscribe = AsyncMock(
        side_effect=[unsubscribe_events, RuntimeError("reviews_subscribe_failed")]
    )
    with (
        patch(
            "custom_components.frigate_vision.frigate.async_subscribe",
            subscribe,
        ),
        pytest.raises(RuntimeError, match="reviews_subscribe_failed"),
    ):
        await async_subscribe_frigate(
            hass, "frigate", lambda payload: None, lambda payload: None
        )
    unsubscribe_events.assert_called_once_with()


def test_zero_zone_person_review_is_kept_when_analyzing_all() -> None:
    message = parse_review_payload(
        json.dumps(
            {
                "type": "end",
                "after": {
                    "id": "review_1",
                    "camera": "front",
                    "start_time": 15,
                    "end_time": 20,
                    "data": {
                        "detections": ["event_1"],
                        "objects": ["person"],
                        "zones": ["other"],
                    },
                },
            }
        ),
        entry_id="entry_1",
        camera="front",
        allowed_zones={"near"},
        analyze_all_person_reviews=True,
    )
    assert message is not None
    assert message.current_zones == ()
    assert message.detection_ids == ("event_1",)


def test_zone_filter_preserved_when_not_analyzing_all_reviews() -> None:
    message = parse_review_payload(
        json.dumps(
            {
                "type": "end",
                "after": {
                    "id": "review_1",
                    "camera": "front",
                    "start_time": 15,
                    "end_time": 20,
                    "data": {
                        "detections": ["event_1"],
                        "objects": ["person"],
                        "zones": ["other"],
                    },
                },
            }
        ),
        entry_id="entry_1",
        camera="front",
        allowed_zones={"near"},
        analyze_all_person_reviews=False,
    )
    assert message is None


def test_person_review_without_detections_is_still_rejected() -> None:
    with pytest.raises(FrigatePayloadError, match="invalid_review_payload"):
        parse_review_payload(
            json.dumps(
                {
                    "type": "end",
                    "after": {
                        "id": "review_1",
                        "camera": "front",
                        "start_time": 15,
                        "end_time": 20,
                        "data": {
                            "detections": [],
                            "objects": ["person"],
                            "zones": ["other"],
                        },
                    },
                }
            ),
            entry_id="entry_1",
            camera="front",
            allowed_zones={"near"},
            analyze_all_person_reviews=True,
        )
