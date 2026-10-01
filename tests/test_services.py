from __future__ import annotations

import io
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from homeassistant.core import Context, HomeAssistant
from PIL import Image
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision.models import (
    ActivityRecord,
    ActivitySource,
    ActivityStage,
)
from custom_components.frigate_vision.runtime import (
    EntryRuntime,
    IntegrationRuntime,
)
from custom_components.frigate_vision.services import (
    async_register_services,
)
from custom_components.frigate_vision.store import ActivityStore
from custom_components.frigate_vision.vision import VisionClient, VisionConfig

_ENTRY_ID = "entry_1"
_ACTIVITY_ID = "activity_1"

# Full completions paths, so `VisionConfig.endpoint()` returns them unchanged and
# the fake session can route by URL without re-deriving anything.
_PRIMARY_URL = "https://primary.example.com/v1/chat/completions"
_FALLBACK_URL = "https://backup.example.com/v1/chat/completions"

#: Every key `get_activity` is allowed to return. Pinned as a whole set rather
#: than one assertion per field, so a field added "just for diagnostics" has to
#: be a deliberate edit here instead of slipping into the contract unnoticed.
_EXPECTED_KEYS = frozenset(
    {
        "activity_id",
        "stage",
        "classification",
        "description",
        "confidence",
        "error_code",
        "evidence_url",
        "evidence_expired",
        "review_ids",
        "provider",
    }
)


async def test_get_activity_returns_only_whitelisted_state(hass: HomeAssistant) -> None:
    store = ActivityStore(hass, "entry_1")
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id="activity_1",
            entry_id="entry_1",
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.COMPLETED,
            created_at=1,
            updated_at=2,
            camera="front",
            classification="visitor",
            description="有人到访。",
            confidence=80,
        )
    )

    async def handler(value: object) -> None:
        return None

    queue = EntryRuntime(queue_size=1, handler=handler)
    await queue.async_start()
    runtime = IntegrationRuntime(hass=hass, store=store, queue=queue)
    entry = MockConfigEntry(domain="frigate_vision", title="Front", entry_id="entry_1")
    entry.add_to_hass(hass)
    entry.runtime_data = runtime
    await async_register_services(hass)
    response = await hass.services.async_call(
        "frigate_vision",
        "get_activity",
        {"entry_id": "entry_1", "activity_id": "activity_1"},
        blocking=True,
        return_response=True,
    )
    assert response["classification"] == "visitor"
    assert "evidence_path" not in response
    assert "claimed_side_effects" not in response
    await queue.async_stop()


# --------------------------------------------------------------------------- #
# The provider the service reports must be the one that actually answered.
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


class _Response:
    """One scripted reply, shaped like the aiohttp slice `async_request` uses."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _RoutingSession:
    """Answer per-URL, so the two providers are distinguishable in one call.

    Routing by URL rather than by call order matters here for the same reason it
    does in `test_fallback_provider`: an ordering-based fake would silently swap
    which provider it was answering as, and the assertion "the fallback
    answered" would then be testing the fake instead of the failover loop.
    """

    def __init__(self, by_url: dict[str, list[tuple[int, str]]]) -> None:
        self._by_url = by_url
        self.calls: list[str] = []

    def post(self, url: str, *, json: dict[str, Any], **_kwargs: Any) -> _Response:
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


async def _wire_analysis(
    hass: HomeAssistant,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_script: list[tuple[int, str]],
    fallback_script: list[tuple[int, str]] | None = None,
) -> tuple[IntegrationRuntime, VisionClient, _RoutingSession]:
    """A registered runtime holding a real vision client over a scripted session.

    The analysis itself is run by `VisionClient`, not simulated: the provider the
    service reports is then the one the failover loop actually recorded, rather
    than one this test wrote into `_last_provider` by hand.
    """
    sheet = tmp_path / f"{_ACTIVITY_ID}.png"
    sheet.write_bytes(_png(640, 240))

    store = ActivityStore(hass, _ENTRY_ID)
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id=_ACTIVITY_ID,
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

    by_url: dict[str, list[tuple[int, str]]] = {_PRIMARY_URL: primary_script}
    if fallback_script is not None:
        by_url[_FALLBACK_URL] = fallback_script
    session = _RoutingSession(by_url)
    monkeypatch.setattr(
        "custom_components.frigate_vision.vision.async_get_clientsession",
        lambda _hass: session,
    )
    client = VisionClient(
        hass,
        store,
        _provider_config(_PRIMARY_URL, "primary-model"),
        (
            _provider_config(_FALLBACK_URL, "backup-model")
            if fallback_script is not None
            else None
        ),
    )

    async def handler(value: object) -> None:
        return None

    queue = EntryRuntime(queue_size=1, handler=handler)
    await queue.async_start()
    runtime = IntegrationRuntime(
        hass=hass, store=store, queue=queue, vision=client, entry_id=_ENTRY_ID
    )
    entry = MockConfigEntry(domain="frigate_vision", title="Front", entry_id=_ENTRY_ID)
    entry.add_to_hass(hass)
    entry.runtime_data = runtime
    await async_register_services(hass)
    return runtime, client, session


async def _get_activity(hass: HomeAssistant) -> dict[str, Any]:
    response = await hass.services.async_call(
        "frigate_vision",
        "get_activity",
        {"entry_id": _ENTRY_ID, "activity_id": _ACTIVITY_ID},
        blocking=True,
        return_response=True,
    )
    assert response is not None
    return dict(response)


async def test_get_activity_reports_which_provider_answered(
    hass: HomeAssistant,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """排障的第一个问题是「这条是谁答的」。"""
    runtime, client, session = await _wire_analysis(
        hass,
        tmp_path,
        monkeypatch,
        primary_script=[(200, _ok_body("主组的描述。"))],
        fallback_script=[(200, _ok_body("备用组的描述。"))],
    )

    done = await client.async_analyze(_ACTIVITY_ID)

    assert done.stage is ActivityStage.ANALYSIS_DONE
    # The premise of the assertion below: the primary really did answer alone,
    # so `"primary"` here is a measurement and not a default that happens to
    # agree with an unused fallback.
    assert session.calls_to(_PRIMARY_URL) == 1
    assert session.calls_to(_FALLBACK_URL) == 0
    assert client.provider_for(_ACTIVITY_ID) == "primary"

    response = await _get_activity(hass)

    assert response["provider"] == "primary"
    assert set(response) == _EXPECTED_KEYS
    await runtime.queue.async_stop()


async def test_get_activity_reports_the_fallback_when_it_answered(
    hass: HomeAssistant,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """转移过的那条必须说「备用组答的」，否则排障时无从分辨。

    A 500 rather than a 503: it is failover-eligible but not in
    `AUTOMATIC_RETRY_STATUSES`, so the primary is called exactly once and the
    test reaches the fallback without serving the 26-second backoff schedule.
    """
    runtime, client, session = await _wire_analysis(
        hass,
        tmp_path,
        monkeypatch,
        primary_script=[(500, "primary down")],
        fallback_script=[(200, _ok_body("备用组答对了。"))],
    )

    done = await client.async_analyze(_ACTIVITY_ID)

    assert done.stage is ActivityStage.ANALYSIS_DONE
    assert done.description == "备用组答对了。"
    assert session.calls_to(_FALLBACK_URL) == 1
    assert client.provider_for(_ACTIVITY_ID) == "fallback"

    response = await _get_activity(hass)

    assert response["provider"] == "fallback"
    assert response["description"] == "备用组答对了。"
    assert set(response) == _EXPECTED_KEYS
    await runtime.queue.async_stop()


async def test_get_activity_reports_no_provider_before_analysis(
    hass: HomeAssistant,
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """尚未分析时该字段为 None，而不是猜一个。"""
    runtime, client, _session = await _wire_analysis(
        hass,
        tmp_path,
        monkeypatch,
        primary_script=[(200, _ok_body("主组的描述。"))],
        # Both providers are configured, so a `None` here cannot be explained
        # away by there being only one of them: this entry simply has not asked
        # the question yet.
        fallback_script=[(200, _ok_body("备用组的描述。"))],
    )

    assert client.provider_for(_ACTIVITY_ID) is None

    response = await _get_activity(hass)

    assert "provider" in response, "字段必须在，不能因为还没分析就消失"
    assert response["provider"] is None
    assert set(response) == _EXPECTED_KEYS
    await runtime.queue.async_stop()


async def _register_with_one_activity(hass: HomeAssistant):
    """Register the services over a runtime holding one sealed activity."""
    store = ActivityStore(hass, _ENTRY_ID)
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id=_ACTIVITY_ID,
            entry_id=_ENTRY_ID,
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.SEALED,
            created_at=1,
            updated_at=2,
            camera="front",
        )
    )

    async def handler(value: object) -> None:
        return None

    queue = EntryRuntime(queue_size=1, handler=handler)
    await queue.async_start()
    runtime = IntegrationRuntime(
        hass=hass, store=store, queue=queue, entry_id=_ENTRY_ID
    )
    entry = MockConfigEntry(domain="frigate_vision", title="Front", entry_id=_ENTRY_ID)
    entry.add_to_hass(hass)
    entry.runtime_data = runtime
    await async_register_services(hass)
    return runtime, queue


async def test_a_non_admin_cannot_call_the_services(hass: HomeAssistant) -> None:
    """非管理员不得调用这些服务——其中两个会花掉机主的额度。

    `retry_failed` 与 `process_review` 都会真的触发一次分析（计费），`get_activity`
    返回"谁在摄像头前、穿了什么"的描述。HA 不会替我们拦：`call_service` 这个
    WebSocket 命令没有 `require_admin`，而 `helpers/service.py` 的实体权限检查只对
    **带实体目标**的服务生效——这些服务用的是 `entry_id`，`services.yaml` 里也没有
    实体选择器，所以什么都不跑。

    因此检查必须落在服务自己身上，而 `context.user_id` 是服务能拿到的唯一调用者身份。
    """
    runtime, queue = await _register_with_one_activity(hass)

    # A minimal stand-in: only `is_admin` is read, and constructing the real
    # `auth.models.User` needs a permission lookup and groups that add nothing here.
    non_admin = SimpleNamespace(is_admin=False)
    hass.auth.async_get_user = AsyncMock(return_value=non_admin)  # type: ignore[method-assign]

    with pytest.raises(Exception) as caught:
        await hass.services.async_call(
            "frigate_vision",
            "get_activity",
            {"entry_id": _ENTRY_ID, "activity_id": _ACTIVITY_ID},
            blocking=True,
            return_response=True,
            context=Context(user_id="non-admin-1"),
        )
    assert "admin_required" in str(caught.value), (
        f"非管理员被放进来了，实际异常：{caught.value!r}"
    )
    await queue.async_stop()


async def test_an_admin_can_call_the_services(hass: HomeAssistant) -> None:
    """管理员必须照常可用——否则这个门禁就是把功能关掉了。"""
    runtime, queue = await _register_with_one_activity(hass)
    hass.auth.async_get_user = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(is_admin=True)
    )

    response = await hass.services.async_call(
        "frigate_vision",
        "get_activity",
        {"entry_id": _ENTRY_ID, "activity_id": _ACTIVITY_ID},
        blocking=True,
        return_response=True,
        context=Context(user_id="admin-1"),
    )
    assert response is not None
    await queue.async_stop()


async def test_an_unknown_user_id_is_refused(hass: HomeAssistant) -> None:
    """查不到的 user_id 必须拒绝，而不是放行。

    默认方向要选"关"：一个已删除用户的 context 不能因为"查不到"就被当作管理员。
    """
    runtime, queue = await _register_with_one_activity(hass)
    hass.auth.async_get_user = AsyncMock(return_value=None)  # type: ignore[method-assign]

    with pytest.raises(Exception) as caught:
        await hass.services.async_call(
            "frigate_vision",
            "get_activity",
            {"entry_id": _ENTRY_ID, "activity_id": _ACTIVITY_ID},
            blocking=True,
            return_response=True,
            context=Context(user_id="deleted-user"),
        )
    assert "admin_required" in str(caught.value)
    await queue.async_stop()


async def test_a_call_without_a_user_is_still_allowed(hass: HomeAssistant) -> None:
    """没有 user_id 的调用必须放行——自动化与蓝图正是这样调的。

    随集成发布的蓝图用 `frigate_vision.ack_delivery` 收尾通知，而自动化没有用户身
    份。若把「没有 user_id」判为拒绝，交付流程会当场断掉。
    """
    runtime, queue = await _register_with_one_activity(hass)
    hass.auth.async_get_user = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("a user-less call must not look up a user")
    )

    response = await hass.services.async_call(
        "frigate_vision",
        "get_activity",
        {"entry_id": _ENTRY_ID, "activity_id": _ACTIVITY_ID},
        blocking=True,
        return_response=True,
    )
    assert response is not None, "自动化/蓝图的调用被门禁挡住了"
    await queue.async_stop()


async def test_get_activity_reports_no_provider_when_vision_is_not_configured(
    hass: HomeAssistant,
) -> None:
    """没有视觉客户端时该字段为 None，而不是让整个服务调用失败。

    这是每个只配了 Frigate、还没填视觉服务的 entry 的常态：`runtime.vision`
    为 None。此时读数不存在（真的没问过任何人），但它绝不能把 `get_activity`
    整个打挂——那会让排障的人连活动记录都看不到。
    """
    store = ActivityStore(hass, _ENTRY_ID)
    await store.async_load()
    await store.async_create(
        ActivityRecord(
            activity_id=_ACTIVITY_ID,
            entry_id=_ENTRY_ID,
            source=ActivitySource.STANDALONE_REVIEW,
            stage=ActivityStage.SEALED,
            created_at=1,
            updated_at=2,
            camera="front",
        )
    )

    async def handler(value: object) -> None:
        return None

    queue = EntryRuntime(queue_size=1, handler=handler)
    await queue.async_start()
    runtime = IntegrationRuntime(
        hass=hass, store=store, queue=queue, entry_id=_ENTRY_ID
    )
    assert runtime.vision is None
    entry = MockConfigEntry(domain="frigate_vision", title="Front", entry_id=_ENTRY_ID)
    entry.add_to_hass(hass)
    entry.runtime_data = runtime
    await async_register_services(hass)

    response = await _get_activity(hass)

    assert response["provider"] is None
    assert set(response) == _EXPECTED_KEYS
    await queue.async_stop()
