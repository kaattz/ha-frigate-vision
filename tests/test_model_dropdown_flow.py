"""The model dropdown, driven through the real options flow.

Separate from `test_model_list.py` (which model ids come back) and
`test_model_field.py` (what the field looks like): this file covers the wiring --
which endpoint each field queries, when a cached answer is reused, and that the
manual refresh really re-queries.

The provider is stubbed at `async_fetch_model_ids`, one level below the cache, so
the cache logic under test is the real one. Stubbing `async_model_choices` instead
would test nothing: it is the function whose behaviour these assertions describe.

Network access is never allowed here. The existing suite keeps a stubbed provider
for the same reason (`test_provider_ui_bugs.py`): a real call would depend on the
internet, slow CI, and fail for reasons unrelated to the assertions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.frigate_vision import config_flow
from custom_components.frigate_vision.const import (
    CONF_FALLBACK_LLM_MODEL,
    CONF_LLM_MODEL,
)

DOMAIN = "frigate_vision"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEEPSEEK_URL = "https://api.deepseek.com/v1"

BASE = {
    "llm_base_url": GEMINI_URL,
    "llm_api_key": "primary-key",
    "llm_model": "gemini-3.5-flash",
    "llm_reasoning_effort": "default",
    "llm_thinking": "default",
    "target_width": 767,
    "max_tokens": 4000,
    "output_language": "zh-CN",
    "scene_description": "",
    "scene_labels": "",
    "prompt_override": "",
    "history_retention_days": 30,
    "media_retention_days": 7,
    "queue_size": 10,
    "min_review_seconds": 5,
    "analyze_night_unknown": False,
    "analyze_all_far_reviews": True,
}


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """Start every test with an empty cache.

    The cache is module state that outlives a test, so without this a test would
    see the previous test's answer and the assertions about request counts would
    pass or fail depending on execution order.
    """
    config_flow._MODEL_LIST_CACHE.clear()  # noqa: SLF001
    yield
    config_flow._MODEL_LIST_CACHE.clear()  # noqa: SLF001


def _field(schema, name: str):
    for marker, selector_ in schema.schema.items():
        if getattr(marker, "schema", None) == name:
            return selector_
    raise AssertionError(f"{name} is not in the schema")


def _options(schema, name: str) -> list[str]:
    selector_ = _field(schema, name)
    assert type(selector_).__name__ == "SelectSelector", (
        f"{name} 应是下拉框，实际 {type(selector_).__name__}"
    )
    return list(selector_.config["options"])


async def _open_settings_form(hass: HomeAssistant, entry: MockConfigEntry):
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    return await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )


async def test_choosing_gemini_offers_geminis_models(hass: HomeAssistant) -> None:
    """The feature's headline behaviour, through the real flow.

    Before this change the model was a text box prefilled with one preset name, so
    the other models the key can reach were invisible unless the user already knew
    them.
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids",
        AsyncMock(return_value=("gemini-3.8-flash", "gemini-3.5-flash")),
    ):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "provider"}
        )
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "provider_gemini"}
        )
    options = _options(result["data_schema"], CONF_LLM_MODEL)
    assert "gemini-3.8-flash" in options
    assert "gemini-3.5-flash" in options


async def test_the_stored_model_survives_an_empty_lookup(hass: HomeAssistant) -> None:
    """A failed lookup must not offer to change the model.

    The deployment's own case: `gemini-3.5-flash` is stored, and it is absent from
    the built-in preset list. If the field's options dropped it, the form would
    render as "nothing selected" and the next unrelated save would silently move
    the entry onto a different model.
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids",
        AsyncMock(return_value=()),
    ):
        result = await _open_settings_form(hass, entry)
    assert "gemini-3.5-flash" in _options(result["data_schema"], CONF_LLM_MODEL)


async def test_a_failed_lookup_does_not_show_a_form_error(hass: HomeAssistant) -> None:
    """Silent degradation, asserted on the flow result rather than in prose.

    A lookup failure is not a configuration error: the model name is validated by
    the analysis path against the API that actually matters, and a red field here
    would send the user to inspect a working configuration.
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids",
        AsyncMock(return_value=()),
    ):
        result = await _open_settings_form(hass, entry)
    assert result["type"] is FlowResultType.FORM
    assert not result.get("errors"), f"不该有表单错误：{result.get('errors')!r}"


async def test_both_groups_are_queried_with_their_own_endpoint(
    hass: HomeAssistant,
) -> None:
    """Two providers, two catalogues.

    Offering the primary's models under the fallback's field would invite a name
    the second provider does not serve, and the mistake would only surface during
    a failover -- the moment the user can least afford it.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={
            **BASE,
            "fallback_llm_base_url": DEEPSEEK_URL,
            "fallback_llm_api_key": "fallback-key",
            "fallback_llm_model": "deepseek-v4-pro",
        },
    )
    entry.add_to_hass(hass)
    seen: list[str] = []

    async def _fetch(_session, *, base_url, api_key, **_kw):  # noqa: ANN001
        seen.append(base_url)
        return ("m-from-primary",) if base_url == GEMINI_URL else ("m-from-fallback",)

    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids", _fetch
    ):
        result = await _open_settings_form(hass, entry)

    # The options are the union of online, preset and stored, so what is asserted
    # is which endpoint each group ASKED -- not that the list equals the stub's
    # return value, which the preset and the stored model also contribute to.
    primary = _options(result["data_schema"], CONF_LLM_MODEL)
    fallback = _options(result["data_schema"], CONF_FALLBACK_LLM_MODEL)
    assert "m-from-primary" in primary
    assert "m-from-fallback" not in primary, "主组不该出现备用组的模型"
    assert "m-from-fallback" in fallback
    assert "m-from-primary" not in fallback, "备用组不该出现主组的模型"
    assert set(seen) == {GEMINI_URL, DEEPSEEK_URL}


async def test_two_groups_pointing_at_one_endpoint_ask_once(
    hass: HomeAssistant,
) -> None:
    """This deployment's configuration: both groups on the same URL and key.

    Measured on the live entry, so it is the common case rather than a corner.
    Each form render asks for both lists; without the cache that is two identical
    HTTP requests every time the form is opened or re-rendered on a validation
    error.
    """
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options={
            **BASE,
            "fallback_llm_base_url": GEMINI_URL,
            "fallback_llm_api_key": BASE["llm_api_key"],
            "fallback_llm_model": "gemini-3.5-flash",
        },
    )
    entry.add_to_hass(hass)
    fetch = AsyncMock(return_value=("gemini-3.8-flash",))
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids", fetch
    ):
        await _open_settings_form(hass, entry)
    assert fetch.await_count == 1, (
        f"同址同 key 应只请求一次，实际 {fetch.await_count} 次"
    )


async def test_a_second_render_reuses_the_cache(hass: HomeAssistant) -> None:
    """Reopening the form within the TTL must not re-query."""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    fetch = AsyncMock(return_value=("gemini-3.8-flash",))
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids", fetch
    ):
        await _open_settings_form(hass, entry)
        await _open_settings_form(hass, entry)
    assert fetch.await_count == 1


async def test_a_validation_error_keeps_the_dropdown(hass: HomeAssistant) -> None:
    """The re-render on a validation error must rebuild the same field shape.

    Otherwise the form the user sees after being told something else was wrong has
    a different model control than the one before it -- reading as the feature
    intermittently not working, which is exactly the class of bug this module
    already has a test file named after.
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids",
        AsyncMock(return_value=("gemini-3.8-flash",)),
    ):
        result = await _open_settings_form(hass, entry)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {**BASE, "scene_labels": "not a valid label line"},
        )
    assert result["type"] is FlowResultType.FORM
    assert result.get("errors"), "这次提交应当因标签格式而报错"
    # Asserted as membership + shape, not equality: the options are the union of
    # online, the preset and the stored model, and what this test is about is that
    # the control is still a dropdown with the fetched names in it.
    selector_ = _field(result["data_schema"], CONF_LLM_MODEL)
    assert type(selector_).__name__ == "SelectSelector", "报错重渲染后不能退回文本框"
    assert "gemini-3.8-flash" in selector_.config["options"]


async def test_the_manual_refresh_requeries_despite_the_cache(
    hass: HomeAssistant,
) -> None:
    """The refresh entry exists precisely to defeat the cache."""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    fetch = AsyncMock(return_value=("gemini-3.8-flash",))
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids", fetch
    ):
        await _open_settings_form(hass, entry)
        assert fetch.await_count == 1
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "refresh_models"}
        )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "refresh_models"
    assert fetch.await_count == 2, "刷新必须绕过缓存重新请求"


async def test_the_refresh_reports_both_counts(hass: HomeAssistant) -> None:
    """Counts are actionable; a bare "refreshed" is not.

    "primary 3, fallback 0" tells the user the fallback has no endpoint configured.
    A success message would leave them believing both were fine.

    The exact primary count is the union of the stub's two names, the gemini
    preset's one, and the stored model -- which is itself the same as one of the
    stub's, so three. Pinned as a number because the placeholder is what the user
    reads; a change to the merge order or the preset would show up here.
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids",
        AsyncMock(return_value=("gemini-3.8-flash", "gemini-2.5-flash")),
    ):
        result = await hass.config_entries.options.async_init(entry.entry_id)
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {"next_step_id": "refresh_models"}
        )
    assert result["description_placeholders"]["primary"] == "3"
    assert result["description_placeholders"]["fallback"] == "0"


async def test_a_blank_password_box_still_queries_with_the_stored_key(
    hass: HomeAssistant,
) -> None:
    """The password box renders empty even when a key is stored.

    Querying with the rendered value would use "" for every entry that already has
    a key -- that is, every entry after the first save -- and silently degrade the
    dropdown to the built-in list for a perfectly healthy configuration.
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    seen: list[str] = []

    async def _fetch(_session, *, base_url, api_key, **_kw):  # noqa: ANN001
        seen.append(api_key)
        return ("gemini-3.8-flash",)

    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids", _fetch
    ):
        result = await _open_settings_form(hass, entry)
        # Submit the form as the browser would: the key box is left empty.
        result = await hass.config_entries.options.async_configure(
            result["flow_id"], {**BASE, "llm_api_key": ""}
        )
    assert seen and seen[0] == "primary-key", f"应以已存的 key 查询，实际 {seen!r}"


async def test_a_typed_key_wins_over_the_stored_one(hass: HomeAssistant) -> None:
    """A newly typed key must be the one used on the re-render, not the old one.

    Otherwise the user has just corrected their credential and the list still
    reflects the previous one -- the case where a stale list is most confusing.

    Driven through the ERROR path deliberately: a successful submit closes the flow
    (CREATE_ENTRY) and renders nothing, so the only render that sees a typed key is
    the one after a validation failure. That is also the common case in practice --
    a user fixing a key and a label in the same edit.
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(BASE)
    )
    entry.add_to_hass(hass)
    seen: list[str] = []

    async def _fetch(_session, *, base_url, api_key, **_kw):  # noqa: ANN001
        seen.append(api_key)
        return ("gemini-3.8-flash",)

    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids", _fetch
    ):
        result = await _open_settings_form(hass, entry)
        assert seen == ["primary-key"], "首次渲染应使用已存的 key"
        result = await hass.config_entries.options.async_configure(
            result["flow_id"],
            {
                **BASE,
                "llm_api_key": "brand-new-key",
                "scene_labels": "not a valid label line",
            },
        )
    assert result.get("errors"), "这次提交应因标签格式而报错并重渲染"
    assert "brand-new-key" in seen, f"重渲染应使用新输入的 key，实际 {seen!r}"
