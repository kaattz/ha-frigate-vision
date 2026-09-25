"""Reproduce the two reported bugs through the real flow.

Both are reported from the UI, so this drives the options flow the way the
frontend does -- submitting every field the form displayed, with the URL left at
its shown value -- rather than a minimal hand-built dict. A minimal dict can pass
while the real submission fails, because the schema injects defaults for absent
keys and that injection is what the guard keys off.
"""

from __future__ import annotations

from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from pytest_homeassistant_custom_component.common import MockConfigEntry

# Importing the module is what registers the integration with the config-entry
# flow manager; without it the flow manager raises UnknownHandler.
from custom_components.frigate_vision import config_flow  # noqa: F401

DOMAIN = "frigate_vision"

# The live entry: a local router URL that matches no preset.
LIVE = {
    "processing_mode": "live",
    "llm_base_url": "http://192.168.166.50:7864/v1",
    "llm_api_key": "k",
    "llm_model": "Deepseek-V4.1-Flash",
    "llm_reasoning_effort": "high",
    "llm_thinking": "default",
    "target_width": 767,
    "max_tokens": 20000,
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


async def _open_settings(hass: HomeAssistant, options: dict):
    entry = MockConfigEntry(domain=DOMAIN, title="Front Door", data={}, options=options)
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    assert result["step_id"] == "settings"
    return entry, result


def _suggested_for(schema, field: str):
    for marker, _selector in schema.schema.items():
        if getattr(marker, "schema", None) == field:
            description = marker.description
            if isinstance(description, dict):
                return description.get("suggested_value")
            return description
    raise AssertionError(f"{field} not in schema")


async def test_bug1_the_form_shows_the_provider_that_matches_the_stored_url(
    hass: HomeAssistant,
) -> None:
    """A local-router URL matches no preset, so the form must show "custom".

    `llm_provider` was never stored (it did not exist when this entry was made),
    so the field falls back to the schema default `deepseek` and the dialog claims
    a provider the entry is not using.
    """
    _entry, result = await _open_settings(hass, dict(LIVE))
    assert _suggested_for(result["data_schema"], "llm_provider") == "custom", (
        "the stored URL is a local router, matching no preset"
    )


async def test_bug1b_a_preset_url_still_shows_its_own_provider(
    hass: HomeAssistant,
) -> None:
    """The inference must not break the normal case."""
    options = dict(LIVE, llm_base_url="https://api.deepseek.com/v1")
    _entry, result = await _open_settings(hass, options)
    assert _suggested_for(result["data_schema"], "llm_provider") == "deepseek"


async def test_bug2_switching_provider_writes_the_url_in_one_submit(
    hass: HomeAssistant,
) -> None:
    """Selecting Gemini and submitting must save Gemini's URL immediately.

    One submit, not two. Home Assistant's native form runs no server code when a
    dropdown is *selected* -- only on submit -- so a design that re-rendered the
    form to show a prefilled URL did nothing visible: the user picked Gemini,
    submitted, and neither the URL nor the entry changed.

    Submits the full displayed form (as the browser does), with the URL left at
    the value shown, because the schema carries a default and an untouched box
    never arrives empty.
    """
    entry, result = await _open_settings(hass, dict(LIVE))
    submitted = dict(entry.options)
    submitted["llm_provider"] = "gemini"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submitted
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY, (
        "the save must complete in this submit, not come back as a form"
    )
    assert result["data"]["llm_base_url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert result["data"]["llm_provider"] == "gemini"


async def test_bug2c_a_hand_typed_url_survives_a_provider_change(
    hass: HomeAssistant,
) -> None:
    """A URL the user typed must not be replaced by the preset.

    Picking a provider is a shortcut for filling the URL in, not an override: on
    this deployment the URL points at a local router, and rewriting it to a public
    endpoint would break a working setup. The user typed a URL that differs from
    what the form showed, which is the signal that it is theirs.
    """
    entry, result = await _open_settings(hass, dict(LIVE))
    submitted = dict(entry.options)
    submitted["llm_provider"] = "gemini"
    submitted["llm_base_url"] = "http://10.0.0.9:9999/v1"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submitted
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["llm_base_url"] == "http://10.0.0.9:9999/v1", (
        "a hand-typed URL must win over the preset"
    )


async def test_bug2b_a_no_op_save_does_not_hijack_the_url(
    hass: HomeAssistant,
) -> None:
    """Saving without touching the provider must keep the local-router URL.

    Otherwise any unrelated edit would rewrite the endpoint to whatever provider
    the inferred field happens to hold -- turning a working local endpoint into a
    public one.
    """
    entry, result = await _open_settings(hass, dict(LIVE))
    submitted = dict(entry.options)
    submitted["llm_provider"] = "custom"
    submitted["prompt_override"] = "只按可见动作判断。"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submitted
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["llm_base_url"] == LIVE["llm_base_url"]
    assert result["data"]["prompt_override"] == "只按可见动作判断。"
