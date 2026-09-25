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


async def test_bug2_switching_provider_prefills_the_url_on_submit(
    hass: HomeAssistant,
) -> None:
    """Selecting Gemini and submitting must return the form with Gemini's URL.

    Submits the full displayed form (as the browser does), not a minimal dict:
    the URL is not empty, it carries the value the form showed. A guard that
    treats "non-empty" as "the user typed this" never fires here.
    """
    entry, result = await _open_settings(hass, dict(LIVE))
    submitted = dict(entry.options)
    submitted["llm_provider"] = "gemini"
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submitted
    )
    assert result["type"] is FlowResultType.FORM, (
        "the form must come back so the user sees the filled-in URL"
    )
    assert _suggested_for(result["data_schema"], "llm_base_url") == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
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
