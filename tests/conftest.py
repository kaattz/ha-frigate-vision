from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable loading custom integrations in every test."""


@pytest.fixture(autouse=True)
def _no_live_model_discovery():
    """Stop a form render from reaching the network.

    The model dropdown queries the provider for its catalogue while the form is
    being built. That is deliberate in production -- best effort, 5 s timeout,
    degrades to the built-in list -- but it means any test that renders the
    settings form on an entry holding an api_key performs real I/O.

    This was a CI-only failure, and the reason is worth recording. The suite passes
    on Windows because `tests/windows_socket.py` neutralises
    `pytest_socket.disable_socket` (asyncio needs a socketpair for its self-pipe and
    Windows has no AF_UNIX one). CI runs on Linux with the stock guard, which
    rejects the connection and fails four pre-existing tests at teardown:

        ERROR tests/test_config_flow.py::test_a_valid_fallback_url_is_still_accepted
        ERROR tests/test_provider_ui_bugs.py::test_choosing_custom_changes_...
        ERROR tests/test_provider_ui_bugs.py::test_the_settings_form_no_longer_...
        ERROR tests/test_provider_ui_bugs.py::test_the_settings_menu_groups_...

    So a green local run could not see this, and neither could the review that
    approved the commit. Stubbing here fixes the class rather than the four: any
    future test that opens the form is covered, on both platforms.

    Patched at `config_flow.async_fetch_model_ids`, one level above the cache and
    below the callers, which is the same seam `tests/test_model_dropdown_flow.py`
    uses. A test that wants a specific catalogue patches over this and wins.
    """
    with patch(
        "custom_components.frigate_vision.config_flow.async_fetch_model_ids",
        AsyncMock(return_value=()),
    ) as stub:
        yield stub


@pytest.fixture(autouse=True)
async def _unload_entries_after_each_test(hass: HomeAssistant) -> None:
    """Unload every config entry a test left loaded.

    Saving options makes Home Assistant set the entry up, so a test that drives
    the options flow ends with a live runtime: a worker task and an EntityPlatform
    interval timer. Home Assistant's `verify_cleanup` fixture then reports both as
    leaks -- and because `HASocketBlockedError.instances` is only cleared in the
    teardown of the test that raised, a socket opened by an outliving task is
    blamed on whichever test happens to be running next. Both shapes show up as
    "ERROR at teardown" on tests whose own assertions all passed.

    Unloading here is the fix rather than a suppression: the integration's unload
    path stops exactly those resources, and this is the same call Home Assistant
    makes when a user removes or reloads the entry.
    """
    yield
    for entry in list(hass.config_entries.async_entries()):
        if entry.state is ConfigEntryState.LOADED:
            await hass.config_entries.async_unload(entry.entry_id)
    await hass.async_block_till_done()
