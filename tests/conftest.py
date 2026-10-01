from __future__ import annotations

import pytest
from homeassistant.config_entries import ConfigEntryState
from homeassistant.core import HomeAssistant


@pytest.fixture(autouse=True)
def auto_enable_custom_integrations(enable_custom_integrations: None) -> None:
    """Enable loading custom integrations in every test."""


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
