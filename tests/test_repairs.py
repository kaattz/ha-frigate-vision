import pytest
from homeassistant.core import HomeAssistant
from homeassistant.helpers import issue_registry as ir

from custom_components.frigate_vision.repairs import (
    async_clear_issue,
    async_set_issue,
)


async def test_unknown_repair_issue_is_rejected(hass: HomeAssistant) -> None:
    with pytest.raises(ValueError, match="unknown_repair_issue"):
        async_set_issue(hass, "entry_1", "invented")


async def test_repair_create_deduplicate_and_clear(hass: HomeAssistant) -> None:
    async_set_issue(hass, "entry_1", "frigate_unavailable")
    async_set_issue(hass, "entry_1", "frigate_unavailable")
    registry = ir.async_get(hass)
    assert (
        "frigate_vision",
        "entry_1_frigate_unavailable",
    ) in registry.issues
    async_clear_issue(hass, "entry_1", "frigate_unavailable")
    assert (
        "frigate_vision",
        "entry_1_frigate_unavailable",
    ) not in registry.issues
