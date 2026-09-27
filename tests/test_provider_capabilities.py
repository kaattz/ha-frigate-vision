"""Which optional body fields each provider actually accepts.

`thinking: {"type": "disabled"}` is a hard HTTP 400 on Google's
OpenAI-compatible endpoint:

    Invalid JSON payload received. Unknown name "thinking": Cannot find field.

That endpoint is the `gemini` preset, and the UI offers "disabled" as a
Reasoning mode for every provider -- so choosing it on this deployment made
*every* analysis fail with `provider_http_400`. A 400 is not retried (correctly:
the request itself is malformed), which made the option a trap rather than a
cost control.

`reasoning_effort` is deliberately *not* gated: probed against the same endpoint
and it is accepted by the schema -- the endpoint answers 429 on quota, which is
checked after validation, where `thinking` was rejected before quota was ever
consulted. Gating a field that works would remove a working cost control.
"""

from __future__ import annotations

import pytest

from custom_components.frigate_vision.const import (
    THINKING_UNSUPPORTED_PROVIDERS,
    provider_for_url,
)
from custom_components.frigate_vision.vision import VisionConfig, build_payload

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEEPSEEK_URL = "https://api.deepseek.com/v1"
GLM_URL = "https://open.bigmodel.cn/api/paas/v4"
OPENAI_URL = "https://api.openai.com/v1"
# A local router or reverse proxy: the provider cannot be named from the URL.
CUSTOM_URL = "http://192.168.166.50:7864/v1"


def _config(base_url: str, **overrides: object) -> VisionConfig:
    base: dict[str, object] = {
        "base_url": base_url,
        "api_key": "secret",
        "model": "some-model",
        "thinking": "default",
    }
    base.update(overrides)
    return VisionConfig(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        (GEMINI_URL, "gemini"),
        (DEEPSEEK_URL, "deepseek"),
        (GLM_URL, "glm"),
        (OPENAI_URL, "openai"),
        (GEMINI_URL + "/", "gemini"),
        ("https://generativelanguage.googleapis.com/v1beta/openai/", "gemini"),
        (CUSTOM_URL, "custom"),
        # An empty URL falls back to the schema default rather than to `custom`.
        # That is the pre-existing behaviour of the provider dropdown (it must show
        # *something*), and it is preserved here rather than "corrected": an empty
        # URL cannot produce a request at all -- `vision_is_configured` requires a
        # base URL -- so the choice has no effect on what gets sent, and changing
        # it would be an unrelated behaviour change riding along with this fix.
        ("", "deepseek"),
    ],
)
def test_the_provider_is_derived_from_the_url(base_url: str, expected: str) -> None:
    """The URL is the authoritative signal, not the stored provider name.

    An entry created before the provider field existed has no stored value, and a
    user may pick a preset and then edit the URL. The URL is what actually decides
    which endpoint is spoken to, so it is what the capability check must read.
    """
    assert provider_for_url(base_url) == expected


def test_gemini_omits_the_thinking_field() -> None:
    """The fix: selecting "disabled" must not send a field Google rejects.

    Omitting it means the provider's own default applies -- reasoning stays on,
    which costs more, but the activity is analysed and delivered instead of
    failing. A working analysis that costs more beats a lost one.
    """
    payload = build_payload(_config(GEMINI_URL, thinking="disabled"), "p", b"jpeg")
    assert "thinking" not in payload


@pytest.mark.parametrize("base_url", [DEEPSEEK_URL, GLM_URL, OPENAI_URL])
def test_a_provider_that_accepts_the_toggle_still_receives_it(base_url: str) -> None:
    """The cost control must keep working everywhere it is honoured.

    Measured on this project: reasoning dominated cost, spending 400-1900 tokens
    before any output. Dropping the field for providers that accept it would
    silently restore that cost.
    """
    payload = build_payload(_config(base_url, thinking="disabled"), "p", b"jpeg")
    assert payload["thinking"] == {"type": "disabled"}


def test_an_unnameable_provider_keeps_the_existing_behaviour() -> None:
    """A local router must not change behaviour because of this fix.

    Its capabilities are unknown, and this project's rule is that a new feature
    does not change what an existing deployment does. Sending the field is what
    every deployment does today.
    """
    payload = build_payload(_config(CUSTOM_URL, thinking="disabled"), "p", b"jpeg")
    assert payload["thinking"] == {"type": "disabled"}


def test_gemini_omits_thinking_even_when_enabled() -> None:
    """`enabled` is the same unknown field with a different value.

    Google rejects the key, not the value, so both spellings must be omitted --
    gating only "disabled" would leave the other half of the option broken.
    """
    payload = build_payload(_config(GEMINI_URL, thinking="enabled"), "p", b"jpeg")
    assert "thinking" not in payload


def test_gemini_still_receives_the_rest_of_the_request() -> None:
    """Omitting one field must not degrade into sending a broken body.

    The model, the messages and the token budget are what make the call work at
    all; dropping the reasoning toggle must not take anything else with it.
    """
    payload = build_payload(
        _config(GEMINI_URL, thinking="disabled", reasoning_effort="low"),
        "the prompt",
        b"jpeg",
    )
    assert payload["model"] == "some-model"
    assert payload["max_tokens"] == 4000
    assert payload["stream"] is False
    assert payload["messages"][0]["content"][0]["text"] == "the prompt"
    # Probed as accepted by the same endpoint, so it must survive the gate.
    assert payload["reasoning_effort"] == "low"


def test_the_gemini_preset_is_the_one_that_is_gated() -> None:
    """Guard the table against drifting away from the presets it names.

    A typo in the set would silently un-gate Gemini and reintroduce the 400 --
    the failure would only appear as activities that stop being analysed.
    """
    from custom_components.frigate_vision.const import PROVIDER_PRESETS

    assert "gemini" in PROVIDER_PRESETS
    assert THINKING_UNSUPPORTED_PROVIDERS <= set(PROVIDER_PRESETS)
    # And the preset whose URL this deployment uses is the gated one.
    assert provider_for_url(PROVIDER_PRESETS["gemini"]["base_url"]) == "gemini"
