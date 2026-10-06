"""Which optional body fields each provider actually accepts.

`thinking: {"type": "disabled"}` is a hard HTTP 400 on Google's
OpenAI-compatible endpoint:

    Invalid JSON payload received. Unknown name "thinking": Cannot find field.

That endpoint is the `gemini` preset, and the UI offers "disabled" as a
Reasoning mode for every provider -- so choosing it on this deployment made
*every* analysis fail with `provider_http_400`. A 400 is not retried (correctly:
the request itself is malformed), which made the option a trap rather than a
cost control.

`reasoning_effort` is a *different* question, and the first fix got it half
right. The key is accepted by that endpoint -- probed, and it answered 429 on
quota, which is checked after validation, where `thinking` was rejected before
quota was consulted. But acceptance of the key says nothing about its values,
and the original probe stopped at the key. Probed properly on 2026-10-06, one
text-only request per value, Google answers `max` with:

    Invalid reasoning_effort: max. Valid values are: high, low, medium, minimal, none

and `minimal` with "Thinking level MINIMAL is not supported for this model" on
`gemini-3.8-flash` (though not on 3.5). So `max` was the same trap one field
over: offered by the UI, rejected by the provider, unbillable and unfailoverable
past. This module now pins both the option list and the translation.
"""

from __future__ import annotations

import pytest

from custom_components.frigate_vision.const import (
    REASONING_EFFORT_TRANSLATIONS,
    THINKING_UNSUPPORTED_PROVIDERS,
    provider_for_url,
    reasoning_effort_for,
)
from custom_components.frigate_vision.vision import (
    REASONING_EFFORTS,
    VisionConfig,
    async_test_connection,
    build_payload,
)

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


# --- reasoning_effort values -------------------------------------------------


def test_the_option_list_drops_max_and_offers_medium() -> None:
    """The list must match what the endpoint accepts, not what sounds reasonable.

    Measured: `max` is a hard 400 and `medium` is a 200. Shipping `max` is
    exactly the defect the module docstring describes; omitting `medium` removed
    a working cost control. Both halves are pinned because either one alone
    would look fine in review.
    """
    assert "max" not in REASONING_EFFORTS
    assert "medium" in REASONING_EFFORTS


def test_minimal_is_not_offered() -> None:
    """`minimal` is a coin flip by model, so offering it would be a trap.

    Probed: 400 on `gemini-3.8-flash` ("Thinking level MINIMAL is not supported
    for this model"), 200 on `gemini-3.5-flash`. A UI option whose validity
    depends on a model name the user can change independently is a failure
    waiting for the next model release.
    """
    assert "minimal" not in REASONING_EFFORTS
    assert "minimal" not in REASONING_EFFORT_TRANSLATIONS.get("gemini", {})


def test_a_stored_max_is_translated_not_dropped() -> None:
    """The option list is not the only way a value reaches the request.

    An entry that stored `max` before this fix keeps it until someone reopens
    the form. Sending it verbatim would 400 every activity on a deployment whose
    owner has no reason to suspect the setting.
    """
    payload = build_payload(_config(GEMINI_URL, reasoning_effort="max"), "p", b"jpeg")
    assert payload["reasoning_effort"] == "high"


def test_an_accepted_value_is_sent_unchanged() -> None:
    """Translation must not rewrite values the provider already accepts."""
    for value in ("low", "medium", "high"):
        payload = build_payload(
            _config(GEMINI_URL, reasoning_effort=value), "p", b"jpeg"
        )
        assert payload["reasoning_effort"] == value, value


def test_default_still_sends_no_field() -> None:
    """`default` means "let the provider decide", which is the absence of a field."""
    payload = build_payload(
        _config(GEMINI_URL, reasoning_effort="default"), "p", b"jpeg"
    )
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize("base_url", [DEEPSEEK_URL, GLM_URL, OPENAI_URL])
def test_another_provider_keeps_max(base_url: str) -> None:
    """Only the measured provider is translated.

    `max` was never probed against these, and this project's rule is that a fix
    does not change behaviour for a provider it did not measure. Rewriting it
    everywhere would silently cap a knob that may work.
    """
    payload = build_payload(_config(base_url, reasoning_effort="max"), "p", b"jpeg")
    assert payload["reasoning_effort"] == "max"


def test_a_local_router_keeps_max() -> None:
    """An unnameable endpoint's capabilities are unknown, so nothing is rewritten."""
    assert reasoning_effort_for(CUSTOM_URL, "max") == "max"


@pytest.mark.parametrize(
    ("base_url", "value", "expected"),
    [
        (GEMINI_URL, "max", "high"),
        (GEMINI_URL, "MAX", "MAX"),
        (GEMINI_URL, "auto", "auto"),
        (GEMINI_URL, "", ""),
        (GEMINI_URL, "default", ""),
        (DEEPSEEK_URL, "default", ""),
    ],
)
def test_the_translation_helper_is_the_single_decision_point(
    base_url: str, value: str, expected: str
) -> None:
    """The probe and the analysis must answer identically.

    A probe that sent a rejected value would report HTTP 400 for a configuration
    that works perfectly, sending the user to inspect their key and model name.
    Both call this helper, so the mapping is checked here once.

    Case is NOT folded: the endpoint's own message lists lowercase values only,
    and inventing a case-insensitive match would hide a future `HIGH` that the
    provider might start accepting.
    """
    assert reasoning_effort_for(base_url, value) == expected


async def test_the_connection_probe_uses_the_translation() -> None:
    """The probe is the other caller, and it must not send a rejected value."""
    sent: list[dict] = []

    class _Response:
        status = 200

        async def text(self) -> str:
            return '{"choices":[{"message":{"content":"OK"}}]}'

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    class _Session:
        def post(self, url, *, json, headers, timeout):
            sent.append(json)
            return _Response()

    await async_test_connection(
        _Session(),  # type: ignore[arg-type]
        _config(GEMINI_URL, reasoning_effort="max"),
    )
    assert sent, "the probe sent no request"
    assert sent[0]["reasoning_effort"] == "high"


def test_the_translation_table_cannot_name_an_unknown_provider() -> None:
    """A typo here would silently stop translating, which is the whole defect."""
    from custom_components.frigate_vision.const import PROVIDER_PRESETS

    assert set(REASONING_EFFORT_TRANSLATIONS) <= set(PROVIDER_PRESETS)
