"""Discover which model ids a provider offers, for the config form's dropdown.

The model field used to be free text, so a user had to know the provider's model
ids by heart. This module fetches them, best effort, and merges them with the
built-in list so the form can offer a dropdown that still allows typing.

**Nothing here may raise.** The only caller is a config form, and a form that
cannot render is worse than a form with fewer options: a failed lookup degrades
to the built-in list, which is exactly what the field offered before this module
existed. That is the same rule `_async_focus_zone_choices` follows for the zone
dropdown, and for the same reason.

Filtering is three layers, and the layering is load-bearing rather than
defensive. Measured against the real 50-entry response (2026-10-06):

* prefix allow-list -- the ONLY layer that rejects `lyria-3.5`,
  `antigravity-preview-*` and `deep-research-*`, because those carry
  `generateContent` just like a chat model does. A capability check cannot see
  them.
* `supportedGenerationMethods` -- rejects `gemini-embedding-*`, which the
  allow-list accepts (the name is fine, the purpose is not).
* marker words -- rejects `*-tts`, `*-image`, `gemini-3.5-transcribe`,
  `gemini-*-computer-use-*` and `gemini-robotics-*`, all of which pass the first
  two layers.

Each layer catches what the others cannot; dropping any one of them lets a
non-chat model into the dropdown. `tests/test_model_list.py` pins that per layer.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import aiohttp

from .const import GEMINI_PRESET, provider_for_url

_LOGGER = logging.getLogger(__name__)

# Short on purpose. This runs while a config form is being rendered, so a slow
# provider must not hold the form; a 5s miss costs nothing because the built-in
# list is a usable answer.
MODEL_LIST_TIMEOUT_SECONDS = 5.0

# Layer 1. A positive list rather than a negative one: measured, `lyria-*`,
# `antigravity-*`, `deep-research-*`, `nano-banana-*`, `aqa` and `veo-*` all pass
# a capability check, and enumerating every non-chat family by hand would need
# updating for each new one. `gemma-` is included because `gemma-4-26b-a4b-it` is
# a real chat model with function calling (home-assistant/core#184261).
_ALLOWED_NAME_PREFIXES = ("gemini-", "gemma-")

# Layer 3. Names that survive the allow-list and the capability check but are not
# chat models. `computer-use` and `robotics` were both found in the real
# response; `omni` is a video-generation family that is served through a
# different API entirely.
#
# `audio` was added after a live verification run, and it matters more than its
# size suggests: the native endpoint's list does not contain the Live-API
# `gemini-2.5-flash-native-audio-*` models AT ALL, but the compatible endpoint's
# does, and the compatible endpoint is what answers whenever the native one fails
# (observed once in six consecutive runs). So without this marker the dropdown
# silently changes contents -- and gains three Live-API models this integration
# cannot call -- depending on whether one HTTP request succeeded.
_EXCLUDED_NAME_MARKERS = (
    "tts",
    "image",
    "embedding",
    "live",
    "vision",
    "transcribe",
    "computer-use",
    "robotics",
    "omni",
    "audio",
)

_CAPABILITY_KEY = "supportedGenerationMethods"
_GENERATE_CONTENT = "generateContent"

# A model id is compared and displayed, never used to build a path or URL, but it
# is still remote input. Bounded so a hostile or broken response cannot put a
# paragraph into a dropdown row.
_MAX_MODEL_NAME_LENGTH = 128

_ENDPOINT_TIMEOUT = aiohttp.ClientTimeout(total=MODEL_LIST_TIMEOUT_SECONDS)


def native_models_url(base_url: str) -> str | None:
    """The Gemini native models URL implied by a base URL, or None.

    Google's OpenAI-compatible surface lives at `.../v1beta/openai`, and the
    native model list sits one level up at `.../v1beta/models`. The native
    endpoint is worth trying first because it carries `supportedGenerationMethods`
    (layer 2), which the compatible one does not -- measured, the compatible
    response has only `id`/`object`/`owned_by`/`display_name`.

    Only a base URL that ends in `/openai` produces a URL: for any other address
    the derived path would be a guess, and a guess here means a request that is
    certain to 404.
    """
    text = base_url.strip().rstrip("/")
    if not text.endswith("/openai"):
        return None
    return f"{text[: -len('/openai')]}/models"


def _is_chat_model(name: str, entry: Mapping[str, Any]) -> bool:
    """Whether one model entry is something this integration can send an image to.

    Applied identically to both endpoints' responses, so the two cannot drift
    into accepting different sets of names.
    """
    lowered = name.lower()
    if not lowered.startswith(_ALLOWED_NAME_PREFIXES):
        return False
    methods = entry.get(_CAPABILITY_KEY)
    # `None` means the field is absent, which is NOT the same as "not capable":
    # a future response could rename the field, and treating an unknown shape as
    # a rejection would empty the dropdown. Absent therefore falls through to
    # the marker check. Measured, all 50 entries do carry the field, so this
    # branch is protection against a schema change rather than the normal path.
    if isinstance(methods, Sequence) and not isinstance(methods, str):
        if methods and _GENERATE_CONTENT not in methods:
            return False
    return not any(marker in lowered for marker in _EXCLUDED_NAME_MARKERS)


def _clean_name(raw: Any) -> str | None:
    """One model id, normalised, or None when it is unusable."""
    if not isinstance(raw, str):
        return None
    # Both endpoints prefix with `models/` (measured); the chat-completions
    # endpoint this integration calls does not, and the stored value must match
    # what that endpoint expects.
    name = raw.removeprefix("models/").strip()
    if not name or len(name) > _MAX_MODEL_NAME_LENGTH:
        return None
    return name


def model_ids_from_native(payload: Any) -> tuple[str, ...]:
    """Chat-capable model ids from a Gemini `/v1beta/models` response.

    Always filtered: this endpoint only exists for Gemini, so the three layers
    are the right ones by construction.
    """
    if not isinstance(payload, Mapping):
        return ()
    entries = payload.get("models")
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        return ()
    return _collect(entries, "name", gemini=True)


def model_ids_from_openai(payload: Any, *, gemini: bool) -> tuple[str, ...]:
    """Chat-capable model ids from an OpenAI-style `/models` response.

    `gemini` selects the filter, and it is required rather than defaulted so a
    caller has to state which catalogue it is reading. The two answers are not
    interchangeable:

    * **Gemini** -- all three layers apply. The compatible endpoint returns no
      capability field, so this path gets the allow-list and the markers only.
      That is a weaker filter than the native path and not equivalent to it.
    * **Anything else** -- no filtering at all. The three layers are calibrated
      against Google's catalogue, and applying them to another provider's names
      rejects working models: OpenAI's `gpt-4-vision-preview` carries a marker
      word, and the allow-list would reject the entire list of every non-Google
      provider (measured against DeepSeek's, which is all `deepseek-*`). A
      dropdown that is empty for three of the four presets is worse than one
      that lists a model the user then types over.
    """
    if not isinstance(payload, Mapping):
        return ()
    entries = payload.get("data")
    if not isinstance(entries, Sequence) or isinstance(entries, str):
        return ()
    return _collect(entries, "id", gemini=gemini)


def _collect(entries: Sequence[Any], key: str, *, gemini: bool) -> tuple[str, ...]:
    """Names that pass the filter, de-duplicated, in the response's own order.

    Order is preserved rather than sorted: the provider lists roughly
    newest-first, which is more useful in a dropdown than alphabetical. Without
    `gemini`, only shape and uniqueness are enforced.
    """
    found: list[str] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        name = _clean_name(entry.get(key))
        if name is None or name in found:
            continue
        if not gemini or _is_chat_model(name, entry):
            found.append(name)
    return tuple(found)


def merge_model_options(
    online: Sequence[str],
    preset: Sequence[str],
    saved: str,
) -> list[str]:
    """The dropdown's options: online, then built-in, then whatever is stored.

    The stored value is kept even when no list mentions it, which is the same
    union rule `_focus_zone_options` applies: Home Assistant renders a dropdown
    whose current value is absent as "nothing selected", so dropping it would
    silently rewrite the user's configuration as a side effect of opening a form.
    """
    options: list[str] = []
    for candidate in (*online, *preset):
        if isinstance(candidate, str) and candidate and candidate not in options:
            options.append(candidate)
    if isinstance(saved, str) and saved and saved not in options:
        options.append(saved)
    return options


def _host_of(url: str) -> str:
    """Just the host of a URL, for logging.

    Never the full address: the user's base URL may carry a path or a query
    string that they consider private, and a native-API key would be in a header
    rather than the URL only by our own choice -- logging the whole URL invites a
    future edit that puts it back.
    """
    text = url.split("://", 1)[-1]
    return text.split("/", 1)[0] or url


async def async_fetch_model_ids(
    session: aiohttp.ClientSession,
    *,
    base_url: str,
    api_key: str,
    request_timeout: float = MODEL_LIST_TIMEOUT_SECONDS,
) -> tuple[str, ...]:
    """Model ids from the provider, or `()` when anything at all goes wrong.

    Returning an empty tuple rather than raising is the module's whole contract:
    every caller treats it as "we could not ask", and falls back to the built-in
    list. There is deliberately no error taxonomy here -- a caller cannot act on
    the difference between a timeout, a 401 and a malformed body, and none of
    them should reach the user as a form error (the credential and the endpoint
    are validated by the analysis path, against the API that actually matters).
    """
    if not api_key or not base_url:
        return ()
    client_timeout = aiohttp.ClientTimeout(total=request_timeout)
    is_gemini = provider_for_url(base_url) == GEMINI_PRESET

    # The native attempt gets its OWN try/except rather than sharing one with the
    # compatible attempt. Sharing would make the fallback unreachable: a failure
    # in the first call would jump straight to the outer handler and return ()
    # without ever trying the second. Measured, the compatible endpoint answers
    # 200 with 61 models, so that fallback is a real path and must be reachable.
    native = native_models_url(base_url) if is_gemini else None
    if native is not None:
        try:
            payload = await _get_json(
                session, native, {"x-goog-api-key": api_key}, client_timeout
            )
        except Exception as exc:  # noqa: BLE001 - the next attempt is the fallback
            _LOGGER.debug(
                "Native model list failed for %s (%s); trying the compatible one",
                _host_of(base_url),
                type(exc).__name__,
            )
        else:
            found = model_ids_from_native(payload)
            if found:
                return found

    try:
        payload = await _get_json(
            session,
            f"{base_url.rstrip('/')}/models",
            {"Authorization": f"Bearer {api_key}"},
            client_timeout,
        )
        return model_ids_from_openai(payload, gemini=is_gemini)
    except Exception as exc:  # noqa: BLE001 - a form must render whatever happens
        _LOGGER.debug(
            "Model list unavailable for %s (%s); using the built-in list",
            _host_of(base_url),
            type(exc).__name__,
        )
        return ()


async def _get_json(
    session: aiohttp.ClientSession,
    url: str,
    headers: Mapping[str, str],
    client_timeout: aiohttp.ClientTimeout,
) -> Any:
    """GET a URL and decode its JSON body, or raise.

    The raise is caught by the caller, which is what turns every failure mode --
    transport, status, decode -- into the same benign empty result.
    """
    async with session.get(
        url, headers=dict(headers), timeout=client_timeout
    ) as response:
        if response.status != 200:
            raise RuntimeError(f"model list status {response.status}")
        return await response.json(content_type=None)
