"""The model dropdown's option source: parsing, filtering, merging, fetching.

Samples are real entries captured from
`GET https://generativelanguage.googleapis.com/v1beta/models` on 2026-10-06
(the full 50-entry response, minus the descriptions), because the whole point of
the filter is which *actual* models it keeps and drops. A hand-written fixture
would let the filter pass here and fail in production.

The layering is the thing under test. Measured, `lyria-3.5`,
`antigravity-preview-*` and `deep-research-*` all carry `generateContent`, so the
capability check cannot reject them and only the prefix allow-list can -- and
`gemini-embedding-*` is the reverse. Each layer therefore gets a test that fails
if that layer alone is removed.
"""

from __future__ import annotations

import logging

import pytest

from custom_components.frigate_vision.models_list import (
    async_fetch_model_ids,
    merge_model_options,
    model_ids_from_native,
    model_ids_from_openai,
    native_models_url,
)

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


def _entry(name: str, *methods: str, **extra: object) -> dict:
    """A native-list entry. Methods default to generateContent, as real ones are."""
    entry: dict = {"name": f"models/{name}"}
    if methods:
        entry["supportedGenerationMethods"] = list(methods)
    entry.update(extra)
    return entry


def _native(*entries: dict) -> dict:
    return {"models": list(entries)}


def _chat(name: str) -> dict:
    """A native entry for a plain chat model, which is the common case here."""
    return _entry(name, "generateContent", "countTokens")


def _openai(*ids: str) -> dict:
    return {"data": [{"id": f"models/{i}"} for i in ids]}


class TestNativeUrl:
    """The derived URL is a specific string, so it is pinned rather than described."""

    def test_googles_compatible_root_maps_to_its_native_list(self) -> None:
        assert native_models_url(GEMINI_URL) == (
            "https://generativelanguage.googleapis.com/v1beta/models"
        )

    def test_a_trailing_slash_is_ignored(self) -> None:
        assert native_models_url(GEMINI_URL + "/") == (
            "https://generativelanguage.googleapis.com/v1beta/models"
        )

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.deepseek.com/v1",
            "https://open.bigmodel.cn/api/paas/v4",
            "https://api.openai.com/v1",
            "http://192.168.166.50:7864/v1",
            "",
        ],
    )
    def test_a_url_that_is_not_googles_maps_to_nothing(self, base_url: str) -> None:
        """No guess: a wrong derived URL is a request that is certain to 404."""
        assert native_models_url(base_url) is None


class TestPrefixAllowList:
    """Layer 1 -- the only layer that can reject a capable-looking non-chat model."""

    @pytest.mark.parametrize(
        "name",
        [
            "lyria-3.5",
            "lyria-3-clip-preview",
            "lyria-3-pro-preview",
            "antigravity-preview-09-2026",
            "deep-research-pro-preview-12-2025",
            "nano-banana-pro-preview",
            "aqa",
            "veo-3.1-generate-preview",
        ],
    )
    def test_a_non_gemini_family_is_dropped_even_though_it_is_capable(
        self, name: str
    ) -> None:
        """These carry generateContent, so only the allow-list rejects them.

        Removing layer 1 makes exactly these names appear in the dropdown -- a
        music generator or a video generator offered as a vision model.
        """
        payload = _native(_entry(name, "generateContent", "countTokens"))
        assert model_ids_from_native(payload) == ()

    @pytest.mark.parametrize(
        "name", ["gemini-3.8-flash", "gemini-2.5-pro", "gemma-4-26b-a4b-it"]
    )
    def test_a_chat_family_is_kept(self, name: str) -> None:
        payload = _native(_entry(name, "generateContent", "countTokens"))
        assert model_ids_from_native(payload) == (name,)

    def test_gemma_is_kept_because_it_is_a_real_chat_model(self) -> None:
        """`gemma-4-*` supports function calling, which Assist-sized work needs.

        Source: home-assistant/core#184261, which measured it against the real
        API. Excluding it would drop a working (and cheaper) option.
        """
        payload = _native(_entry("gemma-4-31b-it", "generateContent", "countTokens"))
        assert model_ids_from_native(payload) == ("gemma-4-31b-it",)


class TestCapabilityLayer:
    """Layer 2 -- the only layer that rejects a well-named non-chat model."""

    def test_an_embedding_model_is_dropped_despite_its_name(self) -> None:
        """`gemini-embedding-2` passes the allow-list; the methods list rejects it.

        Removing layer 2 makes this appear, because its name contains no marker
        word that layer 3 looks for... except `embedding`, which it does -- so
        the two layers overlap here, and this test says which one is doing the
        work by using a name with no marker at all.
        """
        payload = _native(_entry("gemini-vector-2", "embedContent", "countTokens"))
        assert model_ids_from_native(payload) == ()

    def test_a_real_embedding_model_is_dropped(self) -> None:
        payload = _native(
            _entry("gemini-embedding-2", "embedContent", "countTextTokens")
        )
        assert model_ids_from_native(payload) == ()

    def test_an_absent_capability_field_does_not_reject(self) -> None:
        """Absent is not the same as incapable.

        Measured, all 50 entries carry the field, so this branch exists for a
        future response that renames it. Treating an unknown shape as a
        rejection would blank the dropdown -- strictly worse than listing a few
        names too many, since typing is always still allowed.
        """
        payload = _native({"name": "models/gemini-3.8-flash"})
        assert model_ids_from_native(payload) == ("gemini-3.8-flash",)

    def test_an_empty_methods_list_does_not_reject(self) -> None:
        payload = _native(_entry("gemini-3.8-flash"))
        assert model_ids_from_native(payload) == ("gemini-3.8-flash",)


class TestMarkerLayer:
    """Layer 3 -- rejects what the first two accept but the integration cannot use."""

    @pytest.mark.parametrize(
        "name",
        [
            "gemini-3.8-flash-tts",
            "gemini-3.8-flash-lite-tts",
            "gemini-3.1-flash-tts-preview",
            "gemini-3.1-flash-image",
            "gemini-2.5-flash-image",
            "gemini-3.5-transcribe",
            "gemini-robotics-er-2-preview",
            "gemini-2.5-computer-use-preview-10-2025",
            "gemini-omni-1.1-flash",
            # Present only in the COMPATIBLE endpoint's list, so these reach the
            # dropdown exactly when the native lookup failed. Measured: absent from
            # the native 18, present in the compatible 21.
            "gemini-2.5-flash-native-audio-latest",
            "gemini-2.5-flash-native-audio-preview-12-2025",
        ],
    )
    def test_a_marker_word_drops_the_entry(self, name: str) -> None:
        """All of these pass layers 1 and 2 with a plain generateContent list."""
        payload = _native(_entry(name, "generateContent", "countTokens"))
        assert model_ids_from_native(payload) == ()

    def test_the_two_endpoints_agree_on_the_outcome(self) -> None:
        """Both endpoints must yield the same set, because either may answer.

        Measured: they do NOT offer the same models -- the compatible list carries
        the Live-API `native-audio` family that the native list omits, and only the
        native list carries capability metadata. Which one answers depends on
        whether a single HTTP request succeeded, so any difference the filter does
        not remove becomes a dropdown whose contents change between two openings
        with no user action. This pins that the filter removes it.
        """
        wanted = ("gemini-3.8-flash", "gemini-3.5-flash")
        native = _native(*[_chat(n) for n in wanted], _chat("lyria-3.5"))
        compat = {
            "data": [
                {"id": f"models/{n}"}
                for n in (*wanted, "gemini-2.5-flash-native-audio-latest")
            ]
        }
        assert model_ids_from_native(native) == wanted
        assert model_ids_from_openai(compat, gemini=True) == wanted

    def test_omni_is_excluded_on_the_documented_grounds(self) -> None:
        """Omni generates video through a different API, not text through this one.

        The integration posts to chat/completions and parses the reply as JSON,
        so an omni model would fail as `invalid_provider_response` -- an error a
        long way from its cause. Google's Omni documentation specifies the
        Interactions API and a video output.
        """
        payload = _native(
            _entry("gemini-omni-1.1-flash", "generateContent", "countTokens")
        )
        assert model_ids_from_native(payload) == ()


class TestParsing:
    """Shape handling, so a malformed response degrades instead of raising."""

    def test_the_models_prefix_is_removed(self) -> None:
        """The chat endpoint takes bare ids; the list endpoint returns prefixed ones."""
        payload = _native(_entry("gemini-3.6-flash", "generateContent"))
        assert model_ids_from_native(payload) == ("gemini-3.6-flash",)

    def test_duplicates_keep_their_first_position(self) -> None:
        """Aliases appear twice in the real response; the dropdown shows one row."""
        payload = _native(
            _entry("gemini-3.8-flash", "generateContent"),
            _entry("gemini-2.5-flash", "generateContent"),
            _entry("gemini-3.8-flash", "generateContent"),
        )
        assert model_ids_from_native(payload) == (
            "gemini-3.8-flash",
            "gemini-2.5-flash",
        )

    def test_order_is_the_response_order_not_sorted(self) -> None:
        """Newest-first is more useful in a dropdown than alphabetical."""
        payload = _native(
            _entry("gemini-3.8-flash", "generateContent"),
            _entry("gemini-2.5-flash", "generateContent"),
        )
        assert model_ids_from_native(payload) == (
            "gemini-3.8-flash",
            "gemini-2.5-flash",
        )

    @pytest.mark.parametrize(
        "name",
        ["", "   ", None, 123, "x" * 129],
    )
    def test_an_unusable_name_is_skipped(self, name: object) -> None:
        """Bounded and type-checked: this value is remote input."""
        payload = {"models": [{"name": f"models/{name}"}]}
        assert model_ids_from_native(payload) == ()

    @pytest.mark.parametrize(
        "payload",
        [None, [], "text", 42, {}, {"models": "not-a-list"}, {"models": None}],
    )
    def test_a_malformed_body_yields_nothing(self, payload: object) -> None:
        assert model_ids_from_native(payload) == ()
        assert model_ids_from_openai(payload, gemini=True) == ()
        assert model_ids_from_openai(payload, gemini=False) == ()

    def test_a_non_mapping_entry_is_skipped(self) -> None:
        payload = {"models": ["gemini-3.8-flash", _chat("gemini-2.5-flash")]}
        assert model_ids_from_native(payload) == ("gemini-2.5-flash",)


class TestOpenAiShape:
    """The compatible endpoint's response, which the filter must handle identically."""

    def test_ids_are_read_from_data(self) -> None:
        payload = {
            "object": "list",
            "data": [
                {"id": "models/gemini-3.8-flash", "object": "model"},
                {"id": "models/gemini-2.5-flash", "object": "model"},
            ],
        }
        assert model_ids_from_openai(payload, gemini=True) == (
            "gemini-3.8-flash",
            "gemini-2.5-flash",
        )

    def test_the_same_layers_apply_without_a_capability_field(self) -> None:
        """This Gemini path gets no methods list, so only layers 1 and 3 can act.

        Stated as a test rather than a comment because the two Gemini paths are
        NOT equivalent: with no capability field, a hypothetical model excluded
        only by layer 2 would survive here.
        """
        payload = {
            "data": [
                {"id": "models/lyria-3.5"},  # layer 1
                {"id": "models/gemini-3.8-flash-tts"},  # layer 3
                {"id": "models/gemini-3.8-flash"},  # kept
            ]
        }
        assert model_ids_from_openai(payload, gemini=True) == ("gemini-3.8-flash",)

    def test_another_provider_is_not_filtered_by_geminis_rules(self) -> None:
        """The three layers describe Google's catalogue, not every catalogue.

        Measured against DeepSeek's `/models` (all `deepseek-*`): the allow-list
        would reject every entry and the dropdown would be empty for three of the
        four presets. Marker words misfire too -- OpenAI ships
        `gpt-4-vision-preview`, which contains `vision` and is a working chat
        model.
        """
        payload = {
            "data": [
                {"id": "deepseek-v4.1-flash"},
                {"id": "deepseek-v4-pro"},
                {"id": "some-embedding-model"},
            ]
        }
        assert model_ids_from_openai(payload, gemini=False) == (
            "deepseek-v4.1-flash",
            "deepseek-v4-pro",
            "some-embedding-model",
        )

    def test_another_provider_still_gets_shape_and_dedup(self) -> None:
        """Unfiltered is not unvalidated: unusable entries are still skipped."""
        payload = {"data": [{"id": "a"}, {"id": "a"}, {"id": ""}, {"nope": 1}, "x"]}
        assert model_ids_from_openai(payload, gemini=False) == ("a",)


class TestMerge:
    """The three sources, in priority order, with the stored value always kept."""

    def test_online_comes_first_then_preset_then_saved(self) -> None:
        merged = merge_model_options(
            ["gemini-3.8-flash"], ["gemini-3.8-flash", "fallback-model"], "old-model"
        )
        assert merged == ["gemini-3.8-flash", "fallback-model", "old-model"]

    def test_a_saved_value_no_list_mentions_is_kept(self) -> None:
        """The deployment's own case: stored `gemini-3.5-flash`, absent from the list.

        Dropping it would render the dropdown as "nothing selected", so the next
        submit would silently rewrite the model.
        """
        merged = merge_model_options(
            ["gemini-3.8-flash"], ["gemini-3.8-flash"], "gemini-3.5-flash"
        )
        assert merged[-1] == "gemini-3.5-flash"

    def test_an_empty_saved_value_adds_no_option(self) -> None:
        """An unconfigured entry must not get a blank row."""
        assert merge_model_options([], ["a"], "") == ["a"]

    def test_duplicates_across_sources_collapse(self) -> None:
        merged = merge_model_options(["a", "b"], ["b", "c"], "c")
        assert merged == ["a", "b", "c"]

    def test_everything_empty_is_an_empty_list(self) -> None:
        """The caller reads this as "render a text field"."""
        assert merge_model_options([], [], "") == []


# --- fetching ---------------------------------------------------------------


class _Response:
    def __init__(
        self, status: int, body: object = None, raw: str | None = None
    ) -> None:
        self.status = status
        self._body = body
        self._raw = raw

    async def json(self, **_kwargs: object) -> object:
        if self._raw is not None:
            import json as _json

            return _json.loads(self._raw)
        return self._body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _Session:
    """Record every GET and answer from a script keyed by URL substring."""

    def __init__(self, responses: list[tuple[str, _Response]]) -> None:
        self._responses = responses
        self.urls: list[str] = []
        self.headers: list[dict] = []

    def get(self, url: str, *, headers: dict, **_kwargs: object) -> _Response:
        self.urls.append(url)
        self.headers.append(headers)
        for fragment, response in self._responses:
            if fragment in url:
                return response
        raise AssertionError(f"unexpected URL: {url}")


class TestFetch:
    """`async_fetch_model_ids` must never raise, whatever the provider does."""

    async def test_the_native_endpoint_is_tried_first_and_wins(self) -> None:
        session = _Session(
            [
                ("/v1beta/models", _Response(200, _native(_chat("gemini-3.8-flash")))),
                ("/openai/models", _Response(200, _openai("other-model"))),
            ]
        )

        found = await async_fetch_model_ids(
            session,
            base_url=GEMINI_URL,
            api_key="k",  # type: ignore[arg-type]
        )
        assert found == ("gemini-3.8-flash",)
        assert len(session.urls) == 1, "the compatible endpoint must not be called too"

    async def test_the_native_key_travels_in_a_header_not_the_url(self) -> None:
        """A query string would reach logs, traces and diagnostic dumps."""
        session = _Session(
            [("/v1beta/models", _Response(200, _native(_chat("gemini-3.8-flash"))))]
        )

        await async_fetch_model_ids(
            session,
            base_url=GEMINI_URL,
            api_key="secret-key",  # type: ignore[arg-type]
        )
        assert "secret-key" not in session.urls[0]
        assert session.headers[0]["x-goog-api-key"] == "secret-key"

    async def test_a_failing_native_endpoint_falls_back_to_the_compatible_one(
        self,
    ) -> None:
        """Measured: the compatible endpoint answers 200 with 61 models.

        So this is a real fallback path, not decoration -- which is why the
        attempt order and the separate error handling both matter. A shared
        try/except around the two calls would make this unreachable.
        """
        session = _Session(
            [
                ("/v1beta/models", _Response(404)),
                ("/openai/models", _Response(200, _openai("gemini-3.5-flash"))),
            ]
        )

        found = await async_fetch_model_ids(
            session,
            base_url=GEMINI_URL,
            api_key="k",  # type: ignore[arg-type]
        )
        assert found == ("gemini-3.5-flash",)
        assert len(session.urls) == 2

    async def test_a_native_response_with_no_usable_models_also_falls_back(
        self,
    ) -> None:
        """An empty native answer is a failure, not a reason to show an empty list."""
        session = _Session(
            [
                ("/v1beta/models", _Response(200, {"models": []})),
                ("/openai/models", _Response(200, _openai("gemini-3.5-flash"))),
            ]
        )

        found = await async_fetch_model_ids(
            session,
            base_url=GEMINI_URL,
            api_key="k",  # type: ignore[arg-type]
        )
        assert found == ("gemini-3.5-flash",)

    @pytest.mark.parametrize("status", [401, 403, 404, 429, 500, 503])
    async def test_every_error_status_yields_an_empty_tuple(self, status: int) -> None:
        session = _Session(
            [
                ("/v1beta/models", _Response(status)),
                ("/openai/models", _Response(status)),
            ]
        )

        assert (
            await async_fetch_model_ids(
                session,
                base_url=GEMINI_URL,
                api_key="k",  # type: ignore[arg-type]
            )
            == ()
        )

    async def test_a_non_json_body_yields_an_empty_tuple(self) -> None:
        session = _Session(
            [
                ("/v1beta/models", _Response(200, raw="<html>nope</html>")),
                ("/openai/models", _Response(200, raw="<html>nope</html>")),
            ]
        )

        assert (
            await async_fetch_model_ids(
                session,
                base_url=GEMINI_URL,
                api_key="k",  # type: ignore[arg-type]
            )
            == ()
        )

    async def test_a_transport_error_yields_an_empty_tuple(self) -> None:
        class _Boom:
            def get(self, *_a: object, **_k: object) -> object:
                raise OSError("dns")

        assert (
            await async_fetch_model_ids(
                _Boom(),
                base_url=GEMINI_URL,
                api_key="k",  # type: ignore[arg-type]
            )
            == ()
        )

    async def test_no_key_means_no_request_at_all(self) -> None:
        """The first render of the setup form has no key typed yet.

        Issuing an unauthenticated request would waste a round trip and log a
        401 on every keystroke-to-submit cycle.
        """
        session = _Session([])

        assert (
            await async_fetch_model_ids(
                session,
                base_url=GEMINI_URL,
                api_key="",  # type: ignore[arg-type]
            )
            == ()
        )
        assert session.urls == []

    async def test_a_non_gemini_provider_skips_the_native_endpoint(self) -> None:
        session = _Session(
            [("/models", _Response(200, {"data": [{"id": "deepseek-v4.1-flash"}]}))]
        )

        found = await async_fetch_model_ids(
            session,
            base_url="https://api.deepseek.com/v1",
            api_key="k",  # type: ignore[arg-type]
        )
        assert found == ("deepseek-v4.1-flash",)
        assert session.urls[0] == "https://api.deepseek.com/v1/models"


class TestLogging:
    """The one log line must be useful and must not leak the credential."""

    async def test_the_key_never_appears_in_the_log(self, caplog) -> None:
        session = _Session(
            [("/v1beta/models", _Response(401)), ("/openai/models", _Response(401))]
        )

        with caplog.at_level(logging.DEBUG):
            await async_fetch_model_ids(
                session,
                base_url=GEMINI_URL,
                api_key="super-secret",  # type: ignore[arg-type]
            )
        assert "super-secret" not in caplog.text
        assert "?key=" not in caplog.text

    async def test_the_log_names_the_host_only(self, caplog) -> None:
        """The full base URL can carry a private path, so only the host is logged."""
        session = _Session([("/models", _Response(500)), ("/models", _Response(500))])

        with caplog.at_level(logging.DEBUG):
            await async_fetch_model_ids(
                session,
                base_url="https://gw.example.com/private/v1",
                api_key="k",  # type: ignore[arg-type]
            )
        assert "gw.example.com" in caplog.text
        assert "/private/" not in caplog.text
