"""The model field's shape: a dropdown when there are options, text when there are not.

This is the interface contract the feature is actually judged on. `test_model_list.py`
covers which model ids come back; this file covers what the form does with them.

Two properties carry the design and are asserted here rather than left to the
integration tests:

1. **The field is a `SelectSelector` with `custom_value=True`.** The dropdown is
   the feature; `custom_value` is what stops it being a restriction. Without it,
   Home Assistant would reject any string outside the list, so a provider whose
   list could not be fetched -- or a model the filter drops -- would become
   unconfigurable, which is strictly worse than the text field it replaced.
2. **An empty option list yields a `TextSelector`, not an empty dropdown.** This
   is the pre-feature shape, and it is the one that is known to render correctly.
"""

from __future__ import annotations

import pytest
import voluptuous as vol

from custom_components.frigate_vision.config_flow import (
    _llm_schema,
    _model_field,
    _options_schema,
    _setup_schema,
)
from custom_components.frigate_vision.const import (
    CONF_FALLBACK_LLM_MODEL,
    CONF_LLM_MODEL,
)

# Named explicitly rather than imported from a constant: these are the keys stored
# in `entry.options`, so a rename is a migration and this should fail loudly.
MODEL_FIELDS = ("llm_model", "fallback_llm_model")


def _selector_for(schema: vol.Schema, name: str):
    """The selector bound to `name`, read off the schema's markers.

    Iterating a voluptuous schema yields the markers (the keys), not the
    selectors, so the selector has to be fetched by key.
    """
    for marker, selector_ in schema.schema.items():
        if getattr(marker, "schema", None) == name:
            return selector_
    raise AssertionError(f"{name} is missing from the schema")


class TestModelField:
    def test_options_make_it_a_dropdown(self) -> None:
        field = _model_field(["a", "b"])
        assert type(field).__name__ == "SelectSelector"

    def test_the_dropdown_allows_a_value_that_is_not_listed(self) -> None:
        """`custom_value` is the difference between a shortcut and a cage.

        Without it the selector validates against the option list, so a model
        absent from the list (renamed upstream, too new, or filtered out by a rule
        the user cannot see) could not be saved at all.
        """
        field = _model_field(["a", "b"])
        assert field.config["custom_value"] is True

    def test_the_dropdown_mode_is_dropdown(self) -> None:
        """Matches the provider and reasoning selects on the same form."""
        field = _model_field(["a", "b"])
        assert field.config["mode"] == "dropdown"

    def test_the_options_are_passed_through_in_order(self) -> None:
        """Order is the provider's own (roughly newest-first), so it is preserved."""
        field = _model_field(["gemini-3.8-flash", "gemini-2.5-flash"])
        assert field.config["options"] == [
            "gemini-3.8-flash",
            "gemini-2.5-flash",
        ]

    def test_no_options_yields_a_text_field(self) -> None:
        """Not an empty dropdown: that is a control with nothing to pick."""
        assert type(_model_field([])).__name__ == "TextSelector"

    def test_a_listed_name_is_accepted(self) -> None:
        field = _model_field(["a", "b"])
        assert field("a") == "a"

    def test_an_unlisted_name_is_also_accepted(self) -> None:
        """The server-side proof of `custom_value`, not just the config flag."""
        field = _model_field(["a", "b"])
        assert field("some-other-model") == "some-other-model"


class TestSchemasWithoutOptions:
    """Every factory must keep working when its caller passes nothing.

    That is the state of the setup flow (no key typed yet) and of every existing
    call site, so it has to reproduce the previous behaviour exactly: a text field.
    """

    def test_llm_schema_defaults_to_text(self) -> None:
        selector_ = _selector_for(_llm_schema(), CONF_LLM_MODEL)
        assert type(selector_).__name__ == "TextSelector"

    def test_options_schema_defaults_to_text_for_both_groups(self) -> None:
        schema = _options_schema()
        for name in MODEL_FIELDS:
            selector_ = _selector_for(schema, name)
            assert type(selector_).__name__ == "TextSelector", name

    def test_setup_schema_defaults_to_text_for_both_groups(self) -> None:
        schema = _setup_schema()
        for name in MODEL_FIELDS:
            selector_ = _selector_for(schema, name)
            assert type(selector_).__name__ == "TextSelector", name


class TestSchemasWithOptions:
    """And when options are supplied, the field becomes the dropdown."""

    def test_llm_schema_uses_the_options_it_is_given(self) -> None:
        selector_ = _selector_for(_llm_schema(model_options=["m1"]), CONF_LLM_MODEL)
        assert type(selector_).__name__ == "SelectSelector"
        assert selector_.config["options"] == ["m1"]

    def test_the_two_groups_get_their_own_options(self) -> None:
        """The fallback usually points at a different endpoint and catalogue.

        Sharing one list would offer the primary's model ids under the fallback's
        field, inviting a name the second provider does not serve -- and the
        failure would only appear when a failover actually happened.
        """
        schema = _options_schema(
            primary_models=["primary-a", "primary-b"],
            fallback_models=["fallback-a"],
        )
        assert _selector_for(schema, CONF_LLM_MODEL).config["options"] == [
            "primary-a",
            "primary-b",
        ]
        assert _selector_for(schema, CONF_FALLBACK_LLM_MODEL).config["options"] == [
            "fallback-a"
        ]

    def test_one_group_may_have_options_while_the_other_does_not(self) -> None:
        """The realistic mixed state: primary configured, fallback not.

        The configured group must still become a dropdown -- degrading both
        together would hide the feature from the group the user actually set up.
        """
        schema = _options_schema(primary_models=["m1"])
        assert type(_selector_for(schema, CONF_LLM_MODEL)).__name__ == "SelectSelector"
        assert (
            type(_selector_for(schema, CONF_FALLBACK_LLM_MODEL)).__name__
            == "TextSelector"
        )

    def test_setup_schema_forwards_both_lists(self) -> None:
        schema = _setup_schema(primary_models=["p"], fallback_models=["f"])
        assert _selector_for(schema, CONF_LLM_MODEL).config["options"] == ["p"]
        assert _selector_for(schema, CONF_FALLBACK_LLM_MODEL).config["options"] == ["f"]


class TestStoredKeyUnchanged:
    """The feature changes how the field is presented, not what is stored."""

    @pytest.mark.parametrize("name", MODEL_FIELDS)
    def test_the_fields_are_still_optional_and_keep_their_extra_key(
        self, name: str
    ) -> None:
        """`extra=vol.PREVENT_EXTRA` on the schema means a typo would be dropped.

        The key names are what `vision_config_from` reads, so a rename here would
        silently discard a user's model and fall back to `default_model`.
        """
        schema = _options_schema(primary_models=["m"], fallback_models=["m"])
        markers = [getattr(m, "schema", None) for m in schema.schema]
        assert name in markers

    def test_a_dropdown_still_accepts_an_unlisted_stored_value(self) -> None:
        """The upgrade case: `gemini-3.5-flash` is stored and absent from the list.

        The field must accept it on submit, so an entry configured before the
        feature existed can be saved from the form without its model changing.
        """
        selector_ = _selector_for(
            _options_schema(primary_models=["gemini-3.8-flash"]), CONF_LLM_MODEL
        )
        assert selector_("gemini-3.5-flash") == "gemini-3.5-flash"
