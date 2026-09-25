"""Tests for the provider switch, driven through the real flow.

Reported from the UI, so these drive the options flow the way the frontend does --
tapping menu entries and submitting whole forms -- rather than calling helpers
directly. The reported symptom was that switching provider "did nothing", and the
cause lived in the round trips between steps, which only a real flow reproduces.
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


def _suggested_for(schema, field: str):
    for marker, _selector in schema.schema.items():
        if getattr(marker, "schema", None) == field:
            description = marker.description
            if isinstance(description, dict):
                return description.get("suggested_value")
            return description
    raise AssertionError(f"{field} not in schema")


def test_the_provider_inferred_from_a_url_is_used_by_the_menu() -> None:
    """`_provider_for_url` 仍在使用中（菜单靠它显示当前服务商），必须继续正确。

    它此前服务于表单里的下拉框；下拉框删掉后唯一的调用点是菜单标签。这里直接测
    这个函数本身，因为它现在没有第二个观察面——菜单那两条测试只覆盖了本部署的
    两个具体取值，推断本身的边界（每个预设、空地址）只能在这里守住。
    """
    for name, preset in config_flow.PROVIDER_PRESETS.items():
        assert config_flow._provider_for_url(preset["base_url"]) == name  # noqa: SLF001

    # 末尾斜杠两种写法都合理，不该改变判断结果。
    deepseek = config_flow.PROVIDER_PRESETS["deepseek"]["base_url"]
    assert config_flow._provider_for_url(f"{deepseek}/") == "deepseek"  # noqa: SLF001

    # 匹配不到任何预设就是「其他」：本部署的本地路由器正是这一类。
    assert config_flow._provider_for_url(LIVE["llm_base_url"]) == "custom"  # noqa: SLF001

    # 空地址回退到默认服务商，而不是「其他」——这是既有行为，不要改。
    assert config_flow._provider_for_url("") == "deepseek"  # noqa: SLF001


async def test_the_menu_shows_the_current_provider(hass: HomeAssistant) -> None:
    """菜单项要带上当前服务商，否则用户无法在不打开表单的情况下确认状态。

    `llm_provider` 不存储（真相在 URL 里），所以这个显示值是从 URL 推导的。
    本部署的地址是本地路由器，匹配不到任何预设，应显示「其他」——而且必须是有
    可读文字的回退值，不能把 'custom' 这个内部键名直接显示出来。
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert result["type"] is FlowResultType.MENU
    labels = result["menu_options"]
    assert isinstance(labels, dict), "dict 形式才能带自定义文字"
    assert "provider" in labels, "菜单里要有切换服务商这一项"
    item = labels["provider"]
    assert "其他" in item, f"应显示可读的「其他」，实际：{item!r}"
    assert "custom" not in item, f"内部键名不该显示给用户：{item!r}"


async def test_the_menu_shows_a_preset_name_when_the_url_matches(
    hass: HomeAssistant,
) -> None:
    """地址匹配某个预设时，菜单显示那个预设的名字而不是「其他」。"""
    entry = MockConfigEntry(
        domain=DOMAIN,
        title="Front Door",
        data={},
        options=dict(LIVE, llm_base_url="https://api.deepseek.com/v1"),
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    item = result["menu_options"]["provider"]
    assert "DeepSeek" in item, f"实际：{item!r}"
    assert "其他" not in item, f"不该回退到「其他」：{item!r}"


async def test_choosing_gemini_returns_the_form_with_the_url_filled_in(
    hass: HomeAssistant,
) -> None:
    """选 Gemini 后，配置表单里地址必须已经是 Gemini 的。

    这是本功能的核心：用户点一下就能看到地址变了，不必先提交再猜。
    菜单点击是真正的服务端往返，所以服务端有机会把 suggested_value 填好。
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider"}
    )
    assert result["type"] is FlowResultType.MENU, "应先显示服务商菜单"
    assert set(result["menu_options"]) == {
        "provider_deepseek",
        "provider_gemini",
        "provider_glm",
        "provider_openai",
        "provider_custom",
    }

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider_gemini"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "settings_form"
    assert _suggested_for(result["data_schema"], "llm_base_url") == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )


async def test_choosing_gemini_also_fills_in_the_model_name(
    hass: HomeAssistant,
) -> None:
    """模型名必须一起换。

    否则地址指向 Gemini 而模型名仍是 Deepseek 的，Gemini 不认那个名字 ——
    地址对了照样调不通，用户更难看出问题在哪。
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider_gemini"}
    )
    assert _suggested_for(result["data_schema"], "llm_model") == "gemini-3.8-flash"


async def test_choosing_custom_changes_neither_url_nor_model(
    hass: HomeAssistant,
) -> None:
    """「其他」不预填任何东西：地址保持当前值，模型不动。

    选它的用户本来就打算自己填；替他改地址只会破坏一个能用的端点。
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider_custom"}
    )
    assert result["type"] is FlowResultType.FORM
    assert _suggested_for(result["data_schema"], "llm_base_url") == LIVE["llm_base_url"]
    assert _suggested_for(result["data_schema"], "llm_model") == LIVE["llm_model"]


async def test_confirming_after_choosing_gemini_saves_both(
    hass: HomeAssistant,
) -> None:
    """选完 Gemini 再提交，地址与模型名都要存下来。"""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "provider_gemini"}
    )
    submitted = {
        **LIVE,
        "llm_base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "llm_model": "gemini-3.8-flash",
    }
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], submitted
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"]["llm_base_url"] == (
        "https://generativelanguage.googleapis.com/v1beta/openai"
    )
    assert result["data"]["llm_model"] == "gemini-3.8-flash"


async def test_the_settings_form_no_longer_has_a_provider_field(
    hass: HomeAssistant,
) -> None:
    """服务商不再是配置表单里的字段，只存在于菜单。

    它留在表单里正是「以为没改」的根源：表单显示的当前值是从 URL 推导的，而提交
    时判断「有没有改过」用的是存储里的原始值，两者不一致时会静默失效。
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )
    fields = {getattr(m, "schema", None) for m in result["data_schema"].schema}
    assert "llm_provider" not in fields
    assert "llm_base_url" in fields, "地址仍须可编辑"


async def test_the_settings_menu_groups_the_form_with_the_connection_test(
    hass: HomeAssistant,
) -> None:
    """「配置参数」应是子菜单，把表单与连通性测试放在一起。

    用户指出这两项在顶层并列时，测试看起来与配置无关。HA 的表单放不了第二个
    按钮（表单只有一个「提交」动作），所以用菜单层级表达归属。
    """
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    assert set(result["menu_options"]) == {"settings", "provider"}, (
        "顶层不应再有 test_connection —— 它属于配置参数"
    )

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    assert result["type"] is FlowResultType.MENU, "「配置参数」应是子菜单"
    assert set(result["menu_options"]) == {"settings_form", "test_connection"}

    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings_form"}
    )
    assert result["type"] is FlowResultType.FORM
    assert "llm_base_url" in {
        getattr(m, "schema", None) for m in result["data_schema"].schema
    }


async def test_the_connection_test_is_reachable_from_inside_the_settings_menu(
    hass: HomeAssistant,
) -> None:
    """连通性测试仍能到达（只是位置变了），别把它弄丢。"""
    entry = MockConfigEntry(
        domain=DOMAIN, title="Front Door", data={}, options=dict(LIVE)
    )
    entry.add_to_hass(hass)
    result = await hass.config_entries.options.async_init(entry.entry_id)
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "settings"}
    )
    result = await hass.config_entries.options.async_configure(
        result["flow_id"], {"next_step_id": "test_connection"}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "test_connection"
