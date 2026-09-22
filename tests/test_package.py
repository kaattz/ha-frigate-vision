from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

COMPONENT = ROOT / "custom_components/frigate_vision"


def _repair_issue_codes() -> set[str]:
    """Read ISSUES without importing homeassistant (unavailable on Windows)."""
    tree = ast.parse((COMPONENT / "repairs.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "ISSUES":
                    return {
                        element.value
                        for element in node.value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, str)
                    }
    raise AssertionError("ISSUES not found in repairs.py")


def test_integration_package_exists() -> None:
    assert (ROOT / "custom_components/frigate_vision/__init__.py").is_file()


def test_manifest_and_strings_expose_supported_contract() -> None:
    manifest = json.loads(
        (ROOT / "custom_components/frigate_vision/manifest.json").read_text()
    )
    strings = json.loads(
        (ROOT / "custom_components/frigate_vision/strings.json").read_text()
    )
    assert manifest["integration_type"] == "service"
    assert manifest["dependencies"] == ["mqtt", "media_source"]
    # The vision request is built in-process: the reasoning toggle that
    # dominates cost is not reachable through another component's service.
    assert "after_dependencies" not in manifest
    assert set(strings["config"]["step"]) == {
        "user",
        "door",
        "llmvision",
        "options",
        "reconfigure",
        "reconfigure_door",
    }
    assert {"cannot_connect", "invalid_zones", "llm_missing_key"} <= set(
        strings["config"]["error"]
    )


def test_every_repair_issue_code_is_translated() -> None:
    """A Repair without a translation renders as a bare `domain: key` card.

    Home Assistant falls back to `frigate_vision: <code>` when the
    issue_registry key has no entry under `issues`, which is unreadable in the
    Repairs panel. Adding a code to ISSUES without both translations is the
    regression this guards.
    """
    codes = _repair_issue_codes()
    assert codes, "expected at least one repair issue code"

    for name in ("strings.json", "translations/zh-Hans.json"):
        payload = json.loads((COMPONENT / name).read_text(encoding="utf-8"))
        translated = set(payload.get("issues", {}))
        missing = sorted(codes - translated)
        assert not missing, f"{name} is missing repair issues: {missing}"
        assert not sorted(translated - codes), (
            f"{name} declares issues absent from repairs.ISSUES: "
            f"{sorted(translated - codes)}"
        )
        for code in sorted(codes):
            entry = payload["issues"][code]
            assert entry.get("title"), f"{name}: {code} has no title"
            assert entry.get("description"), f"{name}: {code} has no description"
