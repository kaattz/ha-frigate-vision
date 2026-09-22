"""Opt-in live harness for the vision client.

Drives the shipped client against the real provider over this deployment's own
contact sheets. Not named `test_*.py`, so the default run skips it: it needs
network access, a credential and a running Home Assistant.

Run explicitly:

    pytest tests/e2e_llm_live.py -q -s

Every deployment coordinate -- host, credential and config entry -- comes from
the environment, so nothing about a real installation is hardcoded here. The
harness skips itself when they are absent.

    FRIGATE_VISION_SSH_HOST       host running Home Assistant (required)
    FRIGATE_VISION_SSH_PASSWORD   SSH password for that host (required)
    FRIGATE_VISION_ENTRY_ID       config entry to read provider settings from
    FRIGATE_VISION_SSH_PORT       SSH port (default 22)
    FRIGATE_VISION_SSH_USER       SSH user (default root)
    FRIGATE_VISION_PLINK          path to plink.exe (default: standard install)
    FRIGATE_VISION_MEDIA         media directory holding the sheets
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import aiohttp
import pytest

from custom_components.frigate_vision.vision import (
    MODE_CLASSIFICATIONS,
    VisionConfig,
    VisionError,
    async_analyze,
)

SSH_HOST = os.environ.get("FRIGATE_VISION_SSH_HOST", "")
SSH_PASSWORD = os.environ.get("FRIGATE_VISION_SSH_PASSWORD", "")
ENTRY_ID = os.environ.get("FRIGATE_VISION_ENTRY_ID", "")
SSH_PORT = os.environ.get("FRIGATE_VISION_SSH_PORT", "22")
SSH_USER = os.environ.get("FRIGATE_VISION_SSH_USER", "root")
PLINK_EXE = os.environ.get(
    "FRIGATE_VISION_PLINK", r"C:\Program Files\PuTTY\plink.exe"
)
MEDIA = os.environ.get("FRIGATE_VISION_MEDIA", "/media/frigate_vision/verify")

pytestmark = pytest.mark.skipif(
    not (SSH_HOST and SSH_PASSWORD and ENTRY_ID),
    reason=(
        "live harness needs FRIGATE_VISION_SSH_HOST, "
        "FRIGATE_VISION_SSH_PASSWORD and FRIGATE_VISION_ENTRY_ID"
    ),
)

# Real sheets captured from this deployment. Rename or replace these with your
# own before running: the names are only labels, the keys are epoch seconds.
SHEETS = [
    ("05:45-was-failed", "1789854302"),
    ("08:27-cleaner", "1789864065"),
    ("08:58", "1789865912"),
    ("10:24", "1789871048"),
]


def _plink() -> list[str]:
    """Build the SSH command from the environment, never from literals."""
    return [
        PLINK_EXE,
        "-batch",
        "-ssh",
        "-P",
        SSH_PORT,
        "-l",
        SSH_USER,
        "-pw",
        SSH_PASSWORD,
        SSH_HOST,
    ]


def _remote(program: str) -> str:
    return subprocess.run(
        _plink() + ["sudo python3 -"],
        input=program.encode(),
        capture_output=True,
        timeout=240,
    ).stdout.decode(errors="replace")


def _read_entry() -> dict:
    out = _remote(
        "import json\n"
        "d = json.load(open('/config/.storage/core.config_entries'))\n"
        f"e = [x for x in d['data']['entries'] if x['entry_id'] == '{ENTRY_ID}'][0]\n"
        "print(json.dumps({'data': e.get('data') or {}, "
        "'options': e.get('options') or {}}))\n"
    )
    return json.loads(out[out.index("{") :])


@pytest.mark.parametrize(
    ("thinking", "max_tokens"),
    [("disabled", 4000), ("default", 4000)],
)
async def test_live_analysis_over_real_sheets(
    tmp_path: Path, thinking: str, max_tokens: int
) -> None:
    entry = _read_entry()
    options = entry["options"]
    if not options.get("llm_api_key"):
        pytest.skip("entry has no api key configured")

    config = VisionConfig(
        base_url=str(options["llm_base_url"]),
        api_key=str(options["llm_api_key"]),
        model=str(options["llm_model"]),
        thinking=thinking,
        max_tokens=max_tokens,
        target_width=int(options.get("target_width", 768)),
        language=str(options.get("output_language", "zh-CN")),
    )
    print(f"\nmodel={config.model} thinking={thinking} max_tokens={max_tokens}")

    outcomes = []
    async with aiohttp.ClientSession() as session:
        for name, key in SHEETS:
            local = tmp_path / f"{key}.png"
            local.write_bytes(
                subprocess.run(
                    _plink() + [f"sudo cat {MEDIA}/{key}.png"],
                    capture_output=True,
                    timeout=300,
                ).stdout
            )
            try:
                classification, description, confidence, _version = await async_analyze(
                    session,
                    config,
                    evidence_path=str(local),
                    evidence_mode="review_six",
                    allowed=MODE_CLASSIFICATIONS["review_six"],
                )
            except VisionError as exc:
                print(f"  {name:<20} FAILED {exc}")
                outcomes.append((name, f"FAILED {exc}", None))
                continue
            print(
                f"  {name:<20} {classification:<20} conf={confidence:<4} "
                f"{description[:60]}"
            )
            outcomes.append((name, classification, confidence))

    usable = [o for _n, o, _c in outcomes if not o.startswith("FAILED")]
    print(f"\n  classified {len(usable)}/{len(outcomes)} with thinking={thinking}")
    assert usable, outcomes
