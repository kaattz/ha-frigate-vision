"""Minimal `resource` shim so HA's test suite can import on Windows.

`homeassistant.util.resource` uses getrlimit/setrlimit only to raise the open
file-descriptor soft limit at startup. Reporting a high soft limit makes that
function a no-op, which is the correct behaviour on Windows. Local test aid
only; never deployed (lives outside custom_components/).
"""

from __future__ import annotations

RLIMIT_NOFILE = 7
RLIM_INFINITY = -1

# Report an effectively unlimited soft limit so set_open_file_descriptor_limit
# takes its "already >= desired" early return.
_SOFT = 1_048_576
_HARD = 1_048_576


def getrlimit(resource: int) -> tuple[int, int]:
    if resource != RLIMIT_NOFILE:
        return (_HARD, _HARD)
    return (_SOFT, _HARD)


def setrlimit(resource: int, limits: tuple[int, int]) -> None:
    """No-op: Windows has no per-process descriptor limit."""


__all__ = [
    "RLIM_INFINITY",
    "RLIMIT_NOFILE",
    "getrlimit",
    "setrlimit",
]
