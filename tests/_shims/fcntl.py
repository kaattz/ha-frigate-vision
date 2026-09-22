"""Minimal fcntl shim so Home Assistant's test suite can import on Windows.

`homeassistant.runner` is the only module that imports fcntl, and it uses it
solely to hold an exclusive lock on `.HA_RUNNING` at process start. Tests never
reach that path, so no-op flock/LOCK_* constants are sufficient to let
`homeassistant` import. This file is a local-only test aid and is never
deployed: it lives outside custom_components/.
"""

from __future__ import annotations

import errno

LOCK_EX = 2
LOCK_NB = 4
LOCK_SH = 1
LOCK_UN = 8

F_GETFD = 1
F_SETFD = 2
FD_CLOEXEC = 1


def flock(fd: int, operation: int) -> None:
    """No-op: Windows has no flock, and no test depends on the lock."""


def lockf(fd: int, cmd: int, length: int = 0, start: int = 0, whence: int = 0) -> None:
    """No-op equivalent of fcntl.lockf."""


def fcntl(fd: int, cmd: int, arg: int = 0) -> int:
    return 0


def ioctl(fd: int, request: int, arg: object = None, mutate_flag: bool = True) -> int:
    return 0


__all__ = [
    "F_GETFD",
    "F_SETFD",
    "FD_CLOEXEC",
    "LOCK_EX",
    "LOCK_NB",
    "LOCK_SH",
    "LOCK_UN",
    "errno",
    "fcntl",
    "flock",
    "ioctl",
    "lockf",
]
