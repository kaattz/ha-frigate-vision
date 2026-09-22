"""Local-only pytest plugin that neutralises socket blocking on Windows.

Why this exists
---------------
`pytest_homeassistant_custom_component` calls
`pytest_socket.disable_socket(allow_unix_socket=True)` on every test, because
asyncio on Linux builds its self-pipe from an AF_UNIX socketpair.

Windows has no AF_UNIX socketpair: `asyncio` falls back to an AF_INET pair, so
the guard rejects the event loop itself and every async test errors before it
can run. The loop is constructed while pytest-asyncio resolves its runner
fixture, which happens before any `pytest_runtest_setup` hook can undo the
guard, so the guard is neutralised at import time instead.

Importing this module replaces `pytest_socket.disable_socket` with a no-op.
The HA plugin still calls `socket_allow_hosts(["127.0.0.1"])`, so this does not
widen real network access; it only stops the guard from rejecting the loop.

Developer aid for running the suite on Windows, loaded via
`-p tests.windows_socket`. It is never part of the integration or of CI, where
the suite runs on Linux and the stock guard applies unchanged.
"""

from __future__ import annotations

import pytest_socket

_original_disable_socket = pytest_socket.disable_socket


def _noop_disable_socket(allow_unix_socket: bool = False) -> None:
    """Keep sockets usable so asyncio can construct its self-pipe."""


pytest_socket.disable_socket = _noop_disable_socket

__all__ = ["_original_disable_socket"]
