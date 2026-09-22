"""Local-only pytest plugin that makes `tmp_path` usable on Windows.

Why this exists
---------------
pytest creates each `tmp_path` directory through `tempfile.mkdtemp`, which asks
for mode `0o700`. The Windows sandbox this suite is developed under turns that
request into an ACL it then refuses to write through, so every test requesting
`tmp_path` dies with `PermissionError: [Errno 13]` as soon as it writes a file.

A directory created without an explicit mode is writable, so this plugin wraps
the factory's `mktemp` to create the directory itself and then delegate the rest
to pytest. Numbering, `tmp_path` wiring and cleanup are unchanged.

This is a developer aid for running the suite on Windows. It is never part of
the integration and never runs in CI, where the suite runs on Linux and the
stock fixtures apply unchanged.
"""

from __future__ import annotations

import os
import pathlib
import shutil
import tempfile
import uuid

import pytest

_PATCHED = "_dsh_windows_tmpdir_patched"


def _writable_mkdtemp(
    *, suffix: str | None = None, prefix: str | None = None, dir: str | None = None
) -> str:
    """`tempfile.mkdtemp` without the mode-0o700 directory.

    The sandbox refuses writes through a directory created with mode 0o700,
    which is what `mkdtemp` asks for, so the directory is created plainly.
    """
    base = pathlib.Path(dir) if dir else pathlib.Path.cwd()
    name = f"{prefix or 'tmp'}{uuid.uuid4().hex[:8]}{suffix or ''}"
    target = base / name
    target.mkdir(parents=True, exist_ok=True)
    return str(target)


def pytest_configure(config: pytest.Config) -> None:  # noqa: ARG001
    """Install the writable `mkdtemp` for the whole session.

    `MediaManager` stages frames via `tempfile.mkdtemp`, so the substitution has
    to be process-wide rather than confined to the `tmp_path` fixture.
    """
    tempfile.mkdtemp = _writable_mkdtemp  # type: ignore[assignment]


@pytest.fixture(scope="session")
def tmp_path_factory(
    request: pytest.FixtureRequest,  # noqa: ARG001
    tmp_path_factory: pytest.TempPathFactory,
) -> pytest.TempPathFactory:
    """Re-root the temp factory in the workspace and avoid mode-0o700 dirs."""
    configured = os.environ.get("DSH_PYTEST_BASETEMP", ".pytest-tmp")
    root = pathlib.Path(configured)
    root = root if root.is_absolute() else pathlib.Path.cwd() / root
    root = root.resolve().with_name(f"{root.name}-{os.getpid()}")
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    if not getattr(tmp_path_factory, _PATCHED, False):

        def _mktemp(basename: str, numbered: bool = True) -> pathlib.Path:
            name = f"{basename}{uuid.uuid4().hex[:8]}" if numbered else basename
            target = root / name
            target.mkdir(parents=True, exist_ok=True)
            return target

        tmp_path_factory.mktemp = _mktemp  # type: ignore[method-assign]
        setattr(tmp_path_factory, _PATCHED, True)

    tmp_path_factory._basetemp = root  # noqa: SLF001 - documented pytest hook
    return tmp_path_factory


__all__: list[str] = []
