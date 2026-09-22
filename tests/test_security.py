from pathlib import Path


def test_production_sources_contain_no_embedded_credentials() -> None:
    """No literal secret may be committed.

    Markers must be credential *values* or provider-specific formats, not Python
    parameter names or the scheme word in an Authorization header. Passing
    `api_key=<variable>` and building `Bearer <key>` are the correct patterns;
    matching those spellings flagged correct code and would push it toward
    hiding the assignment instead of removing a secret.
    """
    root = Path(__file__).parents[1]
    text = "\n".join(
        path.read_text("utf-8")
        for folder in ("custom_components", "blueprints")
        for path in (root / folder).rglob("*")
        if path.is_file() and path.suffix in {".py", ".json", ".yaml"}
    ).lower()
    for marker in (
        "ossaccesskey",
        "signature=",
        "sk-proj-",
        "sk-live-",
        "bearer eyj",  # a JWT, i.e. a literal token rather than a variable
    ):
        assert marker not in text, f"possible embedded credential: {marker}"


def test_cleanup_code_has_no_recursive_root_delete() -> None:
    source = (
        Path(__file__).parents[1] / "custom_components/frigate_vision/media.py"
    ).read_text("utf-8")
    assert "rmtree(self._root" not in source
    assert "unlink()" in source
