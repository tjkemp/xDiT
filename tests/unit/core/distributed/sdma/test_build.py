"""Finding MoRI for the SDMA transport's build, without importing it."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from xfuser.core.distributed.sdma import build


def _installed_mori(root: Path) -> Path:
    """The layout a `mori` wheel installs: its libraries next to a copy of its headers."""
    package = root / "mori"
    (package / "_jit-sources" / "include" / "mori").mkdir(parents=True)
    (package / "_jit-sources" / "3rdparty" / "spdlog" / "include").mkdir(parents=True)
    (package / "libmori_application.so").write_bytes(b"")
    return package


def test_finds_an_installed_mori_package(tmp_path, monkeypatch):
    package = _installed_mori(tmp_path)
    monkeypatch.delenv("XDIT_MORI_INCLUDE", raising=False)
    monkeypatch.delenv("XDIT_MORI_LIB_DIR", raising=False)
    monkeypatch.setattr(
        build.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(package)])
    )
    includes, lib = build.mori_paths()
    assert lib == package
    assert includes == [
        package / "_jit-sources" / "include",
        package / "_jit-sources" / "3rdparty" / "spdlog" / "include",
    ]


def test_a_source_build_named_by_the_environment_wins(tmp_path, monkeypatch):
    package = _installed_mori(tmp_path)
    monkeypatch.setattr(
        build.importlib.util, "find_spec", lambda name: SimpleNamespace(submodule_search_locations=[str(package)])
    )
    monkeypatch.setenv("XDIT_MORI_INCLUDE", f"{tmp_path / 'src' / 'include'}:{tmp_path / 'spdlog'}")
    monkeypatch.setenv("XDIT_MORI_LIB_DIR", str(tmp_path / "build"))
    assert build.mori_paths() == ([tmp_path / "src" / "include", tmp_path / "spdlog"], tmp_path / "build")


def test_no_mori_is_none(tmp_path, monkeypatch):
    monkeypatch.delenv("XDIT_MORI_INCLUDE", raising=False)
    monkeypatch.delenv("XDIT_MORI_LIB_DIR", raising=False)
    monkeypatch.setattr(build.importlib.util, "find_spec", lambda name: None)
    assert build.mori_paths() is None


def test_without_mori_the_build_says_what_to_install(tmp_path, monkeypatch):
    monkeypatch.delenv("XDIT_MORI_INCLUDE", raising=False)
    monkeypatch.delenv("XDIT_MORI_LIB_DIR", raising=False)
    monkeypatch.setattr(build.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match=r"needs MoRI .*XDIT_MORI_INCLUDE"):
        build.build(tmp_path, ["gfx950"])
    assert not list(tmp_path.iterdir())
