"""What the SDMA transport says when it can be neither loaded nor built."""

import pytest

from xfuser.core.distributed.sdma import transport


def test_without_mori_says_what_to_install(monkeypatch):
    monkeypatch.delenv("XDIT_SDMA_LIB", raising=False)
    monkeypatch.setattr(transport, "_packaged", lambda: None)
    monkeypatch.setattr(transport, "mori_paths", lambda: None)
    with pytest.raises(RuntimeError, match=r"--ulysses_a2a sdma needs MoRI .*pip install"):
        transport.check_available()


def test_a_prebuilt_library_needs_no_mori(monkeypatch, tmp_path):
    monkeypatch.setenv("XDIT_SDMA_LIB", str(tmp_path / "libxdit_sdma.so"))
    monkeypatch.setattr(transport, "mori_paths", lambda: None)
    transport.check_available()
