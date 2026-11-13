"""What the SDMA transport says without MoRI."""

import sys

import pytest

from xfuser.core.distributed.sdma import transport


def test_without_mori_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "mori.ccl", None)
    with pytest.raises(RuntimeError, match=r"--ulysses_a2a sdma needs MoRI .*pip install"):
        transport.check_available()
    assert not transport.available()
