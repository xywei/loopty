"""Smoke tests for the loopty placeholder release."""

from __future__ import annotations

import loopty
from loopty import cli


def test_version() -> None:
    assert loopty.__version__ == "0.0.1"


def test_cli_main(capsys) -> None:
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "loopty" in out
    assert loopty.__version__ in out
    assert "work in progress: placeholder release" in out
    assert "https://github.com/xywei/loopty" in out
