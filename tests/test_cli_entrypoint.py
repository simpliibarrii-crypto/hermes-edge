"""Smoke tests for the installed Hermes command."""

from __future__ import annotations

import sys

import pytest

from hermes.cli import main


def test_cli_help_is_available(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    """The console entry point must parse before loading model dependencies."""
    monkeypatch.setattr(sys, "argv", ["hermes", "--help"])

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 0
    assert "Hermes Edge" in capsys.readouterr().out
