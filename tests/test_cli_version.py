"""The command's advertised version follows the installed package."""
import pytest

from dagflow import __version__
from dagflow.cli.__main__ import main


def test_version_matches_package(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"dagflow {__version__}"
