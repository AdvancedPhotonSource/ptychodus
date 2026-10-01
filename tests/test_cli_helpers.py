"""The argparse helpers in ``ptychodus.cli``.

They are tested here rather than through a driver because the whole point of the module is
that it works on a bare install: nothing it imports may come from an optional extra.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pytest

from ptychodus.cli import (
    DirectoryType,
    add_log_level_argument,
    configure_logging,
    positive_int,
)


def test_positive_int_accepts_one_and_above() -> None:
    assert positive_int('1') == 1
    assert positive_int('100') == 100


@pytest.mark.parametrize('text', ['0', '-1'])
def test_positive_int_rejects_zero_and_below(text: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int(text)


def test_positive_int_rejects_non_numeric() -> None:
    with pytest.raises(ValueError):
        positive_int('nope')


def test_directory_type_accepts_a_missing_path_when_it_may_not_exist(tmp_path: Path) -> None:
    resolved = DirectoryType(must_exist=False)(str(tmp_path / 'not-yet'))
    assert resolved == tmp_path / 'not-yet'


def test_directory_type_rejects_a_missing_path_when_it_must_exist(tmp_path: Path) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        DirectoryType(must_exist=True)(str(tmp_path / 'absent'))


def test_directory_type_rejects_a_file_when_it_must_exist(tmp_path: Path) -> None:
    a_file = tmp_path / 'f.txt'
    a_file.write_text('')

    with pytest.raises(argparse.ArgumentTypeError):
        DirectoryType(must_exist=True)(str(a_file))


def test_log_level_argument_defaults_to_info() -> None:
    parser = argparse.ArgumentParser()
    add_log_level_argument(parser)
    assert parser.parse_args([]).log_level == logging.INFO
    assert parser.parse_args(['--log-level', '10']).log_level == logging.DEBUG


def test_configure_logging_is_callable_with_a_parsed_namespace() -> None:
    """A smoke test: basicConfig is a no-op once the root logger has handlers."""
    configure_logging(argparse.Namespace(log_level=logging.WARNING))


def test_module_imports_nothing_from_an_optional_extra() -> None:
    """``convert-to-ptychodus`` runs on a bare install, so this module must stay light."""
    import ptychodus.cli

    source = Path(ptychodus.cli.__file__).read_text()
    for forbidden in ('ptychi', 'torch', 'PyQt5', 'ptychodus.model'):
        assert forbidden not in source
