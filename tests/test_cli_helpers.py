"""The argparse helpers in ``ptychodus.cli``.

They are tested here rather than through a driver because the whole point of the module is
that it works on a bare install: nothing it imports may come from an optional extra.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
import argparse
import io
import logging
import sys

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


@pytest.fixture
def unconfigured_root_logger() -> Iterator[Callable[[], None]]:
    """Yield a callable that empties the root logger, and restore it afterwards.

    ``logging.basicConfig`` does nothing at all once the root logger has handlers,
    and under pytest it always does -- so without emptying it first, the call under
    test has no observable effect and the assertions below would pass vacuously.
    Emptying has to happen inside the test body rather than here: pytest's logging
    plugin installs its capture handler after fixture setup, so anything cleared
    here is back by the time the test runs.
    """
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level

    try:
        yield root.handlers.clear
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_configure_logging_sets_the_root_level_from_the_namespace(
    unconfigured_root_logger: Callable[[], None],
) -> None:
    # A level no default sets: the root logger starts at WARNING, so asserting
    # WARNING would pass even if configure_logging did nothing at all.
    unusual_level = logging.WARNING + 3
    unconfigured_root_logger()

    configure_logging(argparse.Namespace(log_level=unusual_level))

    assert logging.getLogger().level == unusual_level


def test_configure_logging_binds_the_stderr_it_sees_at_call_time(
    unconfigured_root_logger: Callable[[], None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pins the reason this is a separate function from ``add_log_level_argument``.

    ``basicConfig`` captures ``sys.stderr`` by value, so a caller that redirects its
    streams has to do so first or its logs go to the pre-redirect stream.
    """
    stream = io.StringIO()
    monkeypatch.setattr(sys, 'stderr', stream)
    unconfigured_root_logger()

    configure_logging(argparse.Namespace(log_level=logging.INFO))

    # Not "exactly one handler": pytest's logging plugin installs its own capture
    # handler on the root logger too.
    bound_streams = [
        handler.stream
        for handler in logging.getLogger().handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    assert stream in bound_streams


def test_module_imports_nothing_from_an_optional_extra() -> None:
    """``convert-to-ptychodus`` runs on a bare install, so this module must stay light."""
    import ptychodus.cli

    source = Path(ptychodus.cli.__file__).read_text()
    for forbidden in ('ptychi', 'torch', 'PyQt5', 'ptychodus.model'):
        assert forbidden not in source
