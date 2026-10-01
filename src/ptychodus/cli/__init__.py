"""Argument-parsing and logging helpers shared by the packaged ptychodus commands.

Everything here is dependency-free on purpose: ``convert-to-ptychodus`` and
``ptychodus-system-check`` run on a bare install, so this module must not reach anything
behind an optional extra.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

__all__ = [
    'DirectoryType',
    'add_log_level_argument',
    'configure_logging',
    'positive_int',
    'verify_all_arguments_parsed',
]


class DirectoryType:
    def __init__(self, *, must_exist: bool) -> None:
        self._must_exist = must_exist

    def __call__(self, string: str) -> Path:
        path = Path(string)

        if self._must_exist and not path.is_dir():
            raise argparse.ArgumentTypeError(f'"{string}" is not a directory!')

        return path


def positive_int(text: str) -> int:
    """An argparse type for a count that is meaningless at zero or below."""
    value = int(text)

    if value < 1:
        raise argparse.ArgumentTypeError(f'"{text}" must be at least 1!')

    return value


def verify_all_arguments_parsed(parser: argparse.ArgumentParser, argv: list[str]) -> None:
    if argv:
        parser.error('unrecognized arguments: %s' % ' '.join(argv))


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """Add ``--log-level``, so its wording is written once rather than thirteen times."""
    parser.add_argument(
        '--log-level',
        default=logging.INFO,
        type=int,
        help='Python logging level.',
    )


def configure_logging(args: argparse.Namespace) -> None:
    """Send logs to stderr at the level `args` asked for.

    Separate from :func:`add_log_level_argument` because a caller that redirects its
    standard streams has to do so before this runs: ``basicConfig`` captures `sys.stderr`
    by value, and a handler bound to the pre-redirect stream writes to the wrong place.
    """
    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
