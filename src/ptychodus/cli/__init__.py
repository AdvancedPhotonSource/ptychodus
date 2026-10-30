"""Argument-parsing and logging helpers shared by the packaged ptychodus commands.

Everything here is dependency-free on purpose: ``ptychodus-system-check`` runs on a
bare install, so this module must not reach anything behind an optional extra.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import argparse
import logging
import sys
from pathlib import Path

if TYPE_CHECKING:
    from ptychodus.api.assemble import DetectorCorrections
    from ptychodus.api.diffraction import DiffractionDataset
    from ptychodus.api.plugins import PluginRegistry

__all__ = [
    'DirectoryType',
    'add_detector_map_arguments',
    'add_log_level_argument',
    'configure_logging',
    'positive_int',
    'positive_int_or_auto',
    'read_detector_corrections',
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


def positive_int_or_auto(text: str) -> int | Literal['auto']:
    """An argparse type for a positive count, or the word ``auto`` to have one estimated."""
    if text == 'auto':
        return 'auto'

    try:
        return positive_int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f'"{text}" must be a positive integer or "auto"!') from exc


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


def add_detector_map_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the options naming a measured detector background and flat field."""
    parser.add_argument(
        '--background-file',
        type=Path,
        default=None,
        help=(
            'Measured detector background covering the full detector, in counts per frame at'
            " the patterns' exposure. Overrides the diffraction file."
        ),
    )
    parser.add_argument(
        '--background-file-type',
        default='NPY',
        help='Name of the background file reader plugin.',
    )
    parser.add_argument(
        '--subtract-background',
        action='store_true',
        help='Subtract the background from the patterns rather than carrying it alongside.',
    )
    parser.add_argument(
        '--flat-field-file',
        type=Path,
        default=None,
        help=(
            'Flat-field response covering the full detector: the signal under uniform'
            ' illumination, which a measurement is divided by. Overrides the diffraction file.'
        ),
    )
    parser.add_argument(
        '--flat-field-file-type',
        default='NPY',
        help='Name of the flat-field file reader plugin.',
    )
    parser.add_argument(
        '--divide-flat-field',
        action='store_true',
        help='Divide the patterns by the flat field rather than carrying it alongside.',
    )


def read_detector_corrections(
    registry: PluginRegistry, args: argparse.Namespace, dataset: DiffractionDataset
) -> DetectorCorrections:
    """Build detector corrections from the options :func:`add_detector_map_arguments` adds.

    A map whose file option is absent falls back to the one `dataset` carries.
    """
    from ptychodus.api.assemble import DetectorCorrections

    logger = logging.getLogger(__name__)
    background = dataset.get_background()
    flat_field = dataset.get_flat_field()

    if args.background_file is not None:
        logger.info('Reading the detector background from %s', args.background_file)
        reader = registry.background_file_readers.get_strategy_by_name(args.background_file_type)
        background = reader.read(args.background_file)

    if args.flat_field_file is not None:
        logger.info('Reading the flat field from %s', args.flat_field_file)
        reader = registry.flat_field_file_readers.get_strategy_by_name(args.flat_field_file_type)
        flat_field = reader.read(args.flat_field_file)

    return DetectorCorrections(
        background=background,
        flat_field=flat_field,
        subtract_background=args.subtract_background,
        divide_flat_field=args.divide_flat_field,
    )
