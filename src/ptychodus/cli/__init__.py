"""Argument-parsing and logging helpers shared by the packaged ptychodus commands.

Everything here is dependency-free on purpose: ``ptychodus-system-check`` runs on a
bare install, so this module must not reach anything behind an optional extra.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from types import FrameType
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID
import argparse
import logging
import os
import shlex
import signal
import sys
import threading
from pathlib import Path

if TYPE_CHECKING:
    from ptychodus.api.assemble import DetectorCorrections
    from ptychodus.api.diffraction import DiffractionDataset
    from ptychodus.api.io import StandardFileLayout
    from ptychodus.api.plugins import PluginRegistry
    from ptychodus.api.processing import RunRecord
    from ptychodus.api.storage import StorageLayout

__all__ = [
    'CancellationToken',
    'DirectoryType',
    'add_detector_map_arguments',
    'add_log_level_argument',
    'configure_logging',
    'crop_extent_px',
    'install_signal_handlers',
    'open_action_log',
    'positive_int',
    'read_detector_corrections',
    'record_run_end',
    'report_resource',
    'verify_all_arguments_parsed',
]


_LOG_FORMAT = '%(asctime)s %(levelname)s %(name)s: %(message)s'


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


def crop_extent_px(text: str) -> int | Literal['auto', 'full']:
    """An argparse type for a crop side in pixels, ``auto`` to estimate one, or ``full``."""
    if text in ('auto', 'full'):
        return cast(Literal['auto', 'full'], text)

    try:
        return positive_int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f'"{text}" must be a positive integer, "auto" or "full"!'
        ) from exc


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
    logging.basicConfig(level=args.log_level, stream=sys.stderr, format=_LOG_FORMAT)


def open_action_log(
    output_directory: Path,
    log: StandardFileLayout,
    *,
    rank: int = 0,
    capture_descriptors: bool = False,
) -> Path:
    """Append this run to the action's log in `output_directory`, and return the file.

    `log` is the action's ``StandardFileLayout`` log member; a process of rank `rank` above
    zero writes its own suffixed file, so ranks never share one. The run opens with a line
    naming the time and the command, since a log is appended to rather than replaced.

    With `capture_descriptors`, file descriptors 1 and 2 are pointed at the file, which
    captures what C extensions write as well as the log records -- for a process whose
    console no person reads. Call it before :func:`configure_logging`, which binds the
    stream it finds. Otherwise the console stays as it is and the root logger gains a
    handler writing its records to the file too; call it after :func:`configure_logging`.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    log_file = log.rank_path(output_directory, rank)
    command = shlex.join([Path(sys.argv[0]).name, *sys.argv[1:]])

    with open(log_file, 'a') as handle:
        handle.write(f'# {datetime.now().isoformat(timespec="seconds")} {command}\n')

        if capture_descriptors:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)

    if capture_descriptors:
        sys.stdout = os.fdopen(os.dup(1), 'w', buffering=1)
        sys.stderr = os.fdopen(os.dup(2), 'w', buffering=1)
    else:
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.getLogger().addHandler(handler)

    return log_file


def report_resource(logger: logging.Logger, kind: str, uuid: UUID, folder: Path) -> None:
    """Announce a resource a run is producing: logged, and its uuid printed on stdout.

    Printed alone on its own line, so a script can capture the uuid while the run is still
    under way and watch the resource fill in.
    """
    logger.info('Writing %s %s in %s', kind, uuid, folder)
    print(uuid, flush=True)


def record_run_end(
    logger: logging.Logger,
    layout: StorageLayout,
    uuid: UUID,
    *,
    data_file: Path | None,
    run: RunRecord,
) -> bool:
    """Record how a run producing resource `uuid` ended; whether that could be written.

    A failure to write is logged rather than raised, since it leaves the resource
    ``running`` with a stale ``updated_at`` -- which is what a reader sees of a run that
    died -- and the command still has an exit status to give.
    """
    from ptychodus.api.storage import finish_run

    try:
        finish_run(layout, uuid, data_file=data_file, run=run)
    except Exception:
        logger.exception('Could not record the end of %s; it is left running.', uuid)
        return False

    return True


class CancellationToken:
    """A request to stop, raised asynchronously and read at the next safe point.

    The flag is a :class:`threading.Event` so that setting it from a signal handler does no
    real work. Nothing is logged there either: the logging machinery takes a lock, and a
    signal delivered while that lock is held would deadlock the process it was meant to stop.
    `on_cancel` exists for a caller that must react the instant the request arrives rather
    than at the next safe point, and runs under the same constraint.
    """

    def __init__(self, *, on_cancel: Callable[[str], None] | None = None) -> None:
        self._event = threading.Event()
        self._on_cancel = on_cancel
        self._reason = ''

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        """What asked for the cancellation, or an empty string while none has."""
        return self._reason

    def cancel(self, reason: str = '') -> None:
        """Request cancellation. Repeat requests are ignored, so `on_cancel` runs once."""
        if self._event.is_set():
            return

        self._reason = reason
        self._event.set()

        if self._on_cancel is not None:
            self._on_cancel(reason)


def install_signal_handlers(token: CancellationToken | None = None) -> CancellationToken:
    """Make SIGINT and SIGTERM request cancellation rather than kill the process.

    Returns the token they set, creating one when none is supplied. A **second** signal of
    either kind restores the default disposition and re-raises, so a run wedged somewhere
    without an interruption point can still be killed from the same terminal that asked it
    to stop politely.

    Signals delivered before this runs -- during imports, argument parsing, or the first
    file reads -- are not caught and end the process outright. There is nothing computed to
    save that early, which is why this is called once the logging is up rather than first.
    """
    cancellation = CancellationToken() if token is None else token

    def handle_signal(signum: int, _frame: FrameType | None) -> None:
        if cancellation.is_cancelled:
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return

        cancellation.cancel(signal.Signals(signum).name)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    return cancellation


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
