#!/usr/bin/env python
"""Run one pty-chi reconstruction as a self-contained child process.

Reads a standard-layout input directory, streams progress back to the launching process as
newline-delimited JSON, and stops cleanly when it is asked to. No settings registry, no
product repository, no task manager -- the inputs are exactly the files any ptychodus run
leaves behind.

This is an ordinary operating-system process addressed over a pipe, which is what makes it
usable from a service that must not import ``ptychodus.model``. It is unrelated to the
``multiprocessing`` child the GUI spawns per reconstruction, despite the similar name.

Protocol
--------

stdin
    Nothing, unless ``--ptychi-options-file -`` asks for the pty-chi options JSON to be read
    from it. The ``options_class_name`` that JSON carries is the algorithm identity, so
    PIE/ePIE/rPIE cannot silently swap for each other on the way through.

events
    One JSON object per line: ``started``, ``epoch``, ``checkpoint``, ``cancelling``,
    ``finished``, ``error``. They go to stdout by default, and fds 1 and 2 are redirected
    into a log file in the output directory first, so pty-chi progress bars and torch
    chatter cannot corrupt the stream. ``--event-file`` sends them to a file instead and
    leaves stdout and stderr alone, which is what a launcher that owns stdout itself needs.

Cancellation
    SIGTERM or SIGINT sets a flag that is checked between sync chunks. pty-chi's
    ``task.run()`` has no interruption point, so a cancel takes effect up to
    ``--num-sync-epochs`` epochs later; the partially converged product is still written.
    A second signal exits immediately.

Exit codes
    See :class:`ExitCode`: 0 success, 1 reconstruction or I/O failure, 2 bad arguments or
    unparseable options, 130 cancelled.

Scope and limitations
---------------------

- One scan, one algorithm, taken from the options file. There is no queue and no retry.
- Inputs and outputs are the ptychodus standard layout; there are no per-file overrides.
  Point ``-i`` at a directory another driver wrote, or at one staged for this purpose.
- A signal that arrives before the handlers are installed -- during the torch import, the
  argument parsing, or the output redirect -- cannot be caught and kills the process. There
  is nothing computed to save that early, so a launcher should wait for ``started`` before
  it cancels.
- No GPU selection or thread-count side effects. Choose a device with
  ``CUDA_VISIBLE_DEVICES`` in the environment.
- Under a multi-process launch only the main process emits events and writes artifacts; the
  other ranks log to their own files and are silent on the event stream.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import IO, Any

import ptychodus
from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import StandardFileLayout, load_diffraction_data, load_product
from ptychodus.api.reconstruct import (
    PositionIndexFilter,
    ReconstructOutput,
    prepare_reconstruct_input,
)
from ptychodus.cli import DirectoryType
from ptychodus.cli._reconstruct_common import (
    CancellationToken,
    add_ptychi_options_argument,
    install_signal_handlers,
    load_ptychi_options,
    process_rank,
    run_reconstruction,
)

logger = logging.getLogger('reconstruct_subprocess')

LOG_BASENAME = 'reconstruct'


class EventStream:
    """The machine-readable half of this process's output.

    Writing a line per event rather than a final report is what lets a launcher show
    progress and decide to cancel while the reconstruction is still running.
    """

    def __init__(self, events: IO[str] | None) -> None:
        self._events = events
        self._started_s = time.monotonic()

    def elapsed_s(self) -> float:
        return round(time.monotonic() - self._started_s, 3)

    def emit(self, event: str, **fields: Any) -> None:
        """Write one line of the event stream, or nothing on a process that has no stream.

        ``allow_nan=False`` keeps the stream to strict JSON. Python would otherwise emit
        bare ``Infinity`` / ``NaN`` literals, which parsers outside Python reject -- and a
        diverging reconstruction produces infinite losses readily. Non-finite values are
        normalized to ``null`` before they get here.
        """
        if self._events is None:
            return

        print(
            json.dumps({'event': event, **fields}, allow_nan=False),
            file=self._events,
            flush=True,
        )

    def emit_exception(self, exc: BaseException) -> None:
        self.emit(
            'error',
            type=type(exc).__name__,
            message=str(exc),
            traceback=traceback.format_exc(),
        )


def _positive_int(text: str) -> int:
    value = int(text)

    if value < 1:
        raise argparse.ArgumentTypeError(f'"{text}" must be at least 1!')

    return value


def log_file(output_directory: Path, rank: int) -> Path:
    """Where this process writes its human-readable log.

    Every rank of a multi-process launch logs, and a single name would have them all
    truncating one file, so only the main process gets the plain one.
    """
    if rank == 0:
        return output_directory / f'{LOG_BASENAME}.log'

    return output_directory / f'{LOG_BASENAME}.rank{rank:02d}.log'


def resolve_ptychi_options_file(options_file: Path | None, input_directory: Path) -> Path | None:
    """Pick the options file to read, or `None` to fall back to stock options.

    An explicit argument always wins, including the ``-`` that means stdin. Otherwise the
    staged options in the input directory are used when they are there; no current staging
    path writes them, so their absence is the ordinary case and not an error.
    """
    if options_file is not None:
        return options_file

    staged_file = StandardFileLayout.PTYCHI_OPTIONS.path(input_directory)

    if staged_file.is_file():
        return staged_file

    logger.warning(
        'No %s in %s; reconstructing with stock LSQML options. '
        'Pass --ptychi-options-file to choose another algorithm.',
        StandardFileLayout.PTYCHI_OPTIONS.value,
        input_directory,
    )
    return None


def _finite_loss(output: ReconstructOutput) -> float | None:
    """The last loss as strict JSON can carry it: a number, or null."""
    losses = output.product.losses

    if not losses:
        return None

    loss = float(losses[-1].value)
    return loss if math.isfinite(loss) else None


def _reconstruct(
    stream: EventStream,
    cancellation: CancellationToken,
    args: argparse.Namespace,
    log_path: Path | None,
) -> ExitCode:
    input_directory: Path = args.input_directory
    output_directory: Path = args.output_directory

    try:
        task_options = load_ptychi_options(
            resolve_ptychi_options_file(args.ptychi_options_file, input_directory)
        )
    except (OSError, TypeError, ValueError) as exc:
        stream.emit_exception(exc)
        logger.exception('Failed to read the pty-chi options.')
        return ExitCode.USAGE

    assembled_data = load_diffraction_data(
        StandardFileLayout.DIFFRACTION.path(input_directory), mmap_file=args.mmap_file
    )
    product = load_product(StandardFileLayout.PRODUCT.path(input_directory))
    reconstruct_input = prepare_reconstruct_input(
        assembled_data,
        product,
        index_filter=PositionIndexFilter[args.index_filter.upper()],
    )

    num_epochs = int(task_options.reconstructor_options.num_epochs)
    product_file = StandardFileLayout.PRODUCT.path(output_directory)

    stream.emit(
        'started',
        pid=os.getpid(),
        reconstructor=task_options.reconstructor_options.get_reconstructor_type().value,
        options_class=type(task_options).__name__,
        num_epochs=num_epochs,
        num_sync_epochs=args.num_sync_epochs,
        num_patterns=int(reconstruct_input.diffraction_patterns.shape[0]),
        product_output=str(product_file),
        log=None if log_path is None else str(log_path),
    )

    # The sync callback is the only place the epoch number is visible: run_reconstruction
    # returns the product alone, and losses are not written by every reconstructor.
    last_epoch = 0

    def on_sync(output: ReconstructOutput, checkpoint_file: Path | None) -> None:
        nonlocal last_epoch
        last_epoch = output.progress

        stream.emit(
            'epoch',
            epoch=output.progress,
            num_epochs=num_epochs,
            loss=_finite_loss(output),
            elapsed_s=stream.elapsed_s(),
        )

        if checkpoint_file is not None:
            stream.emit('checkpoint', epoch=output.progress, path=str(checkpoint_file))

    final_product = run_reconstruction(
        logger,
        reconstruct_input,
        task_options,
        output_directory,
        num_sync_epochs=args.num_sync_epochs,
        cancellation=cancellation,
        on_sync=on_sync,
    )

    if final_product is None:
        stream.emit('finished', epoch=0, cancelled=True, path=None, elapsed_s=stream.elapsed_s())
        return ExitCode.CANCELLED

    is_cancelled = cancellation.is_cancelled
    stream.emit(
        'finished',
        epoch=last_epoch,
        cancelled=is_cancelled,
        path=str(product_file),
        elapsed_s=stream.elapsed_s(),
    )
    return ExitCode.CANCELLED if is_cancelled else ExitCode.SUCCESS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run one pty-chi reconstruction over a standard-layout directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=True),
        help='Directory holding diffraction.h5, product.h5 and optionally ptychi_options.json.',
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=False),
        help=(
            'Destination directory, written in the ptychodus standard layout: '
            'ptychi_options.json, per-epoch product.NNNNNN.h5 checkpoints, the final '
            'product.h5, and the log.'
        ),
    )
    add_ptychi_options_argument(
        parser, default_note='Defaults to the one in INPUT_DIR, else to stock LSQML.'
    )
    parser.add_argument(
        '--index-filter',
        choices=tuple(member.name.lower() for member in PositionIndexFilter),
        default=PositionIndexFilter.ALL.name.lower(),
        help='Scan index subset to reconstruct.',
    )
    parser.add_argument(
        '--num-sync-epochs',
        default=1,
        type=_positive_int,
        help='Epochs between progress events, checkpoints, and cancellation checks.',
    )
    parser.add_argument(
        '--mmap-file',
        metavar='MMAP_FILE',
        type=Path,
        help='Stage diffraction patterns into this memmap instead of loading into RAM.',
    )
    parser.add_argument(
        '--event-file',
        metavar='EVENT_FILE',
        type=Path,
        help=(
            'Write the event stream here and leave stdout and stderr alone. Without it the '
            'events go to stdout and the rest of the output is redirected into the log.'
        ),
    )
    parser.add_argument(
        '--log-level',
        default=logging.INFO,
        help='Python logging level.',
        type=int,
    )
    parser.add_argument(
        '-v',
        '--version',
        action='version',
        version=ptychodus.VERSION_STRING,
    )
    return parser


def _open_event_stream(event_file: Path | None) -> IO[str]:
    """Open the event stream, taking stdout over when no file was named.

    Taking it over means duplicating fd 1 and then pointing fds 1 and 2 at the log, so that
    the stream survives while everything else written to either -- including from the C
    extensions that ``sys.stdout`` reassignment would not reach -- lands in the log instead.
    """
    if event_file is not None:
        return open(event_file, 'w', buffering=1)

    return os.fdopen(os.dup(1), 'w', buffering=1)


def main() -> ExitCode:
    args = _build_parser().parse_args()

    output_directory: Path = args.output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    rank = process_rank()

    # Only one process may speak on the stream a launcher reads, and only one may write the
    # artifacts; both follow the same rank guard.
    events = _open_event_stream(args.event_file) if rank == 0 else None
    log_path: Path | None = None

    if args.event_file is None:
        # Taking stdout over for the events means everything else has to go somewhere, and
        # a per-rank name keeps the ranks from truncating each other's.
        log_path = log_file(output_directory, rank)

        with open(log_path, 'w') as handle:
            os.dup2(handle.fileno(), 1)
            os.dup2(handle.fileno(), 2)

        sys.stdout = os.fdopen(os.dup(1), 'w', buffering=1)
        sys.stderr = os.fdopen(os.dup(2), 'w', buffering=1)

    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    stream = EventStream(events)
    cancellation = install_signal_handlers(
        CancellationToken(on_cancel=lambda reason: stream.emit('cancelling', signal=reason))
    )

    try:
        return _reconstruct(stream, cancellation, args, log_path)
    except Exception as exc:
        stream.emit_exception(exc)
        logger.exception('Reconstruction failed.')
        return ExitCode.FAILURE


if __name__ == '__main__':
    sys.exit(main())
