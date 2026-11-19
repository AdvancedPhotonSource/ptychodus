#!/usr/bin/env python
"""Run one reconstruction, with any installed reconstructor, as a self-contained child process.

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
    Nothing, unless ``--options -`` asks for the options document to be read from it. The
    document names the library and algorithm beside the backend's own options.

events
    One JSON object per line, in the vocabulary of :mod:`ptychodus.api.processing`:

    - ``started``, with the ``library`` and ``algorithm``;
    - ``progress``, for each output the reconstructor yields: ``progress`` towards
      ``progress_goal`` (zero when unknown) and the last ``loss``;
    - ``checkpoint``, for each one saved: the ``path`` of the snapshot at that ``progress``;
    - ``cancelling``, when a signal asks the run to stop;
    - ``finished``, always last: the run's ``state`` (``succeeded``, ``cancelled`` or
      ``failed``), its ``reason`` (the signal, or the exception's type), the ``path`` of its
      result, and for a failure the traceback in ``detail``.

    They go to stdout by default, and fds 1 and 2 are redirected into ``reconstruct.log``
    in the output directory first, so progress bars and torch chatter cannot corrupt the
    stream. ``--event-file`` sends them to a file instead and leaves stdout and stderr
    alone, which is what a launcher that owns stdout itself needs.

Cancellation
    SIGTERM or SIGINT sets a flag that is checked at each output the reconstructor yields.
    A backend has no interruption point between them, so a cancel takes effect at the next
    one -- every ``num_sync_epochs`` epochs for pty-chi; the partially converged product is
    still written.
    A second signal exits immediately.

Exit codes
    See :class:`ExitCode`: 0 success, 1 reconstruction or I/O failure, 2 bad arguments or
    unparseable options, 130 cancelled.

Scope and limitations
---------------------

- One scan, one algorithm, taken from the options document. There is no queue and no retry.
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
from ptychodus.api.processing import RunState
from ptychodus.api.reconstruct import PositionIndexFilter, ReconstructOutput
from ptychodus.api.plugins import PluginRegistry
from ptychodus.cli import (
    CancellationToken,
    DirectoryType,
    add_log_level_argument,
    configure_logging,
    install_signal_handlers,
    open_action_log,
)
from ptychodus.cli._reconstruct_common import (
    add_options_argument,
    add_print_options_argument,
    default_reconstruction,
    find_options_document,
    load_trained_model,
    pair_patterns_with_positions,
    process_rank,
    read_reconstruction,
    run_reconstruction,
)

logger = logging.getLogger('reconstruct_subprocess')


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

    def emit_failure(
        self,
        exc: BaseException,
        *,
        path: Path | None = None,
        progress: int = 0,
        progress_goal: int = 0,
    ) -> None:
        """The ``finished`` event of a run that failed with `exc`; `path` is what it kept."""
        self.emit(
            'finished',
            state=RunState.FAILED,
            reason=type(exc).__name__,
            progress=progress,
            progress_goal=progress_goal,
            path=None if path is None else str(path),
            elapsed_s=self.elapsed_s(),
            detail=traceback.format_exc(),
        )


def _finite_loss(output: ReconstructOutput) -> float | None:
    """The last loss as strict JSON can carry it: a number, or null."""
    losses = output.product.losses

    if not losses:
        return None

    loss = float(losses[-1].value)
    return loss if math.isfinite(loss) else None


class _SyncReporter:
    """Emit a ``progress`` event, and a ``checkpoint`` one where a file was written, per output."""

    def __init__(self, stream: EventStream) -> None:
        self._stream = stream
        self.progress = 0
        self.progress_goal = 0
        self.last_checkpoint: Path | None = None

    def __call__(self, output: ReconstructOutput, checkpoint_file: Path | None) -> None:
        self.progress = output.progress
        self.progress_goal = output.progress_goal
        self._stream.emit(
            'progress',
            progress=output.progress,
            progress_goal=output.progress_goal,
            loss=_finite_loss(output),
            elapsed_s=self._stream.elapsed_s(),
        )

        if checkpoint_file is not None:
            self.last_checkpoint = checkpoint_file
            self._stream.emit(
                'checkpoint',
                progress=output.progress,
                progress_goal=output.progress_goal,
                path=str(checkpoint_file),
            )


def _reconstruct(
    stream: EventStream,
    cancellation: CancellationToken,
    args: argparse.Namespace,
    log_path: Path | None,
) -> ExitCode:
    input_directory: Path = args.input_directory
    output_directory: Path = args.output_directory

    # A named or staged document is read before anything else is, so that options which do
    # not parse are a usage error however much data sits beside them.
    libraries = PluginRegistry.load_plugins().reconstructor_libraries

    try:
        options_file = find_options_document(args.options, input_directory)
        resolved = None if options_file is None else read_reconstruction(libraries, options_file)
    except (OSError, ValueError) as exc:
        stream.emit_failure(exc)
        logger.exception('Failed to read the options.')
        return ExitCode.USAGE

    assembled_data = load_diffraction_data(
        StandardFileLayout.DIFFRACTION.path(input_directory), mmap_file=args.mmap_file
    )
    product = load_product(StandardFileLayout.PRODUCT.path(input_directory))

    if resolved is None:
        # The stock algorithm follows the propagation regime, which only the product states.
        try:
            resolved = default_reconstruction(libraries, far_field=product.metadata.far_field)
        except ValueError as exc:
            stream.emit_failure(exc)
            logger.exception('No options, and no default to fall back to.')
            return ExitCode.USAGE

    reconstruct_input = pair_patterns_with_positions(
        logger,
        assembled_data,
        product,
        index_filter=PositionIndexFilter[args.index_filter.upper()],
    )
    load_trained_model(logger, resolved.reconstructor, input_directory)
    product_file = StandardFileLayout.PRODUCT.path(output_directory)

    stream.emit(
        'started',
        pid=os.getpid(),
        library=resolved.library,
        algorithm=resolved.reconstructor.name,
        num_patterns=int(reconstruct_input.diffraction_patterns.shape[0]),
        product_output=str(product_file),
        log=None if log_path is None else str(log_path),
    )

    # The sync callback is the only place the progress is visible: run_reconstruction
    # returns the product alone, and losses are not written by every reconstructor.
    reporter = _SyncReporter(stream)

    try:
        final_product = run_reconstruction(
            logger,
            resolved,
            reconstruct_input,
            output_directory,
            cancellation=cancellation,
            on_sync=reporter,
        )
    except Exception as exc:
        stream.emit_failure(
            exc,
            path=reporter.last_checkpoint,
            progress=reporter.progress,
            progress_goal=reporter.progress_goal,
        )
        logger.exception('Reconstruction failed.')
        return ExitCode.FAILURE

    if cancellation.is_cancelled:
        state, code = RunState.CANCELLED, ExitCode.CANCELLED
    else:
        state, code = RunState.SUCCEEDED, ExitCode.SUCCESS

    stream.emit(
        'finished',
        state=state,
        reason=cancellation.reason or None,
        progress=reporter.progress,
        progress_goal=reporter.progress_goal,
        path=None if final_product is None else str(product_file),
        elapsed_s=stream.elapsed_s(),
    )
    return code


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Run one reconstruction over a standard-layout directory.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '-i',
        '--input-directory',
        metavar='INPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=True),
        help='Directory holding diffraction.h5, product.h5 and optionally '
        'reconstruct_options.yaml and a trained model.',
    )
    parser.add_argument(
        '-o',
        '--output-directory',
        metavar='OUTPUT_DIR',
        required=True,
        type=DirectoryType(must_exist=False),
        help=(
            'Destination directory, written in the ptychodus standard layout: '
            'reconstruct_options.yaml, product.NNNNNN.h5 checkpoints, the final '
            'product.h5, and reconstruct.log.'
        ),
    )
    add_options_argument(
        parser,
        default_note=(
            'Defaults to the one in INPUT_DIR, else to stock pty-chi LSQML, or BH for a '
            'near-field product.'
        ),
    )
    add_print_options_argument(parser)
    parser.add_argument(
        '--index-filter',
        choices=tuple(member.name.lower() for member in PositionIndexFilter),
        default=PositionIndexFilter.ALL.name.lower(),
        help='Scan index subset to reconstruct.',
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
    add_log_level_argument(parser)
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
        # a per-rank name keeps the ranks from writing each other's.
        log_path = open_action_log(
            output_directory,
            StandardFileLayout.RECONSTRUCT_LOG,
            rank=rank,
            capture_descriptors=True,
        )
        configure_logging(args)
    else:
        configure_logging(args)
        log_path = open_action_log(output_directory, StandardFileLayout.RECONSTRUCT_LOG, rank=rank)

    stream = EventStream(events)
    cancellation = install_signal_handlers(
        CancellationToken(on_cancel=lambda reason: stream.emit('cancelling', signal=reason))
    )

    try:
        return _reconstruct(stream, cancellation, args, log_path)
    except Exception as exc:
        stream.emit_failure(exc)
        logger.exception('Reconstruction failed.')
        return ExitCode.FAILURE


if __name__ == '__main__':
    sys.exit(main())
