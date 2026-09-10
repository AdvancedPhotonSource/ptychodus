#!/usr/bin/env python
"""Run one pty-chi reconstruction as a self-contained child process.

Reads an ``AssembledDiffractionData`` file and a ``Product`` file from disk,
takes a serialized pty-chi options object on stdin, and streams progress back
to the launching process as newline-delimited JSON on stdout. No settings
registry, no product repository, no task manager -- the inputs are exactly what
``ptychodus-bdp`` stages.

Protocol
--------

stdin
    A JSON envelope ``{"reconstructor": "<enum-value>", "options": {...}}``
    produced by :func:`ptychodus.model.ptychi.task.dump_task_options`, read to
    EOF. The envelope carries the algorithm identity, so PIE/ePIE/rPIE cannot
    silently swap for each other on the way through. Nothing else is read from
    stdin, so the launcher should close it immediately.

stdout
    One JSON object per line: ``started``, ``epoch``, ``checkpoint``,
    ``cancelling``, ``finished``, ``error``. Nothing else is ever written to
    stdout -- fds 1 and 2 are redirected into a log file beside the output
    product before the reconstruction starts, so pty-chi progress bars and
    torch chatter cannot corrupt the stream.

Cancellation
    SIGTERM or SIGINT sets a flag that is checked between sync chunks. pty-chi's
    ``task.run()`` has no interruption point, so a cancel takes effect up to
    ``--num-sync-epochs`` epochs later; the partially converged product is still
    written to ``--product-output``. A second signal exits immediately.

Exit codes
    See :class:`ExitCode`: 0 success, 1 reconstruction or I/O failure, 2 bad
    arguments or unparseable stdin, 130 cancelled.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import traceback
from enum import IntEnum
from pathlib import Path
from types import FrameType
from typing import IO, Any

import ptychodus
from ptychodus.api.io import load_diffraction_data, load_product, save_product
from ptychodus.api.reconstruct import (
    PositionIndexFilter,
    ReconstructOutput,
    prepare_reconstruct_input,
)
from ptychodus.model.ptychi.task import (
    align_task_options_with_product,
    load_task_options,
    reconstruct_with_ptychi,
)

logger = logging.getLogger('ptychodus_reconstruct')


class ExitCode(IntEnum):
    OK = 0
    FAILED = 1
    USAGE = 2
    CANCELLED = 130


class _Run:
    """The mutable state of one reconstruction: event stream, cancel flag, clock.

    Held in a single object passed explicitly to everything that needs it, so
    the event stream a helper writes to is visible in its signature rather than
    reached for through module scope.
    """

    def __init__(self, events: IO[str]) -> None:
        self._events = events
        self._cancel = threading.Event()
        self._started_s = time.monotonic()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    def elapsed_s(self) -> float:
        return round(time.monotonic() - self._started_s, 3)

    def emit(self, event: str, **fields: Any) -> None:
        """Write one line of the machine-readable event stream.

        ``allow_nan=False`` keeps the stream to strict JSON. Python would
        otherwise emit bare ``Infinity`` / ``NaN`` literals, which parsers
        outside Python reject -- and a diverging reconstruction produces
        infinite losses readily. Non-finite values are normalized to ``null``
        before they get here.
        """
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

    def on_signal(self, signum: int, _frame: FrameType | None) -> None:
        if self._cancel.is_set():
            # Second signal: the run loop is wedged inside pty-chi. Restore the
            # default disposition and re-raise so the process actually dies.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return

        self._cancel.set()
        self.emit('cancelling', signal=signal.Signals(signum).name)


def _positive_int(text: str) -> int:
    value = int(text)

    if value < 1:
        raise argparse.ArgumentTypeError(f'"{text}" must be at least 1!')

    return value


def _reconstruct(run: _Run, args: argparse.Namespace, log_file: Path) -> ExitCode:
    product_output_file: Path = args.product_output

    try:
        task_options = load_task_options(sys.stdin.read())
    except (TypeError, ValueError) as exc:
        run.emit_exception(exc)
        logger.exception('Failed to parse pty-chi options from stdin.')
        return ExitCode.USAGE

    assembled_data = load_diffraction_data(args.diffraction_input, mmap_file=args.mmap_file)
    product = load_product(args.product_input)
    reconstruct_input = prepare_reconstruct_input(
        assembled_data,
        product,
        index_filter=PositionIndexFilter[args.index_filter.upper()],
    )
    task_options = align_task_options_with_product(task_options, reconstruct_input.product)
    task_options.check()

    num_epochs = int(task_options.reconstructor_options.num_epochs)
    reconstructor_token = task_options.reconstructor_options.get_reconstructor_type().value

    run.emit(
        'started',
        pid=os.getpid(),
        reconstructor=reconstructor_token,
        options_class=type(task_options).__name__,
        num_epochs=num_epochs,
        num_sync_epochs=args.num_sync_epochs,
        num_patterns=int(reconstruct_input.diffraction_patterns.shape[0]),
        product_output=str(product_output_file),
        log=str(log_file),
    )

    if run.is_cancelled:
        # Cancelled during the file load, before any epoch ran.
        run.emit('finished', epoch=0, cancelled=True, path=None, elapsed_s=run.elapsed_s())
        return ExitCode.CANCELLED

    last_output: ReconstructOutput | None = None

    for result in reconstruct_with_ptychi(reconstruct_input, task_options, args.num_sync_epochs):
        last_output = result

        losses = result.product.losses
        loss = float(losses[-1].value) if losses else None
        if loss is not None and not math.isfinite(loss):
            loss = None

        run.emit(
            'epoch',
            epoch=result.progress,
            num_epochs=num_epochs,
            loss=loss,
            elapsed_s=run.elapsed_s(),
        )

        checkpoint_file = (
            product_output_file.parent
            / f'{product_output_file.stem}.{result.progress:06d}{product_output_file.suffix}'
        )
        save_product(checkpoint_file, result.product)
        run.emit('checkpoint', epoch=result.progress, path=str(checkpoint_file))

        if run.is_cancelled:
            break

    is_cancelled = run.is_cancelled

    if last_output is None:
        if not is_cancelled:
            raise RuntimeError('Reconstruction produced no output!')

        run.emit('finished', epoch=0, cancelled=True, path=None, elapsed_s=run.elapsed_s())
        return ExitCode.CANCELLED

    save_product(product_output_file, last_output.product)
    run.emit(
        'finished',
        epoch=last_output.progress,
        cancelled=is_cancelled,
        path=str(product_output_file),
        elapsed_s=run.elapsed_s(),
    )
    return ExitCode.CANCELLED if is_cancelled else ExitCode.OK


def main() -> ExitCode:
    parser = argparse.ArgumentParser(
        description='Run one pty-chi reconstruction, reading options from stdin.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--diffraction-input',
        metavar='DIFFRACTION_INPUT_FILE',
        required=True,
        type=Path,
        help='Assembled diffraction data file.',
    )
    parser.add_argument(
        '--product-input',
        metavar='PRODUCT_INPUT_FILE',
        required=True,
        type=Path,
        help='Input product file.',
    )
    parser.add_argument(
        '--product-output',
        metavar='PRODUCT_OUTPUT_FILE',
        required=True,
        type=Path,
        help='Output product file. Checkpoints and the log sit beside it.',
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
    args = parser.parse_args()

    product_output_file: Path = args.product_output
    product_output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = product_output_file.parent / f'{product_output_file.stem}.log'

    # Point fds 1 and 2 at the log file, keeping a dup of the real stdout to
    # carry the event stream. Importing pty-chi is silent, but running it is
    # not: PtychographyTask emits progress bars and torch writes warnings.
    # Moving the file descriptors themselves -- rather than just reassigning
    # sys.stdout -- also captures writes from C extensions.
    events = os.fdopen(os.dup(1), 'w', buffering=1)

    with open(log_file, 'w') as handle:
        os.dup2(handle.fileno(), 1)
        os.dup2(handle.fileno(), 2)

    sys.stdout = os.fdopen(os.dup(1), 'w', buffering=1)
    sys.stderr = os.fdopen(os.dup(2), 'w', buffering=1)

    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    run = _Run(events)

    # Before anything that can block: a launcher may cancel during the file
    # load. Signals that arrive earlier still -- while this module's pty-chi
    # and torch imports are running, or during the argument parsing and the
    # output redirect above -- cannot be caught, and kill the process outright;
    # there is nothing computed to save at that point, so launchers should wait
    # for the ``started`` event before cancelling.
    signal.signal(signal.SIGINT, run.on_signal)
    signal.signal(signal.SIGTERM, run.on_signal)

    try:
        return _reconstruct(run, args, log_file)
    except Exception as exc:
        run.emit_exception(exc)
        logger.exception('Reconstruction failed.')
        return ExitCode.FAILED


if __name__ == '__main__':
    sys.exit(main())
