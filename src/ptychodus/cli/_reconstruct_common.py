"""Shared reconstruction tail for the ``cli/reconstruct_*.py`` drivers.

Every driver assembles diffraction data and builds an initial
:class:`~ptychodus.api.product.Product` -- that part differs per instrument. From there on, every
driver resolves its pty-chi options, saves ``diffraction.h5``, aligns those options against the
product, runs the reconstruction to convergence, and writes the standard-layout outputs
identically. :func:`load_ptychi_options`, :func:`save_assembled_diffraction` and
:func:`run_reconstruction` are that common tail, extracted so it is written and tested once
rather than duplicated in eleven files.

Both writers write under :func:`is_main_process`, so a multi-process launch produces one copy of
each artifact instead of having every rank truncate the same paths concurrently.

Cancellation lives here for the same reason. A reconstruction has no interruption point inside
pty-chi, so a signal cannot stop it where it lands; :func:`install_signal_handlers` instead
raises a flag that :func:`run_reconstruction` reads at each sync point, which costs at most
``num_sync_epochs`` epochs and leaves a partially converged ``product.h5`` behind. Under a
multi-process launch that flag is agreed across ranks before it is acted on, because a rank that
left the loop alone would hang its peers at their next collective.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from types import FrameType

from ptychi.api import LSQMLOptions
from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.io import StandardFileLayout, save_diffraction_data, save_product
from ptychodus.api.product import Product
from ptychodus.api.reconstruct import ReconstructInput, ReconstructOutput
from ptychodus.model.ptychi.task import (
    align_task_options_with_product,
    dump_task_options,
    load_task_options,
    reconstruct_with_ptychi,
)

__all__ = [
    'EXIT_CANCELLED',
    'STDIN_ARGUMENT',
    'CancellationToken',
    'add_ptychi_options_argument',
    'install_signal_handlers',
    'is_main_process',
    'load_ptychi_options',
    'process_rank',
    'run_reconstruction',
    'save_assembled_diffraction',
]

# Exit status for a run that stopped because it was asked to, by the shell convention of
# 128 + SIGINT. Defined once so eleven drivers can return it without each spelling out why.
EXIT_CANCELLED = 130

# The conventional "read it from stdin instead" filename, as argparse hands it over once a
# bare ``-`` has been through ``type=Path``.
STDIN_ARGUMENT = Path('-')

# Environment variables that report this process's rank, in the order they are consulted.
# `torchrun` and `torch.distributed.launch` export RANK; plain `srun` exports SLURM_PROCID
# and none of the torch ones.
_RANK_ENVIRONMENT_VARIABLES = ('RANK', 'SLURM_PROCID')


def _environment_rank() -> int:
    """Rank as the launcher reported it, or 0 when no launcher did.

    A value that is empty or not an integer is ignored rather than fatal: it means
    something in the environment claims a rank it cannot state, and declining to write
    the run's output is the worse of the two readings.
    """
    for name in _RANK_ENVIRONMENT_VARIABLES:
        try:
            return int(os.environ[name])
        except (KeyError, ValueError):
            continue

    return 0


def process_rank() -> int:
    """This process's rank, as either the process group or the launcher reports it.

    Two independent sources are consulted because neither alone covers the whole run. The
    torch process group does not exist until the reconstruction task is built, so until
    then every process reports rank 0 and only the environment tells them apart; a group
    created programmatically rather than by a launcher, in turn, exports nothing. A run
    under neither -- an ordinary single-process invocation -- is rank 0 by both, which is
    what makes this invisible outside a multi-process launch.
    """
    # Resolved at call time because ``ptychi.parallel`` cannot be the first ptychi module
    # a process imports: it reaches ``ptychi.io_handles``, which imports back from it. The
    # module-scope ``ptychi.api.options.task`` import above is what breaks that cycle, so
    # by the time this runs the name is already in sys.modules.
    from ptychi.parallel import get_rank

    return get_rank() or _environment_rank()


def is_main_process() -> bool:
    """Whether this process is the one that writes a run's output files.

    Every rank of a multi-process reconstruction runs the same work and ends up holding
    the same product, so each output path would otherwise be written concurrently by all
    of them, every write truncating.
    """
    return process_rank() == 0


class CancellationToken:
    """A request to stop, raised asynchronously and read at the next safe point.

    The flag is a :class:`threading.Event` so that setting it from a signal handler does no
    real work. Nothing is logged there either: the logging machinery takes a lock, and a
    signal delivered while that lock is held would deadlock the process it was meant to stop.
    `on_cancel` exists for a caller that must react the instant the request arrives rather
    than at the next sync point, and runs under the same constraint.
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


def _agree_on_cancellation(token: CancellationToken | None) -> bool:
    """Whether every rank stops after this sync point, or none of them does.

    Each rank catches its own signals, and a launcher does not deliver them to every rank at
    the same instant, so two ranks can read their own flags on opposite sides of one sync
    point. The one that left the loop would then never reach the next collective and the
    others would block there forever -- a cancellation that hangs the job instead of ending
    it. Reducing the flag with MAX makes one rank's request everyone's.

    The reduction is itself a collective, so it is only correct because every rank reaches
    this point the same number of times: ``reconstruct_with_ptychi`` yields in lockstep. With
    no process group -- an ordinary single-process run, and every rank before the task is
    built -- there is nobody to agree with and the local flag stands.
    """
    if token is None:
        return False

    is_cancelled = token.is_cancelled

    # Imported here rather than at module scope, as everywhere else in this package that
    # touches torch: the import is cheap only because ptychi has already pulled it in.
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        return is_cancelled

    # NCCL reduces on the GPU and raises on a CPU tensor, so the device follows the backend.
    device = (
        torch.device('cuda', torch.cuda.current_device())
        if dist.get_backend() == 'nccl'
        else torch.device('cpu')
    )
    flag = torch.tensor([int(is_cancelled)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag[0])


_PTYCHI_OPTIONS_HELP = (
    'pty-chi options JSON, as written to ptychi_options.json by any ptychodus run. Its '
    'algorithm stamp selects the reconstructor. "-" reads the JSON from stdin.'
)


def add_ptychi_options_argument(
    parser: argparse.ArgumentParser, *, default_note: str = 'Defaults to stock LSQML.'
) -> None:
    """Add ``--ptychi-options-file``, so its help text is written once rather than ten times.

    `default_note` completes the help for a driver that resolves a different fallback than
    the stock options :func:`load_ptychi_options` builds from ``None``.
    """
    parser.add_argument(
        '--ptychi-options-file',
        type=Path,
        default=None,
        help=f'{_PTYCHI_OPTIONS_HELP} {default_note}',
    )


def load_ptychi_options(options_file: Path | None) -> PtychographyTaskOptions:
    """Build a fresh options object, so per-scan edits never leak into the next scan."""
    if options_file is None:
        return LSQMLOptions()

    if options_file == STDIN_ARGUMENT:
        return load_task_options(sys.stdin.read())

    return load_task_options(options_file.read_text())


def save_assembled_diffraction(
    logger: logging.Logger,
    output_directory: Path,
    assembled_data: AssembledDiffractionData,
    *,
    skip: bool,
) -> None:
    """Create `output_directory` and write the assembled patterns to ``diffraction.h5``.

    Called ahead of the reconstruction, so an interrupted run still leaves the directory
    usable: the assembled patterns are the one artifact that cannot be rebuilt without the
    raw beamline files. Pass `skip` to create the directory without writing them, for a
    run whose patterns would be larger than they are worth keeping.

    The directory is created on every rank, since every rank needs somewhere to work; only
    the write is confined to the main process.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    diffraction_file = StandardFileLayout.DIFFRACTION.path(output_directory)

    if skip:
        logger.info('Skipping %s as requested', diffraction_file.name)
    elif is_main_process():
        logger.info('Writing %s', diffraction_file)
        save_diffraction_data(diffraction_file, assembled_data)


def run_reconstruction(
    logger: logging.Logger,
    reconstruct_input: ReconstructInput,
    options: PtychographyTaskOptions,
    output_directory: Path,
    *,
    num_sync_epochs: int,
    cancellation: CancellationToken | None = None,
    on_sync: Callable[[ReconstructOutput, Path | None], None] | None = None,
) -> Product | None:
    """Align `options` against the product, run pty-chi to completion, and write the outputs.

    Assumes `output_directory` already exists, and does not touch `diffraction.h5`: that
    artifact depends on the assembled dataset rather than on the reconstruction, so it is
    written by :func:`save_assembled_diffraction` before this runs. Aligns
    `options` against `reconstruct_input.product`, writes the aligned options to
    `ptychi_options.json`, runs `reconstruct_with_ptychi` to completion writing a
    `product.NNNNNN.h5` checkpoint at every sync point, then writes the final `product.h5`.
    Returns the final reconstructed product.

    Output files are written once, by the main process. Only the filesystem writes are
    gated: every rank runs the reconstruction to completion and returns the same product,
    because the reconstructed parameters are broadcast from rank 0 at each epoch boundary.

    With a `cancellation` token, the loop stops at the first sync point after the request
    and still writes `product.h5` from the epochs that did run -- a partially converged
    product is the whole reason to stop politely rather than kill the process. The return is
    then `None` only when the request arrived before any epoch completed, since there is no
    product in that case; an *uncancelled* run that produces nothing is still an error.

    `on_sync` is called at each sync point with the output and the checkpoint path that was
    written for it, which is `None` on a rank that writes nothing. It is how a caller
    reports progress without reimplementing this loop.
    """
    main_process = is_main_process()

    options_file = StandardFileLayout.PTYCHI_OPTIONS.path(output_directory)
    product_file = StandardFileLayout.PRODUCT.path(output_directory)

    task_options = align_task_options_with_product(options, reconstruct_input.product)
    task_options.check()

    if main_process:
        # The aligned options are the ones that ran: they carry the object pixel size, the
        # wavelength and the slice spacings the product supplied, which the unaligned object
        # does not.
        logger.info('Writing %s', options_file)
        options_file.write_text(dump_task_options(task_options))

    num_epochs = int(task_options.reconstructor_options.num_epochs)
    logger.info(
        'Starting reconstruction: %d patterns, %d epochs total, sync every %d',
        reconstruct_input.diffraction_patterns.shape[0],
        num_epochs,
        num_sync_epochs,
    )

    if _agree_on_cancellation(cancellation):
        # Asked to stop while the options were being aligned. The first chunk of epochs
        # would otherwise run to completion before the loop below could look at the flag.
        logger.warning('Cancelled before the first epoch; nothing was reconstructed.')
        return None

    final_output = None

    for output in reconstruct_with_ptychi(
        reconstruct_input, task_options, num_sync_epochs=num_sync_epochs
    ):
        losses = output.product.losses
        last_loss = losses[-1].value if losses else float('nan')
        logger.info('Epoch %d/%d: loss=%.6g', output.progress, num_epochs, last_loss)

        checkpoint_file: Path | None = None

        if main_process:
            checkpoint_file = StandardFileLayout.PRODUCT.checkpoint_path(
                output_directory, output.progress
            )
            save_product(checkpoint_file, output.product)

        if on_sync is not None:
            on_sync(output, checkpoint_file)

        final_output = output

        if _agree_on_cancellation(cancellation):
            logger.warning('Cancelled after epoch %d of %d.', output.progress, num_epochs)
            break

    if final_output is None:
        raise RuntimeError('Reconstruction produced no output.')

    if main_process:
        save_product(product_file, final_output.product)
        logger.info('Saved reconstructed product to %s', product_file)

    return final_output.product
