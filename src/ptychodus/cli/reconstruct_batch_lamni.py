#!/usr/bin/env python
"""Reconstruct a range of APS 31-ID-E LamNI projections from one experiment directory.

A laminography tomogram is one scan per projection -- 1440 and up, split into interleaved
sub-tomograms -- and ``tomography_scannumbers.txt`` is the index that ties each scan number
to its rotation angle and its place in the tomogram. This driver walks that table and runs
a per-scan reconstruction over the selected projections, so a tomogram is one command
rather than a thousand.

Settings come from a YAML file and nowhere else. A run is therefore reproducible from a
file that can be archived, and the copy written into every scan's output directory records
how that scan was reconstructed. The command line carries only the two file names, what to
log, and whether to stop after planning.

Each scan lands in its own standard-layout directory under
``<output_products_folder>/S<NNNNN>/``: the assembled patterns, the resolved pty-chi
options, the settings, per-epoch ``product.NNNNNN.h5`` checkpoints, the final
``product.h5``, a log of everything the scan wrote to its console, and an affine fit of its
own positions. Every product carries its own ``tomography_angle_deg``, taken from the
encoder column of the scan table, which is what lets the resulting stack be fed to a
tomographic solver.

Scans run concurrently, one worker process per visible CUDA device, each pinned to its own.
The job scheduler decides how many that is: ``CUDA_VISIBLE_DEVICES`` is the only control,
and with no devices visible the run proceeds on the CPU, one scan at a time. Because each
worker is a separate process that exits when its scan does, GPU memory is reclaimed between
scans rather than accumulating across a tomogram.

Driving it from Python
----------------------

:func:`main` is a shim over :func:`reconstruct_batch_lamni`, which takes the settings and
the pty-chi options as objects. A caller that already holds a :class:`BatchSettings` --
built in a script, or loaded from a file and modified -- runs a batch without going through
a file and a subprocess, and reads what became of each scan off the returned
:class:`BatchResult` rather than off an exit code. :func:`plan_batch_lamni` resolves the
selection by itself, which is a dry run in code.

Neither function configures logging or installs signal handlers: both belong to whoever
owns the process, and a caller that wants its run interruptible passes in its own
:class:`~ptychodus.cli._reconstruct_common.CancellationToken`. The fan-out spawns a child
per scan, so the calling process must not already hold a CUDA context -- the same
constraint that makes device enumeration happen in a throwaway child.

Per-scan affine estimate
------------------------

Every scan whose positions pty-chi refined gets an ``affine_fit.yaml`` fitted from that
scan alone. A single-scan fit is the noisy one -- the estimator exists to pool several
scans against one shared linear part -- so these are the per-scan inputs to a calibration
computed elsewhere, not a calibration apiece. A transform already known can be applied to
the measured positions before reconstruction instead, through ``positions.affine_matrix``.

Scope and limitations
---------------------

- One tomogram per run, selected by label.
- The crop region is resolved once, from the first selected scan, and reused for every
  scan. A per-scan beam center would put each projection on a different grid, and the
  stack is only a tomogram if the projections share one.
- Repeated acquisitions of one ``(sub-tomogram, projection)`` are all reconstructed. The
  scan table records a re-run as a further row, and which one supersedes the other is not
  something the table says.
- A scan number inside the selected range but absent from the table is not reconstructed,
  because the table is where its rotation angle would have come from.
"""

from __future__ import annotations

import argparse
import logging
import queue
import sys
import threading
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ptychi.api.options.task import PtychographyTaskOptions

from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import StandardFileLayout
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import FilterValuesStep
from ptychodus.cli._batch_lamni_settings import (
    Batch,
    BatchSettings,
    Diffraction,
    InitialObject,
    InitialProbe,
    MetadataOverrides,
    Misc,
    ObjectFromFile,
    Paths,
    Positions,
    ProbeFromFile,
    SimulatedProbe,
    WorkerMessage,
    dump_batch_settings,
    load_batch_settings,
    load_batch_settings_from_text,
    record_payload,
    select_label,
    select_records,
)
from ptychodus.cli._reconstruct_common import (
    CancellationToken,
    add_print_ptychi_options_argument,
    add_ptychi_options_argument,
    install_signal_handlers,
    invalid_count_threshold,
    load_ptychi_options,
    resolve_crop_region,
)
from ptychodus.api.devices import CPU, Device, probe_devices, visibility_environment
from ptychodus.api.spawn import ChildError, run_subprocess
from ptychodus.plugins.ptychi.task import dump_task_options
from ptychodus.plugins.aps31id_lamni.scan_table import (
    APS31IDEScanRecord,
    read_aps31ide_scan_table,
)

# The settings sections are re-exported, so building a run in Python needs one import
# rather than a reach into the module the three processes happen to share.
__all__ = [
    'Batch',
    'BatchResult',
    'BatchSettings',
    'Diffraction',
    'InitialObject',
    'InitialProbe',
    'MetadataOverrides',
    'Misc',
    'ObjectFromFile',
    'Paths',
    'Positions',
    'ProbeFromFile',
    'SimulatedProbe',
    'dump_batch_settings',
    'load_batch_settings',
    'load_batch_settings_from_text',
    'main',
    'plan_batch_lamni',
    'reconstruct_batch_lamni',
    'report_plan',
    'visible_accelerators',
]

logger = logging.getLogger('reconstruct_batch_lamni')

DIFFRACTION_READER = 'APS_LamNI'

_WORKER_ENTRY = 'ptychodus.cli._batch_lamni_worker:reconstruct_scan'


def visible_accelerators() -> list[Device]:
    """The accelerators this run may fan out across, one worker each.

    Probed in a throwaway child by the shared device layer, which also owns the
    arithmetic that makes an inherited mask compose correctly. Empty when this
    process was given none.
    """
    return [device for device in probe_devices() if device != CPU]


@dataclass(frozen=True)
class BatchResult:
    """What became of every scan a run selected.

    Scan numbers rather than counts, because a caller driving the batch from Python is the
    one deciding what a partial tomogram means, and that decision needs to know which
    projections are missing from it. Every selected scan appears in exactly one of the four
    outcome fields.

    `cancellation_reason` is None unless the run stopped early, so it is also how a caller
    tells an empty `failed` that means success from one that means the run never got there.
    """

    selected: tuple[int, ...]
    succeeded: tuple[int, ...]
    skipped: tuple[int, ...]
    failed: tuple[int, ...]
    cancelled: tuple[int, ...]
    cancellation_reason: str | None = None

    @property
    def exit_code(self) -> ExitCode:
        """The status the console command exits with for this result."""
        if self.cancellation_reason is not None:
            return ExitCode.CANCELLED

        return ExitCode.FAILURE if self.failed else ExitCode.SUCCESS


@dataclass
class _Tally:
    """What became of each selected scan, gathered across the worker threads."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    succeeded: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    cancelled: list[int] = field(default_factory=list)
    failed: list[int] = field(default_factory=list)


def _worker_loop(
    settings: BatchSettings,
    work: queue.Queue[tuple[int, APS31IDEScanRecord]],
    environment: Mapping[str, str],
    *,
    settings_text: str,
    options_json: str | None,
    read_region: dict[str, tuple[int, int]] | None,
    num_selected: int,
    tally: _Tally,
    cancellation: CancellationToken,
) -> None:
    """Take scans off the shared queue until it empties or the run is cancelled.

    One of these runs per device, in a thread. Each spawns a worker process per scan,
    restricted to that device by the environment the spawn layer applies, and
    waits on its events; the thread never does work of its own, so a thread apiece costs
    nothing against the reconstruction it is waiting for.

    The worker writes every artifact itself, into that scan's own directory, so nothing of
    substance comes back. The child's log records are dispatched into this process's
    logging tree by the subprocess protocol, and all that is awaited here is how it ended.
    """
    while not cancellation.is_cancelled:
        try:
            index, record = work.get_nowait()
        except queue.Empty:
            return

        position = f'[{index}/{num_selected}]'
        device_name = ', '.join(f'{k}={v}' for k, v in environment.items()) or 'inherited'
        payload: dict[str, object] = {
            'settings_text': settings_text,
            'ptychi_options_json': options_json,
            'scan': record_payload(record),
            'read_region': read_region,
        }

        logger.info('%s scan %05d: starting on %s', position, record.scan_no, device_name)
        completed = False

        try:
            with run_subprocess(_WORKER_ENTRY, payload, environment=environment) as events:
                for event in events:
                    if event[0] == WorkerMessage.SCAN_RESULT:
                        completed = bool(event[1].get('completed'))
        except ChildError as exc:
            with tally.lock:
                tally.failed.append(record.scan_no)

            logger.error(
                '%s scan %05d failed: %s\n%s',
                position,
                record.scan_no,
                exc.child_exception_type,
                exc.child_traceback,
            )

            if not settings.misc.continue_on_error:
                cancellation.cancel('a scan failed')
                return

            continue

        with tally.lock:
            if completed:
                tally.succeeded.append(record.scan_no)
            else:
                # A worker returns without a final product only when the cancellation it
                # installed for itself arrived first.
                tally.cancelled.append(record.scan_no)

        logger.info(
            '%s scan %05d: %s',
            position,
            record.scan_no,
            'done' if completed else 'cancelled',
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Reconstruct a range of APS 31-ID-E LamNI projections from one experiment '
            'directory, as described by a settings file.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--batch-settings',
        type=Path,
        required=True,
        help='YAML settings file: the paths, the scans, and every tuning value.',
    )
    add_ptychi_options_argument(parser)
    add_print_ptychi_options_argument(parser)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Report the resolved scan plan and stop without reading or reconstructing data.',
    )
    parser.add_argument(
        '--log-level',
        default=None,
        type=int,
        help='Python logging level, overriding misc.log_level in the settings file.',
    )
    return parser


def plan_batch_lamni(settings: BatchSettings) -> list[APS31IDEScanRecord]:
    """The scans a run over `settings` would reconstruct, in acquisition order.

    Reads the scan table and nothing else, so this is the whole of what a dry run resolves
    and it touches no detector data. Raises `ValueError` when the settings name no 31-ID-E
    experiment directory, when the requested label is absent from the table, or when the
    scan range selects nothing.
    """
    scan_table_file = settings.scan_table_file
    logger.info('Reading the scan table from %s', scan_table_file)
    records = read_aps31ide_scan_table(scan_table_file)
    label = select_label(records, settings.batch.label)
    selected = select_records(records, settings.batch, label)

    if not selected:
        raise ValueError('No scans matched the requested scan range.')

    logger.info(
        'Selected %d scan(s): %05d..%05d',
        len(selected),
        selected[0].scan_no,
        selected[-1].scan_no,
    )
    return selected


def report_plan(settings: BatchSettings, selected: list[APS31IDEScanRecord]) -> None:
    """Log each selected scan with the files it would read and the directory it would write."""
    for record in selected:
        try:
            diffraction_file_path: Path | str = settings.diffraction_file(record.scan_no)
        except FileNotFoundError as exc:
            diffraction_file_path = f'MISSING ({exc})'

        position_file_path = settings.position_file(record.scan_no)
        position_note = '' if position_file_path.is_file() else ' MISSING'
        logger.info(
            'scan %05d subtomo %d projection %d angle %+.4f deg\n'
            '  diffraction: %s\n'
            '  positions:   %s%s\n'
            '  output:      %s',
            record.scan_no,
            record.subtomo_no,
            record.measurement_id,
            record.encoder_angle_deg,
            diffraction_file_path,
            position_file_path,
            position_note,
            settings.output_directory(record.scan_no),
        )


def _queue_pending_scans(
    settings: BatchSettings, selected: list[APS31IDEScanRecord], tally: _Tally
) -> queue.Queue[tuple[int, APS31IDEScanRecord]]:
    """Enqueue every scan that still needs reconstructing, counting the rest as skipped."""
    work: queue.Queue[tuple[int, APS31IDEScanRecord]] = queue.Queue()

    for index, record in enumerate(selected, start=1):
        product_file = StandardFileLayout.PRODUCT.path(settings.output_directory(record.scan_no))

        if not settings.batch.overwrite_existing_products and product_file.is_file():
            logger.info('scan %05d: skipping, %s already exists', record.scan_no, product_file)
            tally.skipped.append(record.scan_no)
            continue

        work.put((index, record))

    return work


def reconstruct_batch_lamni(
    settings: BatchSettings,
    options: PtychographyTaskOptions | None = None,
    *,
    settings_text: str | None = None,
    cancellation: CancellationToken | None = None,
) -> BatchResult:
    """Reconstruct every scan `settings` selects, and report what became of each.

    Blocks until the batch finishes. Each scan is reconstructed in a child process of its
    own and writes its own standard-layout directory, so nothing of a reconstruction comes
    back here; what is returned is the account of which scans got one.

    A scan that fails is counted rather than raised, because a tomogram with a projection
    missing is a result its owner has to judge. Everything that goes wrong *before* any
    scan starts -- an experiment directory this machine cannot see, an empty selection, an
    unreadable bad-pixel mask or initial probe, a crop that runs off the detector -- raises
    instead, since it describes every scan equally.

    `options` defaults to stock LSQML, resolved in each worker rather than here, which is
    what ``--ptychi-options`` does when it is not given. An options object is serialized
    once and rebuilt in every worker, so per-scan edits cannot leak between scans.

    `settings_text` is the YAML copied into each scan's output directory and rebuilt from
    in each worker. It defaults to a rendering of `settings`; a caller that loaded a file
    passes its text instead, so the archived copy keeps the comments and spelling of the
    file someone wrote.

    `cancellation` stops the run at the first scan boundary after the request, leaving the
    scans already written in place. One is created if none is given, so an uninterruptible
    run needs no argument; installing the signal handlers that set it belongs to whoever
    owns the process.
    """
    selected = plan_batch_lamni(settings)
    resolved_options = (
        load_ptychi_options(None, far_field=settings.metadata_overrides.far_field is not False)
        if options is None
        else options
    )

    if not resolved_options.probe_position_options.optimizable:
        logger.info(
            'pty-chi position correction is disabled, so no scan will have refined '
            'positions to fit a transform from, and none will write affine_fit.yaml.'
        )

    if cancellation is None:
        cancellation = CancellationToken()

    if settings_text is None:
        settings_text = dump_batch_settings(settings)

    registry = PluginRegistry.load_plugins()
    bad_pixels = None

    if settings.diffraction.bad_pixels_file is not None:
        # Read here so a bad path fails the run now rather than once per scan. Each worker
        # reads the mask itself, which keeps an array of this size off the payload.
        logger.info('Reading bad pixels from %s', settings.diffraction.bad_pixels_file)
        bad_pixels = registry.bad_pixels_file_readers.get_strategy_by_name(
            settings.diffraction.bad_pixels_file_type
        ).read(settings.diffraction.bad_pixels_file)

    if settings.initial_probe.probe_source == 'file':
        probe_file = settings.initial_probe.probe_from_file.file_path
        assert probe_file is not None  # the settings model requires it for "file"
        logger.info('Checking the initial probe file %s', probe_file)
        registry.probe_file_readers.get_strategy_by_name(
            settings.initial_probe.probe_from_file.file_type
        ).read(probe_file)

    # One crop for the whole run, settled from the first selected scan.
    first_dataset = registry.diffraction_file_readers.get_strategy_by_name(DIFFRACTION_READER).read(
        settings.diffraction_file(selected[0].scan_no)
    )
    first_metadata = first_dataset.get_metadata()
    max_valid_count = (
        invalid_count_threshold(first_metadata.pattern_dtype)
        if settings.diffraction.max_valid_count is None
        else settings.diffraction.max_valid_count
    )

    # The same cut each scan's pipeline applies, so the frame the beam center is estimated
    # from matches the frames that will be reconstructed.
    value_filter: FilterValuesStep | None = None

    if max_valid_count is not None:
        logger.info('Zeroing pixels at or above %d as invalid', max_valid_count)
        value_filter = FilterValuesStep(lower_bound=0, upper_bound=max_valid_count)

    crop_region = resolve_crop_region(
        logger,
        first_dataset,
        crop_extent_px=settings.diffraction.crop_extent_px,
        beam_center_x_px=settings.diffraction.beam_center_x_px,
        beam_center_y_px=settings.diffraction.beam_center_y_px,
        bad_pixels=bad_pixels,
        value_filter=value_filter,
        subject='first scan',
    )
    del first_dataset, bad_pixels

    read_region = (
        None
        if crop_region is None
        else {'x_range': crop_region.x_range, 'y_range': crop_region.y_range}
    )

    tally = _Tally()
    work = _queue_pending_scans(settings, selected, tally)
    accelerators = visible_accelerators()
    available = [device.identifier for device in probe_devices()]

    if accelerators:
        logger.info(
            'Reconstructing %d scan(s) across %d device(s): %s',
            work.qsize(),
            len(accelerators),
            ', '.join(device.identifier for device in accelerators),
        )
        environments = [
            visibility_environment([device.identifier], available) for device in accelerators
        ]
    else:
        logger.warning(
            'No accelerators are visible. Reconstructing %d scan(s) one at a time on the '
            'CPU, which will be very slow. Set CUDA_VISIBLE_DEVICES, or ask the job '
            'scheduler for GPUs.',
            work.qsize(),
        )
        # One worker, inheriting whatever this process has, so a device it could
        # not see is not found by the child either.
        environments = [{}]

    options_json = None if options is None else dump_task_options(options)
    futures: list[Future[None]] = list()

    with ThreadPoolExecutor(max_workers=len(environments)) as pool:
        for device in environments:
            futures.append(
                pool.submit(
                    _worker_loop,
                    settings,
                    work,
                    device,
                    settings_text=settings_text,
                    options_json=options_json,
                    read_region=read_region,
                    num_selected=len(selected),
                    tally=tally,
                    cancellation=cancellation,
                )
            )

    for future in futures:
        exception = future.exception()

        if exception is not None:
            # A scan that fails raises ChildError, which the loop handles and counts. An
            # exception that reaches here is therefore a fault in this process, and the
            # executor would otherwise swallow it and leave the summary below reporting a
            # clean run over scans nothing ever attempted.
            raise RuntimeError('A worker thread failed; the batch is incomplete.') from exception

    logger.info(
        'Done: %d reconstructed, %d skipped, %d failed, %d cancelled, of %d selected.',
        len(tally.succeeded),
        len(tally.skipped),
        len(tally.failed),
        len(tally.cancelled),
        len(selected),
    )

    if tally.failed:
        logger.error(
            'Failed scans: %s', ', '.join(f'{scan_no:05d}' for scan_no in sorted(tally.failed))
        )

    cancellation_reason = None

    if cancellation.is_cancelled:
        cancellation_reason = cancellation.reason or 'no reason given'
        logger.warning('Cancelled: %s.', cancellation_reason)

    return BatchResult(
        selected=tuple(record.scan_no for record in selected),
        succeeded=tuple(sorted(tally.succeeded)),
        skipped=tuple(sorted(tally.skipped)),
        failed=tuple(sorted(tally.failed)),
        cancelled=tuple(sorted(tally.cancelled)),
        cancellation_reason=cancellation_reason,
    )


def main() -> ExitCode:
    args = _build_parser().parse_args()

    try:
        settings = load_batch_settings(args.batch_settings)
    except (OSError, ValueError) as exc:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        logger.error('%s', exc)
        return ExitCode.USAGE

    log_level = settings.misc.log_level if args.log_level is None else args.log_level
    logging.basicConfig(
        level=log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    settings_text = args.batch_settings.read_text()

    # Installed before the first long operation, so a cancel during the file reads is
    # already honored. Nothing earlier than this is interruptible.
    cancellation = install_signal_handlers()

    if args.dry_run:
        try:
            selected = plan_batch_lamni(settings)
        except (OSError, ValueError) as exc:
            logger.error('%s', exc)
            return ExitCode.USAGE

        report_plan(settings, selected)
        logger.info('Dry run: %d scan(s) planned. Nothing read, nothing written.', len(selected))
        return ExitCode.SUCCESS

    # None rather than the stock options this would build from None: the workers resolve
    # that default themselves, so nothing is serialized for a run that asked for nothing.
    options = None if args.ptychi_options is None else load_ptychi_options(args.ptychi_options)

    try:
        result = reconstruct_batch_lamni(
            settings,
            options,
            settings_text=settings_text,
            cancellation=cancellation,
        )
    except (OSError, ValueError) as exc:
        # Everything the run resolves before the first scan starts, reported as the one
        # line it is rather than as a traceback: each of these says the settings cannot be
        # carried out, and none of them has attempted any work.
        logger.error('%s', exc)
        return ExitCode.USAGE

    return result.exit_code


if __name__ == '__main__':
    sys.exit(main())
