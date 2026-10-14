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
import os
import queue
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from ptychodus.api.exit_codes import ExitCode
from ptychodus.api.io import StandardFileLayout
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import FilterValuesStep
from ptychodus.cli._batch_lamni_settings import (
    BatchSettings,
    load_batch_settings,
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
from ptychodus.model.processing._subprocess_protocol import ChildError, run_subprocess
from ptychodus.model.ptychi.device import probe_devices_via_subprocess
from ptychodus.model.ptychi.task import dump_task_options
from ptychodus.plugins.aps31id_lamni._scan_table import (
    APS31IDEScanRecord,
    read_aps31ide_scan_table,
)

logger = logging.getLogger('reconstruct_batch_lamni')

DIFFRACTION_READER = 'APS_LamNI'

_WORKER_ENTRY = 'ptychodus.cli._batch_lamni_worker:reconstruct_scan'

# ``ptychi.list_available_devices()`` reports e.g. 'NVIDIA A100 80GB (cuda:3)'.
_CUDA_DEVICE_PATTERN = re.compile(r'cuda:(\d+)')


def visible_cuda_devices() -> list[str]:
    """The CUDA devices this run may use, as values for ``CUDA_VISIBLE_DEVICES``.

    Probed in a throwaway child, because asking the question in this process would leave a
    CUDA context behind, and a spawned child of a process holding one is a known source of
    initialization failures.

    The two sides of that mask count differently, which is the whole difficulty here. The
    probe inherits this process's ``CUDA_VISIBLE_DEVICES``, so the ordinals it reports are
    positions *within* that mask: under ``CUDA_VISIBLE_DEVICES=1`` it reports one device
    and calls it ``cuda:0``. But the value a worker sets is read against the physical
    devices, so handing it that ``0`` would send it to the GPU the scheduler did not give
    us -- quite possibly one already running someone else's job. The probe's ordinals are
    therefore indices back into the inherited list, and what each worker is handed is the
    entry found there. With nothing inherited the two numberings coincide and the ordinal
    is passed through. Entries are copied verbatim rather than parsed, so a mask written
    as GPU UUIDs composes the same way.
    """
    ordinals: list[int] = list()

    for description in probe_devices_via_subprocess():
        match = _CUDA_DEVICE_PATTERN.search(description)

        if match is not None:
            ordinals.append(int(match.group(1)))

    inherited = [
        entry.strip()
        for entry in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
        if entry.strip()
    ]

    if not inherited:
        return [str(ordinal) for ordinal in ordinals]

    return [inherited[ordinal] for ordinal in ordinals if ordinal < len(inherited)]


@dataclass
class _Tally:
    """What became of each selected scan, counted across the worker threads."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    succeeded: int = 0
    skipped: int = 0
    cancelled: int = 0
    failed: list[int] = field(default_factory=list)


def _worker_loop(
    settings: BatchSettings,
    work: queue.Queue[tuple[int, APS31IDEScanRecord]],
    device: str,
    *,
    settings_text: str,
    options_json: str | None,
    read_region: dict[str, tuple[int, int]] | None,
    num_selected: int,
    tally: _Tally,
    cancellation: CancellationToken,
) -> None:
    """Take scans off the shared queue until it empties or the run is cancelled.

    One of these runs per device, in a thread. Each spawns a worker process per scan and
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
        device_name = f'cuda:{device}' if device else 'cpu'
        payload: dict[str, object] = {
            'cuda_visible_devices': device,
            'settings_text': settings_text,
            'ptychi_options_json': options_json,
            'scan': record_payload(record),
            'read_region': read_region,
        }

        logger.info('%s scan %05d: starting on %s', position, record.scan_no, device_name)
        completed = False

        try:
            with run_subprocess(_WORKER_ENTRY, payload) as events:
                for event in events:
                    if event[0] == 'output':
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
                tally.succeeded += 1
            else:
                # A worker returns without a final product only when the cancellation it
                # installed for itself arrived first.
                tally.cancelled += 1

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


def _report_plan(settings: BatchSettings, selected: list[APS31IDEScanRecord]) -> None:
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
            tally.skipped += 1
            continue

        work.put((index, record))

    return work


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

    try:
        scan_table_file = settings.scan_table_file
    except ValueError as exc:
        logger.error('%s', exc)
        return ExitCode.USAGE

    logger.info('Reading the scan table from %s', scan_table_file)
    records = read_aps31ide_scan_table(scan_table_file)
    label = select_label(records, settings.batch.label)
    selected = select_records(records, settings.batch, label)

    if not selected:
        logger.error('No scans matched the requested scan range.')
        return ExitCode.FAILURE

    logger.info(
        'Selected %d scan(s): %05d..%05d',
        len(selected),
        selected[0].scan_no,
        selected[-1].scan_no,
    )

    if args.dry_run:
        _report_plan(settings, selected)
        logger.info('Dry run: %d scan(s) planned. Nothing read, nothing written.', len(selected))
        return ExitCode.SUCCESS

    options = load_ptychi_options(args.ptychi_options)

    if not options.probe_position_options.optimizable:
        logger.info(
            'pty-chi position correction is disabled, so no scan will have refined '
            'positions to fit a transform from, and none will write affine_fit.yaml.'
        )

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
    devices = visible_cuda_devices()

    if devices:
        logger.info(
            'Reconstructing %d scan(s) across %d CUDA device(s): %s',
            work.qsize(),
            len(devices),
            ', '.join(f'cuda:{device}' for device in devices),
        )
    else:
        # One worker, inheriting an empty CUDA_VISIBLE_DEVICES, so a device this process
        # could not see is not found by the child either.
        logger.warning(
            'No CUDA devices are visible. Reconstructing %d scan(s) one at a time on the '
            'CPU, which will be very slow. Set CUDA_VISIBLE_DEVICES, or ask the job '
            'scheduler for GPUs.',
            work.qsize(),
        )
        devices = ['']

    options_json = None if args.ptychi_options is None else dump_task_options(options)

    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for device in devices:
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

    logger.info(
        'Done: %d reconstructed, %d skipped, %d failed, %d cancelled, of %d selected.',
        tally.succeeded,
        tally.skipped,
        len(tally.failed),
        tally.cancelled,
        len(selected),
    )

    if tally.failed:
        logger.error(
            'Failed scans: %s', ', '.join(f'{scan_no:05d}' for scan_no in sorted(tally.failed))
        )

    if cancellation.is_cancelled:
        logger.warning('Cancelled: %s.', cancellation.reason or 'no reason given')
        return ExitCode.CANCELLED

    return ExitCode.FAILURE if tally.failed else ExitCode.SUCCESS


if __name__ == '__main__':
    sys.exit(main())
