#!/usr/bin/env python
"""Replay a recorded scan to ``ptychodus-stream``, playing the beamline's part.

Reads a diffraction file and a position file with the reader plugins named on the
command line, and serves them as a live scan would arrive: detector frames as NTNDArrays
on the input channel, positions in the configured schema on the position channel, and
the trigger, scan-number and tomography-angle PVs of a scan engine. It reads the same
configuration file as ``ptychodus-stream``, so channel names, axis names, scales and the
index offset agree by construction, and it is the reference producer for the PV
contract in the streaming guide.

Start ``ptychodus-stream`` first: a consumer that connects mid-replay discards the frame
it connects on, as pvapy does with the value a monitor connects with.

Run from a checkout with the ``stream`` extra installed::

    python scripts/replay_scan.py --config scan.yaml \\
        --diffraction-file fly001_master.h5 --diffraction-file-type APS_Velociprobe \\
        --position-file fly001_pos.csv --position-file-type CSV
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import argparse
import logging
import sys
import time

import numpy
import pvaccess
from pvapy.utility.adImageUtility import AdImageUtility

from ptychodus.api.diffraction import DiffractionDataset
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.cli import add_log_level_argument, configure_logging
from ptychodus.cli._stream import (
    PositionSchema,
    PvProtocol,
    StreamConfiguration,
    pack_batched_positions,
    pack_sgsocket_attributes,
)
from ptychodus.cli.stream import (
    TRIGGER_TYPE,
    make_batched_positions_type,
    make_sgsocket_event,
)

logger = logging.getLogger('replay_scan')


@dataclass(frozen=True)
class _Point:
    event_id: int
    x_m: float
    y_m: float
    i0: float | None
    due_s: float


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Replay a recorded scan to ptychodus-stream, as detector, positioner and '
        'scan engine.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=Path, help='The ptychodus-stream configuration.')
    parser.add_argument('--diffraction-file', type=Path)
    parser.add_argument('--diffraction-file-type', help='Diffraction reader plugin name.')
    parser.add_argument('--position-file', type=Path)
    parser.add_argument('--position-file-type', help='Position reader plugin name.')
    parser.add_argument(
        '--list-file-types', action='store_true', help='Print the reader names and exit.'
    )
    parser.add_argument('--fps', type=float, default=100.0, help='Frames per second.')
    parser.add_argument(
        '--position-batch-size',
        type=int,
        default=100,
        help='Points per batch (batched schema); the last batch of a scan may be smaller.',
    )
    parser.add_argument(
        '--position-delay-s',
        type=float,
        default=0.1,
        help="How long a point's position trails its frame, as a positioner's would.",
    )
    parser.add_argument('--repeat', type=int, default=1, help='Scans to replay back to back.')
    parser.add_argument('--gap-s', type=float, default=5.0, help='Seconds between scans.')
    parser.add_argument(
        '--restart-indexes',
        action='store_true',
        help='Restart frame and event indexes each scan, rather than continuing them.',
    )
    parser.add_argument(
        '--free-run-s',
        type=float,
        default=0.0,
        help='Seconds of frames without positions to send between scans, as a free-running '
        'detector would.',
    )
    parser.add_argument(
        '--angle-step-deg',
        type=float,
        default=1.0,
        help='Tomography angle step between scans, when the configuration names an angle PV.',
    )
    add_log_level_argument(parser)
    return parser


def _list_file_types(registry: PluginRegistry) -> None:
    print('Diffraction file types:', registry.diffraction_file_readers.stringify_plugin_names())
    print('Position file types:', registry.probe_position_file_readers.stringify_plugin_names())


def _frames(dataset: DiffractionDataset) -> Iterator[tuple[int, numpy.ndarray]]:
    """Every pattern of the dataset with its index, read one array at a time."""
    for array in dataset:
        for index, pattern in zip(array.get_indexes(), array.get_patterns()):
            yield int(index), pattern


class _Replayer:
    def __init__(
        self,
        configuration: StreamConfiguration,
        args: argparse.Namespace,
        dataset: DiffractionDataset,
        positions: ProbePositionSequence,
    ) -> None:
        self._configuration = configuration
        self._args = args
        self._dataset = dataset
        self._positions = positions
        self._server = pvaccess.PvaServer()
        self._pending: deque[_Point] = deque()
        self._acknowledged = False
        self._position_axes = [configuration.x_axis.name, configuration.y_axis.name]

        if configuration.i0_name is not None:
            self._position_axes.append(configuration.i0_name)

        placeholder = numpy.zeros((1, 1), dtype=numpy.uint16)
        self._server.addRecord(
            configuration.input_channel, AdImageUtility.generateNtNdArray2D(0, placeholder)
        )

        if configuration.position_schema is PositionSchema.SGSOCKET:
            self._server.addRecord(configuration.position_channel, pvaccess.NtNdArray())
        else:
            self._server.addRecord(
                configuration.position_channel,
                pvaccess.PvObject(make_batched_positions_type(self._position_axes)),
            )

        self._server.addRecord(
            configuration.trigger_pv, pvaccess.PvObject(TRIGGER_TYPE), self._on_trigger_write
        )

        if configuration.scan_number_pv is not None:
            self._server.addRecord(
                configuration.scan_number_pv, pvaccess.PvObject({'value': pvaccess.INT})
            )

        if configuration.tomography_angle_pv is not None:
            self._server.addRecord(
                configuration.tomography_angle_pv, pvaccess.PvObject({'value': pvaccess.DOUBLE})
            )

    def _on_trigger_write(self, pv: Any) -> None:
        if not pv['value']:
            self._acknowledged = True

    def run(self) -> None:
        args = self._args
        indexes = numpy.asarray(self._positions.get_indexes())
        stride = int(indexes.max()) + 1 if indexes.size > 0 else 0

        for scan in range(args.repeat):
            offset = 0 if args.restart_indexes else scan * stride
            self._announce(scan)
            self._replay_scan(offset)
            self._trigger(scan)

            if scan + 1 < args.repeat:
                if args.free_run_s > 0.0:
                    self._free_run(offset + stride)

                time.sleep(args.gap_s)

    def _announce(self, scan: int) -> None:
        configuration = self._configuration

        if configuration.scan_number_pv is not None:
            self._server.update(
                configuration.scan_number_pv,
                pvaccess.PvObject({'value': pvaccess.INT}, {'value': scan + 1}),
            )

        if configuration.tomography_angle_pv is not None:
            angle_deg = scan * self._args.angle_step_deg
            self._server.update(
                configuration.tomography_angle_pv,
                pvaccess.PvObject({'value': pvaccess.DOUBLE}, {'value': angle_deg}),
            )

    def _replay_scan(self, offset: int) -> None:
        period_s = 1.0 / self._args.fps
        points = {int(index): k for k, index in enumerate(self._positions.get_indexes())}
        x_m = self._positions.get_coordinates_x_m()
        y_m = self._positions.get_coordinates_y_m()
        i0 = self._positions.get_probe_photon_counts()
        num_frames = 0
        start_s = time.monotonic()

        for index, pattern in _frames(self._dataset):
            self._server.update(
                self._configuration.input_channel,
                AdImageUtility.generateNtNdArray2D(
                    index + offset, numpy.ascontiguousarray(pattern)
                ),
            )
            now_s = time.monotonic()
            k = points.get(index)

            if k is not None:
                self._pending.append(
                    _Point(
                        event_id=index + offset - self._configuration.index_offset,
                        x_m=float(x_m[k]),
                        y_m=float(y_m[k]),
                        i0=None if i0 is None else float(i0[k]),
                        due_s=now_s + self._args.position_delay_s,
                    )
                )

            self._send_positions(now_s, final=False)
            num_frames += 1
            time.sleep(max(0.0, start_s + num_frames * period_s - time.monotonic()))

        while self._pending:
            self._send_positions(time.monotonic(), final=True)
            time.sleep(0.01)

        logger.info(f'Sent {num_frames} frames at offset {offset}.')

    def _send_positions(self, now_s: float, *, final: bool) -> None:
        due = [point for point in self._pending if point.due_s <= now_s]

        if not due:
            return

        if self._configuration.position_schema is PositionSchema.SGSOCKET:
            for point in due:
                attributes = pack_sgsocket_attributes(
                    self._configuration, x_m=point.x_m, y_m=point.y_m, i0=point.i0
                )
                self._server.update(
                    self._configuration.position_channel,
                    make_sgsocket_event(point.event_id, attributes),
                )
                self._pending.popleft()

            return

        if len(due) < self._args.position_batch_size and not final:
            return

        configuration = self._configuration
        i0 = [point.i0 for point in due]
        value = pack_batched_positions(
            configuration,
            [point.event_id for point in due],
            [point.x_m for point in due],
            [point.y_m for point in due],
            None if any(reading is None for reading in i0) else [float(r or 0.0) for r in i0],
        )
        self._server.update(
            configuration.position_channel,
            pvaccess.PvObject(make_batched_positions_type(self._position_axes), value),
        )

        for _ in due:
            self._pending.popleft()

    def _trigger(self, scan: int) -> None:
        self._acknowledged = False
        self._server.update(
            self._configuration.trigger_pv, pvaccess.PvObject(TRIGGER_TYPE, {'value': 1})
        )
        deadline_s = time.monotonic() + 10.0

        while not self._acknowledged and time.monotonic() < deadline_s:
            time.sleep(0.05)

        if self._acknowledged:
            logger.info(f'Scan {scan + 1} triggered and acknowledged.')
        else:
            logger.warning(f'Scan {scan + 1} triggered; no acknowledgement within 10 s.')

    def _free_run(self, first_index: int) -> None:
        pattern = next(_frames(self._dataset))[1]
        period_s = 1.0 / self._args.fps
        end_s = time.monotonic() + self._args.free_run_s
        index = first_index

        while time.monotonic() < end_s:
            self._server.update(
                self._configuration.input_channel,
                AdImageUtility.generateNtNdArray2D(index, numpy.ascontiguousarray(pattern)),
            )
            index += 1
            time.sleep(period_s)

        logger.info(f'Free-ran {index - first_index} frames without positions.')


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args)
    registry = PluginRegistry.load_plugins()

    if args.list_file_types:
        _list_file_types(registry)
        return 0

    required = ['config', 'diffraction_file', 'diffraction_file_type']
    required += ['position_file', 'position_file_type']

    for name in required:
        if getattr(args, name) is None:
            parser.error(f'--{name.replace("_", "-")} is required.')

    configuration = StreamConfiguration.from_file(args.config)

    if configuration.pv_protocol is not PvProtocol.PVA:
        parser.error('The replay hosts its PVs over PVA; set pv_protocol: pva.')

    diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name(
        args.diffraction_file_type
    )
    position_reader = registry.probe_position_file_readers.get_strategy_by_name(
        args.position_file_type
    )
    dataset = diffraction_reader.read(args.diffraction_file)
    positions = position_reader.read(args.position_file)
    logger.info(f'Replaying {len(positions)} positions from {args.position_file}.')
    _Replayer(configuration, args, dataset, positions).run()
    return 0


if __name__ == '__main__':
    sys.exit(main())
