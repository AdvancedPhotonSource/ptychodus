"""Probe-position reader for ESRF ID16A near-field ptychography scans.

Positions come from the ``spy`` and ``spz`` header entries, one value per frame, in
micrometers. Indexes run consecutively across the frame files in series order, the
same numbering the diffraction reader assigns, so patterns and positions pair by
index rather than by array order.

Motor coordinates are emitted as recorded, without sign flips. Handedness is
under-determined by the files -- LIMA has already applied a vertical flip to the
stored frames, and the stage axes carry no orientation of their own -- so a scan
whose reconstruction comes out mirrored is corrected with one of the affine presets
in the probe-position builder rather than by changing this reader.
"""

from __future__ import annotations

from pathlib import Path
import logging

import h5py

from ptychodus.api.constants import LengthUnit
from ptychodus.api.probe_positions import (
    ProbePosition,
    ProbePositionFileReader,
    ProbePositionParseError,
    ProbePositionSequence,
)

from ._scan import detector_group, parse_float_list, read_header, series_files

logger = logging.getLogger(__name__)


class ESRFID16ANearFieldPositionFileReader(ProbePositionFileReader):
    def get_file_description(self) -> str:
        return 'ESRF ID16A Near-Field Ptychography Files'

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> ProbePositionSequence:
        scale = LengthUnit.MICROMETER.meters_per_unit
        point_list: list[ProbePosition] = []

        try:
            frame_files = series_files(file_path)
        except ValueError as exc:
            raise ProbePositionParseError(str(exc)) from None

        for frame_file in frame_files:
            with h5py.File(frame_file, 'r') as h5_file:
                try:
                    num_frames = len(detector_group(h5_file)['data'])
                    header = read_header(h5_file)
                    spy = parse_float_list(header, 'spy', num_frames)
                    spz = parse_float_list(header, 'spz', num_frames)
                except (KeyError, ValueError) as exc:
                    raise ProbePositionParseError(f'"{frame_file}": {exc}') from None

            for x_um, y_um in zip(spy, spz):
                point_list.append(
                    ProbePosition(
                        index=len(point_list),
                        x_m=scale * x_um,
                        y_m=scale * y_um,
                    )
                )

        return ProbePositionSequence(point_list)
