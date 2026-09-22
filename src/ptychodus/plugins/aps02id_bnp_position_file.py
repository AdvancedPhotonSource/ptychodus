"""Probe position reader for pre-APS-U Bionanoprobe ``*.mda.h5`` files.

The instrument wrote scan positions two ways across the upgrade. Afterwards they live
in the EPICS MDA file, which :mod:`ptychodus.plugins.mda` reads. Before it they live in
the XRF map beside the data, as the two axis vectors ``MAPS/x_axis`` and ``MAPS/y_axis``
spanning a rectangular grid, which is what this reader handles.

Positions are reported as the file states them, in micrometers converted to meters and
not re-centered, matching :class:`MDAPositionFileReader` so the two eras agree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.constants import LengthUnit
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe_positions import (
    ProbePosition,
    ProbePositionFileReader,
    ProbePositionParseError,
    ProbePositionSequence,
)

logger = logging.getLogger(__name__)


class BionanoprobeMAPSPositionFileReader(ProbePositionFileReader):
    """Reader for the ``MAPS`` axis vectors in a pre-APS-U Bionanoprobe XRF map."""

    SIMPLE_NAME: Final[str] = 'APS_BNP_MAPS'
    DISPLAY_NAME: Final[str] = 'APS 2-ID-D Bionanoprobe MAPS Files (*.h5 *.hdf5)'

    X_AXIS_PATH: Final[str] = 'MAPS/x_axis'
    Y_AXIS_PATH: Final[str] = 'MAPS/y_axis'

    def _read_axis(self, h5_file: h5py.File, path: str) -> numpy.ndarray:
        try:
            dataset = h5_file[path]
        except KeyError:
            raise ProbePositionParseError(
                f'"{h5_file.filename}" has no "{path}"; it is not a Bionanoprobe MAPS file.'
            ) from None

        if not isinstance(dataset, h5py.Dataset):
            raise ProbePositionParseError(f'"{path}" in "{h5_file.filename}" is not a dataset.')

        axis = numpy.atleast_1d(numpy.asarray(dataset[()], dtype=float).squeeze())

        if axis.ndim != 1 or axis.size == 0:
            raise ProbePositionParseError(
                f'"{path}" in "{h5_file.filename}" is not a non-empty axis vector '
                f'(shape {axis.shape}).'
            )

        return axis

    def read(self, file_path: Path) -> ProbePositionSequence:
        with h5py.File(file_path, 'r') as h5_file:
            x_axis = self._read_axis(h5_file, self.X_AXIS_PATH)
            y_axis = self._read_axis(h5_file, self.Y_AXIS_PATH)

        logger.debug('Read a %d x %d position grid from %s', y_axis.size, x_axis.size, file_path)

        scale = LengthUnit.MICROMETER.meters_per_unit
        point_list: list[ProbePosition] = []

        # Row-major over the grid, so the index matches the frame order the detector
        # wrote: every point of a row before advancing to the next.
        for y_m in y_axis:
            for x_m in x_axis:
                point_list.append(
                    ProbePosition(
                        index=len(point_list),
                        x_m=float(x_m) * scale,
                        y_m=float(y_m) * scale,
                    )
                )

        return ProbePositionSequence(point_list)


def register_plugins(registry: PluginRegistry) -> None:
    registry.probe_position_file_readers.register_plugin(
        BionanoprobeMAPSPositionFileReader(),
        simple_name=BionanoprobeMAPSPositionFileReader.SIMPLE_NAME,
        display_name=BionanoprobeMAPSPositionFileReader.DISPLAY_NAME,
    )
