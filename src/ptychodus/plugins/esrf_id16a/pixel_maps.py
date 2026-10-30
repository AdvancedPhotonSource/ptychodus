"""Detector background reader for ESRF ID16A dark stacks.

A scan directory carries a ``dark_0000.h5`` holding the frames the detector records
with the beam off. Their per-pixel mean is the background the detector-correction
machinery subtracts; this reader only supplies the map, and whether it is applied is
a detector setting.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy

from ptychodus.api.diffraction import PixelMap, PixelMapFileReader

from ._scan import detector_group


class ESRFID16ADarkFileReader(PixelMapFileReader):
    """Mean of an ID16A dark stack, as a background map on the full detector."""

    def get_file_description(self) -> str:
        return 'ESRF ID16A Dark Field Files'

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> PixelMap:
        with h5py.File(file_path, 'r') as h5_file:
            frames = detector_group(h5_file)['data']

            if not isinstance(frames, h5py.Dataset) or frames.ndim != 3:
                raise ValueError(f'"{file_path}" does not hold a stack of dark frames.')

            # asarray narrows what mean() is typed to return; reducing a 3-D stack
            # over axis 0 is always an array, never the scalar the union admits.
            return numpy.asarray(numpy.mean(frames[()], axis=0, dtype=numpy.float64))
