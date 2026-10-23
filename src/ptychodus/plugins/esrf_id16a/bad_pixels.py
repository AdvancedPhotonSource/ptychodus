"""Bad-pixel reader built from an ESRF ID16A dark stack.

Subtracting the dark removes the pedestal but leaves the pixels that are not
measuring: those whose dark current runs far above the rest of the sensor, and those
stuck at full scale. Both read as signal once the background is gone, so they are
marked rather than corrected.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy

from ptychodus.api.diffraction import BadPixels, BadPixelsFileReader

from ._scan import detector_group


class ESRFID16ADarkBadPixelsFileReader(BadPixelsFileReader):
    """Flag the hot and stuck pixels of an ID16A dark stack.

    `hot_pixel_factor` multiplies the stack's median dark level to set the threshold
    above which a pixel counts as hot. Lower it to catch a sensor with a tighter dark
    distribution, raise it where a legitimately warm region is being discarded.
    """

    def __init__(self, *, hot_pixel_factor: float = 5.0) -> None:
        self._hot_pixel_factor = hot_pixel_factor

    def get_file_description(self) -> str:
        return 'ESRF ID16A Dark Field Files'

    def get_file_extensions(self) -> set[str]:
        return {'.h5', '.hdf5'}

    def read(self, file_path: Path) -> BadPixels:
        with h5py.File(file_path, 'r') as h5_file:
            frames = detector_group(h5_file)['data']

            if not isinstance(frames, h5py.Dataset) or frames.ndim != 3:
                raise ValueError(f'"{file_path}" does not hold a stack of dark frames.')

            stack = frames[()]

        mean = numpy.mean(stack, axis=0, dtype=numpy.float64)
        hot = mean > self._hot_pixel_factor * numpy.median(mean)

        # Full scale in any one frame is enough: a pixel that saturates in the dark
        # carries no headroom for signal in the frames beside it.
        saturated = numpy.max(stack, axis=0) >= numpy.iinfo(stack.dtype).max
        return numpy.logical_or(hot, saturated)
