from pathlib import Path
from typing import Any, Final
import json

import h5py
import numpy

from ptychodus.api.diffraction import BadPixels, BadPixelsFileReader
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.plugins import PluginRegistry


class NPYBadPixelsFileReader(BadPixelsFileReader):
    def read(self, file_path: Path) -> BadPixels:
        return numpy.load(file_path)


class NPYGoodPixelsFileReader(BadPixelsFileReader):
    def read(self, file_path: Path) -> BadPixels:
        return numpy.logical_not(numpy.load(file_path))


class APS12IDValidPixelMaskFileReader(BadPixelsFileReader):
    DATA_PATH: Final[str] = 'valid_pixel_mask'

    def read(self, file_path: Path) -> BadPixels:
        with h5py.File(file_path, 'r') as h5_file:
            valid = h5_file[self.DATA_PATH][()]

        return numpy.logical_not(numpy.asarray(valid, dtype=bool))


class AtomicBadPixelsFileReader(BadPixelsFileReader):
    """Read the bad-pixel list written by the Atomic detector mask editor.

    The file is JSON holding one "Bad pixels" array of ``{"Pixel": [column, row],
    "Set": int}`` objects, one per masked pixel. Note the axis order: the first
    coordinate is the column. The list carries no detector dimensions of its own, so the
    extent is fixed to the EIGER2 CdTe 1M that writes it and a coordinate outside that
    extent is an error. Growing the mask to fit instead would produce an array whose
    shape depends on which corner pixels happen to be masked.

    "Set" is not interpreted. A pixel is bad by virtue of being listed.
    """

    DETECTOR_EXTENT: Final[ImageExtent] = ImageExtent(width_px=1028, height_px=1062)
    BAD_PIXELS_KEY: Final[str] = 'Bad pixels'
    PIXEL_KEY: Final[str] = 'Pixel'

    def read(self, file_path: Path) -> BadPixels:
        with file_path.open(mode='r') as fp:
            contents: Any = json.load(fp)

        try:
            entry_list = contents[self.BAD_PIXELS_KEY]
        except (KeyError, TypeError):
            raise ValueError(f'"{file_path}" has no "{self.BAD_PIXELS_KEY}" array.') from None

        bad_pixels = numpy.zeros(self.DETECTOR_EXTENT.get_shape(), dtype=bool)

        for number, entry in enumerate(entry_list):
            try:
                column, row = entry[self.PIXEL_KEY]
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f'Entry {number} of "{file_path}" is not a'
                    f' {{"{self.PIXEL_KEY}": [column, row]}} object.'
                ) from None

            if row < 0 or row >= self.DETECTOR_EXTENT.height_px:
                raise ValueError(
                    f'Entry {number} of "{file_path}" has row {row}, which is outside the'
                    f' {self.DETECTOR_EXTENT.height_px}-row detector.'
                )

            if column < 0 or column >= self.DETECTOR_EXTENT.width_px:
                raise ValueError(
                    f'Entry {number} of "{file_path}" has column {column}, which is outside the'
                    f' {self.DETECTOR_EXTENT.width_px}-column detector.'
                )

            bad_pixels[row, column] = True

        return bad_pixels


def register_plugins(registry: PluginRegistry) -> None:
    registry.bad_pixels_file_readers.register_plugin(
        NPYBadPixelsFileReader(),
        simple_name='NPY_Bad_Pixels',
        display_name='NumPy Bad Pixel Files (*.npy)',
    )
    registry.bad_pixels_file_readers.register_plugin(
        NPYGoodPixelsFileReader(),
        simple_name='NPY_Good_Pixels',
        display_name='NumPy Good Pixel Files (*.npy)',
    )
    registry.bad_pixels_file_readers.register_plugin(
        APS12IDValidPixelMaskFileReader(),
        simple_name='APS_12ID_Valid_Pixel_Mask',
        display_name='APS 12-ID-E Valid Pixel Mask Files (*.h5 *.hdf5)',
    )
    registry.bad_pixels_file_readers.register_plugin(
        AtomicBadPixelsFileReader(),
        simple_name='APS_Atomic_Bad_Pixels',
        display_name='APS 34-ID Atomic Bad Pixel Files (*.json)',
    )
