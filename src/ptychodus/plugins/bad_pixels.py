from pathlib import Path
from typing import Final

import h5py
import numpy

from ptychodus.api.diffraction import BadPixels, BadPixelsFileReader
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
