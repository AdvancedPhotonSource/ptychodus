from pathlib import Path

import numpy

from ptychodus.api.diffraction import BadPixels, BadPixelsFileReader
from ptychodus.api.plugins import PluginRegistry


class NPYBadPixelsFileReader(BadPixelsFileReader):
    def read(self, file_path: Path) -> BadPixels:
        return numpy.load(file_path)


class NPYGoodPixelsFileReader(BadPixelsFileReader):
    def read(self, file_path: Path) -> BadPixels:
        return numpy.logical_not(numpy.load(file_path))


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
