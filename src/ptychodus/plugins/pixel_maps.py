"""Readers for per-pixel detector maps: a measured background and a flat-field response."""

from pathlib import Path

import numpy

from ptychodus.api.diffraction import PixelMapFileReader
from ptychodus.api.plugins import PluginRegistry


def _read_pixel_map(file_path: Path) -> numpy.ndarray:
    pixel_map = numpy.load(file_path)

    if pixel_map.ndim != 2:
        raise ValueError(f'"{file_path}" holds a {pixel_map.ndim}-D array, not a 2-D map.')

    return pixel_map.astype(numpy.float64)


class NPYPixelMapFileReader(PixelMapFileReader):
    """A two-dimensional map stored as a NumPy array, used as it is.

    The same file holds a background or a flat-field response depending on which
    chooser it is registered in, so the registration states which one it describes.
    """

    def __init__(self, description: str) -> None:
        self._description = description

    def get_file_description(self) -> str:
        return self._description

    def get_file_extensions(self) -> set[str]:
        return {'.npy'}

    def read(self, file_path: Path) -> numpy.ndarray:
        return _read_pixel_map(file_path)


class NPYFlatFieldCorrectionFileReader(PixelMapFileReader):
    """A map of per-pixel correction factors to multiply by, read as the response it inverts.

    A flat-field response is divided out of a measurement, so a correction factor is
    its reciprocal. Non-positive factors have no response to recover and read as zero,
    which marks those pixels bad wherever the flat field is applied.
    """

    def get_file_description(self) -> str:
        return 'NumPy Flat-Field Correction Factor Files'

    def get_file_extensions(self) -> set[str]:
        return {'.npy'}

    def read(self, file_path: Path) -> numpy.ndarray:
        factors = _read_pixel_map(file_path)
        return numpy.divide(1.0, factors, out=numpy.zeros_like(factors), where=factors > 0.0)


def register_plugins(registry: PluginRegistry) -> None:
    registry.background_file_readers.register_plugin(
        NPYPixelMapFileReader('NumPy Background Files'),
        simple_name='NPY',
    )
    registry.flat_field_file_readers.register_plugin(
        NPYPixelMapFileReader('NumPy Flat-Field Response Files'),
        simple_name='NPY',
    )
    registry.flat_field_file_readers.register_plugin(
        NPYFlatFieldCorrectionFileReader(),
        simple_name='NPY_Flat_Field_Correction',
    )
