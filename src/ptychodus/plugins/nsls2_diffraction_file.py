from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class NSLSIIDiffractionFileReader(DiffractionFileReader):
    SIMPLE_NAME: Final[str] = 'NSLS-II'
    DISPLAY_NAME: Final[str] = 'NSLS-II Files (*.mat)'
    ONE_MICRON_M: Final[float] = 1e-6

    def __init__(self) -> None:
        self._data_path = 'det_data'
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            with h5py.File(file_path, 'r') as h5_file:
                contents_tree = self._tree_builder.build(h5_file)

                try:
                    data = h5_file[self._data_path]
                except KeyError:
                    logger.warning('Unable to load data.')
                else:
                    num_patterns, detector_height, detector_width = data.shape
                    pixel_size_m = (
                        float(numpy.squeeze(h5_file['det_pixel_size'][()])) * self.ONE_MICRON_M
                    )

                    metadata = DiffractionMetadata(
                        num_patterns_per_array=num_patterns,
                        num_patterns_total=num_patterns,
                        pattern_dtype=data.dtype,
                        detector_extent=ImageExtent(detector_width, detector_height),
                        detector_pixel_geometry=PixelGeometry(pixel_size_m, pixel_size_m),
                        file_path=file_path,
                    )

                    array = H5DiffractionPatternArray(
                        label=file_path.stem,
                        indexes=numpy.arange(num_patterns),
                        file_path=file_path,
                        data_path=self._data_path,
                    )

                    dataset = SimpleDiffractionDataset(metadata, contents_tree, [array])
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        NSLSIIDiffractionFileReader(),
        simple_name=NSLSIIDiffractionFileReader.SIMPLE_NAME,
        display_name=NSLSIIDiffractionFileReader.DISPLAY_NAME,
    )
