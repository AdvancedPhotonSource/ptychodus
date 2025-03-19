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
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            with h5py.File(file_path, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._data_path]
                    pixelSizeInMicrons = h5File['det_pixel_size']
                except KeyError:
                    logger.warning('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape
                    pixelSizeInMeters = (
                        float(numpy.squeeze(pixelSizeInMicrons[()])) * self.ONE_MICRON_M
                    )

                    metadata = DiffractionMetadata(
                        num_patterns_per_array=numberOfPatterns,
                        num_patterns_total=numberOfPatterns,
                        pattern_dtype=data.dtype,
                        detector_extent=ImageExtent(detectorWidth, detectorHeight),
                        detector_pixel_geometry=PixelGeometry(pixelSizeInMeters, pixelSizeInMeters),
                        file_path=file_path,
                    )

                    array = H5DiffractionPatternArray(
                        label=file_path.stem,
                        indexes=numpy.arange(numberOfPatterns),
                        file_path=file_path,
                        data_path=self._data_path,
                    )

                    dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.warning(f'Unable to read file "{file_path}".')

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        NSLSIIDiffractionFileReader(),
        simple_name=NSLSIIDiffractionFileReader.SIMPLE_NAME,
        display_name=NSLSIIDiffractionFileReader.DISPLAY_NAME,
    )
