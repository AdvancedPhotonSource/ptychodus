from pathlib import Path
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

from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class LYNXDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._dataPath = '/entry/data/eiger_4'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                    pixelSize = float(data.attrs['Pixel_size'].item())
                except KeyError:
                    logger.warning('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    metadata = DiffractionMetadata(
                        num_patterns_per_array=numberOfPatterns,
                        num_patterns_total=numberOfPatterns,
                        pattern_dtype=data.dtype,
                        detector_extent=ImageExtent(detectorWidth, detectorHeight),
                        detector_pixel_geometry=PixelGeometry(pixelSize, pixelSize),
                        file_path=filePath,
                    )

                    array = H5DiffractionPatternArray(
                        label=filePath.stem,
                        indexes=numpy.arange(numberOfPatterns),
                        filePath=filePath,
                        dataPath=self._dataPath,
                    )

                    dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.register_plugin(
        LYNXDiffractionFileReader(),
        simple_name='APS_LYNX',
        display_name='APS LYNX Diffraction Files (*.h5 *.hdf5)',
    )
