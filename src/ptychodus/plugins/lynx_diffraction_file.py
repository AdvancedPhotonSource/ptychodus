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

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class LYNXDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._data_path = '/entry/data/eiger_4'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            with h5py.File(file_path, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._data_path]
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
        LYNXDiffractionFileReader(),
        simple_name='APS_LYNX',
        display_name='APS LYNX Files (*.h5 *.hdf5)',
    )
