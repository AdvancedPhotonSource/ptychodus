from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class NSLS2DiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            metadata = DiffractionMetadata.create_null(file_path)
            contents_tree = self._tree_builder.build(h5_file)

            try:
                data = h5_file['/entry/data/data']
            except KeyError:
                try:
                    data = h5_file['/diff_data/merlin1/det_images']
                except KeyError:
                    raise ValueError('Failed to locate diffraction patterns.')

            if isinstance(data, h5py.Dataset):
                num_patterns, detector_height, detector_width = data.shape

                metadata = DiffractionMetadata(  # FIXME additional metadata
                    num_patterns_per_array=[num_patterns],
                    pattern_dtype=data.dtype,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    file_path=file_path,
                )

                array = H5DiffractionPatternArray(
                    label=file_path.stem,
                    indexes=numpy.arange(num_patterns),
                    file_path=file_path,
                    data_path=data.name,
                )

                return SimpleDiffractionDataset(metadata, contents_tree, [array])
            else:
                raise ValueError(f'Expected {data.name} to be a dataset; got {type(data)}.')


class NSLS2MATLABDiffractionFileReader(DiffractionFileReader):
    ONE_MICRON_M: Final[float] = 1e-6

    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)

            try:
                data = h5_file['det_data']
            except KeyError:
                logger.warning('Unable to load data.')
            else:
                num_patterns, detector_height, detector_width = data.shape
                pixel_size_m = (
                    float(numpy.squeeze(h5_file['det_pixel_size'][()])) * self.ONE_MICRON_M
                )

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_patterns],
                    pattern_dtype=data.dtype,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    detector_pixel_geometry=PixelGeometry(pixel_size_m, pixel_size_m),
                    file_path=file_path,
                )

                array = H5DiffractionPatternArray(
                    label=file_path.stem,
                    indexes=numpy.arange(num_patterns),
                    file_path=file_path,
                    data_path=data.name,
                )

                dataset = SimpleDiffractionDataset(metadata, contents_tree, [array])

        return dataset


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        NSLS2DiffractionFileReader(),
        simple_name='NSLS_II',
        display_name='NSLS-II Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        NSLS2MATLABDiffractionFileReader(),
        simple_name='NSLS_II_MATLAB',
        display_name='NSLS-II MATLAB Files (*.mat)',
    )
