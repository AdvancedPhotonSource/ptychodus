from pathlib import Path
import logging

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class PolarDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._data_path = '/entry/externals/eiger'
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)
            data_link = h5_file.get(self._data_path, getlink=True)

        if not isinstance(data_link, h5py.ExternalLink):
            raise ValueError(
                f'Expected "{self._data_path}" to be an external link; got {type(data_link)}.'
            )

        data_file_path = file_path.parent / data_link.filename
        logger.debug(f'Opening "{data_file_path}"...')

        with h5py.File(data_file_path, 'r') as h5_file:
            data = h5_file[data_link.path]

            if isinstance(data, h5py.Group):
                logger.warning('Link points to group; falling back to "/entry/data/data"')
                data = h5_file['/entry/data/data']

            if isinstance(data, h5py.Dataset):
                num_patterns, detector_height, detector_width = data.shape

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_patterns],
                    pattern_dtype=data.dtype,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    file_path=file_path,
                )

                array = H5DiffractionPatternArray(
                    label=file_path.stem,
                    indexes=numpy.arange(num_patterns),
                    file_path=data_file_path,
                    data_path=data.name,
                )
            else:
                raise ValueError(f'Expected "{data.name}" to be a dataset; got {type(data)}.')

        return SimpleDiffractionDataset(metadata, contents_tree, [array])


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        PolarDiffractionFileReader(),
        simple_name='APS_Polar',
        display_name='APS 4-ID Polar Files (*.hdf)',
    )
