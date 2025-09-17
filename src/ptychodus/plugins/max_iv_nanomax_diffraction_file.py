from pathlib import Path
from typing import Final
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


class NanoMAXDiffractionFileReader(DiffractionFileReader):
    DATA_PATHS: Final[tuple[str, ...]] = (
        '/entry/measurement/eiger1m/frames',
        '/entry/instrument/Eiger/data',
        '/entry/measurement/Eiger/data',
    )

    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)
            data: h5py.Dataset | None = None

            for data_path in self.DATA_PATHS:
                data_link = h5_file.get(data_path, getlink=True)

                if data_link is None:
                    logger.debug(f'Data not found at "{data_path}"')
                else:
                    data = h5_file.get(data_path)
                    break

            if data is None:
                raise ValueError('Failed to find data!')
            elif isinstance(data, h5py.Dataset):
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
                    file_path=data.file.filename,
                    data_path=data.name,
                )

                return SimpleDiffractionDataset(metadata, contents_tree, [array])
            else:
                raise ValueError(f'Expected {data.name} to be a dataset; got {type(data)}.')


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        NanoMAXDiffractionFileReader(),
        simple_name='MAX_IV_NanoMax',
        display_name='MAX IV NanoMax Files (*.h5 *.hdf5)',
    )
