from collections.abc import Mapping
from pathlib import Path
import logging
import re

import h5py
import numpy

from ptychodus.api.geometry import ImageExtent
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    DiffractionArray,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.tree import SimpleTreeNode

from .h5_diffraction_file import H5DiffractionPatternArray

logger = logging.getLogger(__name__)


class APS2IDDiffractionFileReader(DiffractionFileReader):
    def _get_file_series(self, file_path: Path) -> tuple[Mapping[int, Path], str]:
        file_path_dict: dict[int, Path] = dict()

        digits = re.findall(r'\d+', file_path.stem)
        longest_digits = max(digits, key=len)
        file_pattern = file_path.name.replace(longest_digits, f'(\\d{{{len(longest_digits)}}})')

        for fp in file_path.parent.iterdir():
            z = re.match(file_pattern, fp.name)

            if z:
                index = int(z.group(1))
                file_path_dict[index] = fp

        return file_path_dict, file_pattern

    def read(self, file_path: Path) -> DiffractionDataset:
        file_path_mapping, file_pattern = self._get_file_series(file_path)
        data_path = '/entry/data/data'

        with h5py.File(file_path, 'r') as h5_file:
            h5data = h5_file[data_path]

            if isinstance(h5data, h5py.Dataset):
                num_patterns_per_array, detector_height, detector_width = h5data.shape
                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_patterns_per_array] * len(file_path_mapping),
                    pattern_dtype=h5data.dtype,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    file_path=file_path.parent / file_pattern,
                )
                contents_tree = SimpleTreeNode.create_root(['Name', 'Type', 'Details'])
                array_list: list[DiffractionArray] = list()

                for idx, fp in sorted(file_path_mapping.items()):
                    indexes = numpy.arange(num_patterns_per_array) + idx * num_patterns_per_array
                    array = H5DiffractionPatternArray(fp.stem, indexes, fp, data_path)
                    contents_tree.create_child([array.get_label(), 'HDF5', str(idx)])
                    array_list.append(array)

                return SimpleDiffractionDataset(metadata, contents_tree, array_list)
            else:
                raise ValueError(f'Expected dataset at "{data_path}"; found {type(h5data)}.')


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_2IDD',
        display_name='APS 2-ID-D Microprobe Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_2IDE',
        display_name='APS 2-ID-E Microprobe Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        APS2IDDiffractionFileReader(),
        simple_name='APS_BNP',
        display_name='APS 2-ID-D Bionanoprobe Files (*.h5 *.hdf5)',
    )
