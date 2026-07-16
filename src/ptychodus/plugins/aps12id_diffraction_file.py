from pathlib import Path
from typing import Final
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

from .h5_diffraction_file import H5DiffractionFileTreeBuilder, H5DiffractionPatternArray

logger = logging.getLogger(__name__)


def _read_ndattribute_scalar(h5_file: h5py.File, path: str) -> float | None:
    try:
        value = h5_file[path][()]
    except KeyError:
        logger.warning(f'NDAttribute "{path}" not found in "{h5_file.filename}".')
        return None

    array = numpy.atleast_1d(value)
    return float(array[0])


def _glob_h5_series(file_path: Path) -> tuple[dict[int, Path], str]:
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


class APS12IDDiffractionFileReader(DiffractionFileReader):
    DATA_PATH: Final[str] = '/entry/data/data'
    ENERGY_PATH: Final[str] = '/entry/instrument/NDAttributes/monoE'
    EXPOSURE_PATH: Final[str] = '/entry/instrument/NDAttributes/ExposureTime'

    def read(self, file_path: Path) -> DiffractionDataset:
        tree_builder = H5DiffractionFileTreeBuilder()

        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = tree_builder.build(h5_file)

            try:
                h5_data = h5_file[self.DATA_PATH]
            except KeyError:
                logger.warning(f'File "{file_path}" is not an APS 12-ID data file.')
                return SimpleDiffractionDataset.create_null(file_path)

            if not isinstance(h5_data, h5py.Dataset):
                logger.warning(f'Data path "{self.DATA_PATH}" in "{file_path}" is not a dataset.')
                return SimpleDiffractionDataset.create_null(file_path)

            data_shape = h5_data.shape
            data_dtype = h5_data.dtype
            probe_energy_eV = _read_ndattribute_scalar(h5_file, self.ENERGY_PATH)  # noqa: N806
            exposure_time_s = _read_ndattribute_scalar(h5_file, self.EXPOSURE_PATH)

        if len(data_shape) == 3:
            num_patterns, detector_height, detector_width = data_shape

            metadata = DiffractionMetadata(
                num_patterns_per_array=[num_patterns],
                pattern_dtype=data_dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                probe_energy_eV=probe_energy_eV,
                exposure_time_s=exposure_time_s,
                file_path=file_path,
            )
            array = H5DiffractionPatternArray(
                label=file_path.stem,
                indexes=numpy.arange(num_patterns),
                file_path=file_path,
                data_path=self.DATA_PATH,
            )
            return SimpleDiffractionDataset(metadata, contents_tree, [array])

        if len(data_shape) == 2:
            detector_height, detector_width = data_shape
            file_path_dict, file_pattern = _glob_h5_series(file_path)
            array_list: list[DiffractionArray] = list()

            for idx, (_, fp) in enumerate(sorted(file_path_dict.items())):
                indexes = numpy.array([idx])
                array = H5DiffractionPatternArray(fp.stem, indexes, fp, self.DATA_PATH)
                array_list.append(array)

            metadata = DiffractionMetadata(
                num_patterns_per_array=[1] * len(array_list),
                pattern_dtype=data_dtype,
                detector_extent=ImageExtent(detector_width, detector_height),
                probe_energy_eV=probe_energy_eV,
                exposure_time_s=exposure_time_s,
                file_path=file_path.parent / file_pattern,
            )
            return SimpleDiffractionDataset(metadata, contents_tree, array_list)

        logger.warning(
            f'Data path "{self.DATA_PATH}" in "{file_path}" has unsupported shape {data_shape}.'
        )
        return SimpleDiffractionDataset.create_null(file_path)


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        APS12IDDiffractionFileReader(),
        simple_name='APS_PtychoSAXS',
        display_name='APS 12-ID PtychoSAXS Files (*.h5 *.hdf5)',
    )
