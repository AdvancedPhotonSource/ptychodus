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


class CSAXSDiffractionFileReader(DiffractionFileReader):
    ONE_MICRON_M: Final[float] = 1e-6
    ONE_MILLIMETER_M: Final[float] = 1e-3

    def __init__(self) -> None:
        self._data_path = '/entry/data/data'
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            with h5py.File(file_path, 'r') as h5_file:
                contents_tree = self._tree_builder.build(h5_file)

                try:
                    data = h5_file[self._data_path]
                    x_pixel_size_um = h5_file['/entry/instrument/eiger_4/x_pixel_size']
                    y_pixel_size_um = h5_file['/entry/instrument/eiger_4/y_pixel_size']
                    distance_mm = h5_file['/entry/instrument/monochromator/distance']
                    energy_keV = h5_file['/entry/instrument/monochromator/energy']  # noqa: N806
                except KeyError:
                    logger.warning('Unable to load data.')
                else:
                    num_patterns, detector_height, detector_width = data.shape
                    detector_distance_m = float(distance_mm[()]) * self.ONE_MILLIMETER_M
                    detector_pixel_geometry = PixelGeometry(
                        width_m=float(x_pixel_size_um[()]) * self.ONE_MICRON_M,
                        height_m=float(y_pixel_size_um[()]) * self.ONE_MICRON_M,
                    )
                    probe_energy_eV = 1000 * float(energy_keV[()])  # noqa: N806

                    metadata = DiffractionMetadata(
                        num_patterns_per_array=num_patterns,
                        num_patterns_total=num_patterns,
                        pattern_dtype=data.dtype,
                        detector_distance_m=abs(detector_distance_m),
                        detector_extent=ImageExtent(detector_width, detector_height),
                        detector_pixel_geometry=detector_pixel_geometry,
                        probe_energy_eV=probe_energy_eV,
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
        CSAXSDiffractionFileReader(),
        simple_name='SLS_cSAXS',
        display_name='SLS cSAXS Files (*.h5 *.hdf5)',
    )
