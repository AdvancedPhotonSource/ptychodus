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
from ptychodus.api.product import (
    ELECTRON_VOLT_J,
    LIGHT_SPEED_M_PER_S,
    PLANCK_CONSTANT_J_PER_HZ,
)

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class NSLS2Style1DiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            try:
                data = h5_file['/entry/data/data']
            except KeyError:
                data = h5_file['/diff_data/merlin1/det_images']

            if isinstance(data, h5py.Dataset):
                num_patterns, detector_height, detector_width = data.shape

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_patterns],
                    pattern_dtype=data.dtype,
                    detector_extent=ImageExtent(detector_width, detector_height),
                    probe_energy_eV=float(h5_file['/scan/energy'][()]),
                    exposure_time_s=float(h5_file['/scan/exposure_time'][()]),
                    file_path=file_path,
                )
                contents_tree = self._tree_builder.build(h5_file)
                array = H5DiffractionPatternArray(
                    label=file_path.stem,
                    indexes=numpy.arange(num_patterns),
                    file_path=file_path,
                    data_path=data.name,
                )

                return SimpleDiffractionDataset(metadata, contents_tree, [array])
            else:
                raise ValueError(f'Expected {data.name} to be a dataset; got {type(data)}.')


class NSLS2Style2DiffractionFileReader(DiffractionFileReader):
    ONE_MICRON_M: Final[float] = 1.0e-6
    ONE_NANOMETER_M: Final[float] = 1.0e-9

    def __init__(self) -> None:
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            data = h5_file['/diffamp']

            if isinstance(data, h5py.Dataset):
                num_patterns, detector_height, detector_width = data.shape

                pixel_size_m = float(h5_file['/ccd_pixel_um'][()]) * self.ONE_MICRON_M
                wavelength_m = float(h5_file['/lambda_nm'][()]) * self.ONE_NANOMETER_M
                hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S  # noqa: N806
                hc_eVm = hc_Jm / ELECTRON_VOLT_J  # noqa: N806
                probe_energy_eV = hc_eVm / wavelength_m  # noqa: N806

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_patterns],
                    pattern_dtype=data.dtype,
                    detector_distance_m=float(h5_file['z_m'][()]),
                    detector_extent=ImageExtent(detector_width, detector_height),
                    detector_pixel_geometry=PixelGeometry(pixel_size_m, pixel_size_m),
                    probe_energy_eV=probe_energy_eV,
                    tomography_angle_deg=float(h5_file['angle'][()]),
                    file_path=file_path,
                )
                contents_tree = self._tree_builder.build(h5_file)
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
        with h5py.File(file_path, 'r') as h5_file:
            data = h5_file['det_data']

            if isinstance(data, h5py.Dataset):
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
                contents_tree = self._tree_builder.build(h5_file)
                array = H5DiffractionPatternArray(
                    label=file_path.stem,
                    indexes=numpy.arange(num_patterns),
                    file_path=file_path,
                    data_path=data.name,
                )

                return SimpleDiffractionDataset(metadata, contents_tree, [array])
            else:
                raise ValueError(f'Expected {data.name} to be a dataset; got {type(data)}.')


def register_plugins(registry: PluginRegistry) -> None:
    registry.diffraction_file_readers.register_plugin(
        NSLS2Style1DiffractionFileReader(),
        simple_name='NSLS_II_1',
        display_name='NSLS-II Style 1 Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        NSLS2Style2DiffractionFileReader(),
        simple_name='NSLS_II_2',
        display_name='NSLS-II Style 2 Files (*.h5 *.hdf5)',
    )
    registry.diffraction_file_readers.register_plugin(
        NSLS2MATLABDiffractionFileReader(),
        simple_name='NSLS_II_MATLAB',
        display_name='NSLS-II MATLAB Files (*.mat)',
    )
