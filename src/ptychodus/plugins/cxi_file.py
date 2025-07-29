from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import ProbeSequence, ProbeFileReader
from ptychodus.api.product import ELECTRON_VOLT_J
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint
from ptychodus.api.typing import ComplexArrayType

logger = logging.getLogger(__name__)


class CXIDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._data_path = '/entry_1/data_1/data'
        self._tree_builder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        with h5py.File(file_path, 'r') as h5_file:
            contents_tree = self._tree_builder.build(h5_file)

            data = h5_file[self._data_path]

            if isinstance(data, h5py.Dataset):
                num_patterns, detector_height, detector_width = data.shape

                detector_extent = ImageExtent(detector_width, detector_height)
                detector_distance_m = float(
                    h5_file['/entry_1/instrument_1/detector_1/distance'][()]
                )
                detector_pixel_geometry = PixelGeometry(
                    float(h5_file['/entry_1/instrument_1/detector_1/x_pixel_size'][()]),
                    float(h5_file['/entry_1/instrument_1/detector_1/y_pixel_size'][()]),
                )
                probe_energy_J = float(h5_file['/entry_1/instrument_1/source_1/energy'][()])  # noqa: N806
                probe_energy_eV = probe_energy_J / ELECTRON_VOLT_J  # noqa: N806

                # TODO load detector mask; zeros are good pixels
                # /entry_1/instrument_1/detector_1/mask Dataset {512, 512}

                metadata = DiffractionMetadata(
                    num_patterns_per_array=[num_patterns],
                    pattern_dtype=data.dtype,
                    detector_distance_m=detector_distance_m,
                    detector_extent=detector_extent,
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

                return SimpleDiffractionDataset(metadata, contents_tree, [array])
            else:
                raise ValueError(f'Expected dataset at {self._data_path}, got {type(data)}.')


class CXIPositionFileReader(PositionFileReader):
    def read(self, file_path: Path) -> PositionSequence:
        point_list: list[ScanPoint] = list()

        with h5py.File(file_path, 'r') as h5_file:
            xyz_m = h5_file['/entry_1/data_1/translation'][()]

            for idx, (x, y, z) in enumerate(xyz_m):
                point = ScanPoint(idx, x, y)
                point_list.append(point)

        return PositionSequence(point_list)


class CXIProbeFileReader(ProbeFileReader):
    def read(self, file_path: Path) -> ProbeSequence:
        array: ComplexArrayType | None = None

        with h5py.File(file_path, 'r') as h5_file:
            array = h5_file['/entry_1/instrument_1/source_1/illumination'][()]

        return ProbeSequence(array=array, opr_weights=None, pixel_geometry=None)


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'CXI'  # noqa: N806
    DISPLAY_NAME: Final[str] = 'Coherent X-ray Imaging Files (*.cxi)'  # noqa: N806

    registry.diffraction_file_readers.register_plugin(
        CXIDiffractionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
    registry.position_file_readers.register_plugin(
        CXIPositionFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
    registry.probe_file_readers.register_plugin(
        CXIProbeFileReader(),
        simple_name=SIMPLE_NAME,
        display_name=DISPLAY_NAME,
    )
