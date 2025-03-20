from pathlib import Path
from typing import Final
import logging

import h5py
import numpy

from .h5_diffraction_file import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.probe import Probe, ProbeFileReader
from ptychodus.api.product import ELECTRON_VOLT_J
from ptychodus.api.propagator import WavefieldArrayType
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

logger = logging.getLogger(__name__)


class CXIDiffractionFileReader(DiffractionFileReader):
    def __init__(self) -> None:
        self._data_path = '/entry_1/data_1/data'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, file_path: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.create_null(file_path)

        try:
            with h5py.File(file_path, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._data_path]
                except KeyError:
                    logger.warning('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape

                    detectorExtent = ImageExtent(detectorWidth, detectorHeight)
                    detectorDistanceInMeters = float(
                        h5File['/entry_1/instrument_1/detector_1/distance'][()]
                    )
                    detectorPixelGeometry = PixelGeometry(
                        float(h5File['/entry_1/instrument_1/detector_1/x_pixel_size'][()]),
                        float(h5File['/entry_1/instrument_1/detector_1/y_pixel_size'][()]),
                    )
                    probeEnergyInJoules = float(h5File['/entry_1/instrument_1/source_1/energy'][()])
                    probeEnergyInElectronVolts = probeEnergyInJoules / ELECTRON_VOLT_J

                    # TODO load detector mask; zeros are good pixels
                    # /entry_1/instrument_1/detector_1/mask Dataset {512, 512}

                    metadata = DiffractionMetadata(
                        num_patterns_per_array=numberOfPatterns,
                        num_patterns_total=numberOfPatterns,
                        pattern_dtype=data.dtype,
                        detector_distance_m=detectorDistanceInMeters,
                        detector_extent=detectorExtent,
                        detector_pixel_geometry=detectorPixelGeometry,
                        probe_energy_eV=probeEnergyInElectronVolts,
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


class CXIPositionFileReader(PositionFileReader):
    def read(self, file_path: Path) -> PositionSequence:
        scanPointList: list[ScanPoint] = list()

        with h5py.File(file_path, 'r') as h5File:
            xyz_m = h5File['/entry_1/data_1/translation'][()]

            for idx, (x, y, z) in enumerate(xyz_m):
                point = ScanPoint(idx, x, y)
                scanPointList.append(point)

        return PositionSequence(scanPointList)


class CXIProbeFileReader(ProbeFileReader):
    def read(self, file_path: Path) -> Probe:
        array: WavefieldArrayType | None = None

        with h5py.File(file_path, 'r') as h5File:
            array = h5File['/entry_1/instrument_1/source_1/illumination'][()]

        return Probe(array=array, pixel_geometry=None)


def register_plugins(registry: PluginRegistry) -> None:
    SIMPLE_NAME: Final[str] = 'CXI'
    DISPLAY_NAME: Final[str] = 'Coherent X-ray Imaging Files (*.cxi)'

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
