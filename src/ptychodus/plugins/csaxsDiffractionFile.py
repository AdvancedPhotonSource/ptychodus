from pathlib import Path
from typing import Final
import logging

import h5py

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.patterns import (
    DiffractionDataset,
    DiffractionFileReader,
    DiffractionMetadata,
    SimpleDiffractionDataset,
)
from ptychodus.api.plugins import PluginRegistry

from .h5DiffractionFile import H5DiffractionPatternArray, H5DiffractionFileTreeBuilder

logger = logging.getLogger(__name__)


class CSAXSDiffractionFileReader(DiffractionFileReader):
    SIMPLE_NAME: Final[str] = 'SLS_cSAXS'
    DISPLAY_NAME: Final[str] = 'SLS cSAXS Diffraction Files (*.h5 *.hdf5)'
    ONE_MICRON_M: Final[float] = 1e-6
    ONE_MILLIMETER_M: Final[float] = 1e-3

    def __init__(self) -> None:
        self._dataPath = '/entry/data/data'
        self._treeBuilder = H5DiffractionFileTreeBuilder()

    def read(self, filePath: Path) -> DiffractionDataset:
        dataset = SimpleDiffractionDataset.createNullInstance(filePath)

        try:
            with h5py.File(filePath, 'r') as h5File:
                contentsTree = self._treeBuilder.build(h5File)

                try:
                    data = h5File[self._dataPath]
                    x_pixel_size_um = h5File['/entry/instrument/eiger_4/x_pixel_size']
                    y_pixel_size_um = h5File['/entry/instrument/eiger_4/y_pixel_size']
                    distance_mm = h5File['/entry/instrument/monochromator/distance']
                    energy_keV = h5File['/entry/instrument/monochromator/energy']
                except KeyError:
                    logger.warning('Unable to load data.')
                else:
                    numberOfPatterns, detectorHeight, detectorWidth = data.shape
                    detectorDistanceInMeters = float(distance_mm[()]) * self.ONE_MILLIMETER_M
                    detectorPixelGeometry = PixelGeometry(
                        widthInMeters=float(x_pixel_size_um[()]) * self.ONE_MICRON_M,
                        heightInMeters=float(y_pixel_size_um[()]) * self.ONE_MICRON_M,
                    )
                    probeEnergyInElectronVolts = 1000 * float(energy_keV[()])

                    metadata = DiffractionMetadata(
                        numberOfPatternsPerArray=numberOfPatterns,
                        numberOfPatternsTotal=numberOfPatterns,
                        patternDataType=data.dtype,
                        detectorDistanceInMeters=abs(detectorDistanceInMeters),
                        detectorExtent=ImageExtent(detectorWidth, detectorHeight),
                        detectorPixelGeometry=detectorPixelGeometry,
                        probeEnergyInElectronVolts=probeEnergyInElectronVolts,
                        filePath=filePath,
                    )

                    array = H5DiffractionPatternArray(
                        label=filePath.stem,
                        index=0,
                        filePath=filePath,
                        dataPath=self._dataPath,
                    )

                    dataset = SimpleDiffractionDataset(metadata, contentsTree, [array])
        except OSError:
            logger.warning(f'Unable to read file "{filePath}".')

        return dataset


def registerPlugins(registry: PluginRegistry) -> None:
    registry.diffractionFileReaders.registerPlugin(
        CSAXSDiffractionFileReader(),
        simpleName=CSAXSDiffractionFileReader.SIMPLE_NAME,
        displayName=CSAXSDiffractionFileReader.DISPLAY_NAME,
    )
