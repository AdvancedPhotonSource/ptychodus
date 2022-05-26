from __future__ import annotations
from pathlib import Path
import logging

import numpy

from ..api.data import DatasetState
from ..api.observer import Observable, Observer
from ..api.settings import SettingsGroup
from .data import ActiveDataFile
from .detector import CropSettings, DetectorSettings
from .probe import ProbeSettings
from .scan import ScanInitializer

logger = logging.getLogger(__name__)


class VelociprobePresenter(Observable, Observer):

    def __init__(self, velociprobeReader: VelociprobeReader, detectorSettings: DetectorSettings,
                 cropSettings: CropSettings, probeSettings: ProbeSettings,
                 activeDataFile: ActiveDataFile, scanInitializer: ScanInitializer) -> None:
        super().__init__()
        self._velociprobeReader = velociprobeReader
        self._detectorSettings = detectorSettings
        self._cropSettings = cropSettings
        self._probeSettings = probeSettings
        self._activeDataFile = activeDataFile
        self._scanInitializer = scanInitializer

    @classmethod
    def createInstance(cls, velociprobeReader: VelociprobeReader,
                       detectorSettings: DetectorSettings, cropSettings: CropSettings,
                       probeSettings: ProbeSettings, activeDataFile: ActiveDataFile,
                       scanInitializer: ScanInitializer):
        presenter = cls(velociprobeReader, detectorSettings, cropSettings, probeSettings,
                        activeDataFile, scanInitializer)
        velociprobeReader.addObserver(presenter)
        activeDataFile.addObserver(presenter)
        return presenter

    def getDatasetName(self, index: int) -> str:
        return self._activeDataFile[index].datasetName

    def getDatasetState(self, index: int) -> DatasetState:
        return self._activeDataFile[index].datasetState

    def getNumberOfDatasets(self) -> int:
        return len(self._activeDataFile)

    @property
    def _detectorGroup(self) -> DetectorGroup:
        return self._velociprobeReader.entryGroup.instrument.detector

    @property
    def _detectorSpecificGroup(self) -> DetectorSpecificGroup:
        return self._detectorGroup.detectorSpecific

    def syncDetectorPixelCount(self) -> None:
        self._detectorSettings.numberOfPixelsX.value = \
                int(self._detectorSpecificGroup.x_pixels_in_detector)
        self._detectorSettings.numberOfPixelsY.value = \
                int(self._detectorSpecificGroup.y_pixels_in_detector)

    def syncDetectorPixelSize(self) -> None:
        self._detectorSettings.pixelSizeXInMeters.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorGroup.x_pixel_size_m)
        self._detectorSettings.pixelSizeYInMeters.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorGroup.y_pixel_size_m)

    def syncDetectorDistance(self, overrideDistanceUnits: bool = False) -> None:
        value = SettingsGroup.convertFloatToDecimal(self._detectorGroup.detector_distance_m)

        if overrideDistanceUnits:
            value /= 1000

        self._detectorSettings.detectorDistanceInMeters.value = value

    def syncImageCrop(self, syncCenter: bool, syncExtent: bool) -> None:
        if syncCenter:
            self._cropSettings.centerXInPixels.value = \
                    int(round(self._detectorGroup.beam_center_x_px))
            self._cropSettings.centerYInPixels.value = \
                    int(round(self._detectorGroup.beam_center_y_px))

        if syncExtent:
            centerX = self._cropSettings.centerXInPixels.value
            centerY = self._cropSettings.centerYInPixels.value

            extentX = int(self._detectorSpecificGroup.x_pixels_in_detector)
            extentY = int(self._detectorSpecificGroup.y_pixels_in_detector)

            maxRadiusX = min(centerX, extentX - centerX)
            maxRadiusY = min(centerY, extentY - centerY)
            maxRadius = min(maxRadiusX, maxRadiusY)
            cropDiameterInPixels = 1

            while cropDiameterInPixels < maxRadius:
                cropDiameterInPixels <<= 1

            self._cropSettings.extentXInPixels.value = cropDiameterInPixels
            self._cropSettings.extentYInPixels.value = cropDiameterInPixels

    def syncProbeEnergy(self) -> None:
        self._probeSettings.probeEnergyInElectronVolts.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorSpecificGroup.photon_energy_eV)

    def loadScanFile(self) -> None:
        filePathMaster = self._activeDataFile.getFilePath()
        fileName = filePathMaster.stem.replace('master', 'pos') + '.csv'
        filePath = filePathMaster.parents[2] / 'positions' / fileName
        print(filePath.resolve())
        fileFilter = 'Comma-Separated Values Files (*.csv)'  # TODO refactor; get from somewhere
        self._scanInitializer.openScan(filePath, fileFilter)

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobeReader:
            self.notifyObservers()
        elif observable is self._activeDataFile:
            pass # TODO self.notifyObservers()
