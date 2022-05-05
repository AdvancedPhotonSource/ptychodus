from __future__ import annotations
from pathlib import Path
import logging

import h5py
import numpy

from ..api.data import DatasetState
from ..api.observer import Observable, Observer
from ..api.settings import SettingsGroup
from .crop import CropSettings
from .detector import DetectorSettings
from .image import ImageSequence
from .probe import ProbeSettings

logger = logging.getLogger(__name__)


class VelociprobePresenter(Observable, Observer):
    def __init__(self, velociprobeReader: VelociprobeReader, detectorSettings: DetectorSettings,
                 cropSettings: CropSettings, probeSettings: ProbeSettings) -> None:
        super().__init__()
        self._velociprobeReader = velociprobeReader
        self._detectorSettings = detectorSettings
        self._cropSettings = cropSettings
        self._probeSettings = probeSettings

    @classmethod
    def createInstance(cls, velociprobeReader: VelociprobeReader,
                       detectorSettings: DetectorSettings, cropSettings: CropSettings,
                       probeSettings: ProbeSettings):
        presenter = cls(velociprobeReader, detectorSettings, cropSettings, probeSettings)
        velociprobeReader.addObserver(presenter)
        return presenter

    def getDatasetName(self, index: int) -> str:
        datasetName = ''

        if self._velociprobeReader.entryGroup:
            datafile = self._velociprobeReader.entryGroup.data[index]
            datasetName = datafile.name

        return datasetName

    def getDatasetState(self, index: int) -> DatasetState:
        state = DatasetState.MISSING

        if self._velociprobeReader.entryGroup:
            datafile = self._velociprobeReader.entryGroup.data[index]
            state = datafile.getState()

        return state

    def getNumberOfDatasets(self) -> int:
        return 0 if self._velociprobeReader.entryGroup is None \
                else len(self._velociprobeReader.entryGroup.data)

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

    def update(self, observable: Observable) -> None:
        if observable is self._velociprobeReader:
            self.notifyObservers()
