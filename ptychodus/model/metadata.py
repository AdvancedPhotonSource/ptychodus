from typing import Tuple

from .detector import DetectorSettings, CropSettings
from .observer import Observable, Observer
from .probe import ProbeSettings
from .settings import SettingsGroup
from .velociprobe import VelociprobeReader, DetectorGroup, DetectorSpecificGroup


class ImportSettingsPresenter(Observable, Observer
                              ):  # TODO refactor to set using presenters rather than settings
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

    @property
    def _detectorGroup(self) -> DetectorGroup:
        return self._velociprobeReader.entryGroup.instrument.detector

    @property
    def _detectorSpecificGroup(self) -> DetectorSpecificGroup:
        return self._detectorGroup.detectorSpecific

    def syncDetectorPixelSize(self) -> None:
        self._detectorSettings.pixelSizeXInMeters.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorGroup.x_pixel_size_m)
        self._detectorSettings.pixelSizeYInMeters.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorGroup.y_pixel_size_m)

    def syncDetectorDistance(self) -> None:
        self._detectorSettings.detectorDistanceInMeters.value = \
                SettingsGroup.convertFloatToDecimal(self._detectorGroup.detector_distance_m)

    def syncImageCropCenter(self) -> None:
        self._cropSettings.centerXInPixels.value = \
                int(round(self._detectorGroup.beam_center_x_px))
        self._cropSettings.centerYInPixels.value = \
                int(round(self._detectorGroup.beam_center_y_px))

    def syncImageCropExtent(self) -> None:
        centerX = self._cropSettings.centerXInPixels.value
        centerY = self._cropSettings.centerYInPixels.value

        extentX = self._detectorGroup.module.x_data_size
        extentY = self._detectorGroup.module.y_data_size

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
