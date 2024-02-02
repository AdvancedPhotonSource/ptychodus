from __future__ import annotations
from decimal import Decimal

from ...api.observer import Observable, Observer
from ...api.patterns import DiffractionDataset, DiffractionMetadata
from ...api.settings import SettingsRegistry
from ..patterns import Detector, DiffractionPatternSettings
from .builder import MetadataBuilder
from .settings import MetadataSettings


class MetadataPresenter(Observable, Observer):

    def __init__(self, diffractionDataset: DiffractionDataset, detector: Detector,
                 patternSettings: DiffractionPatternSettings,
                 metadataSettings: MetadataSettings) -> None:
        super().__init__()
        self._diffractionDataset = diffractionDataset
        self._detector = detector
        self._patternSettings = patternSettings
        self._metadataSettings = metadataSettings

        diffractionDataset.addObserver(self)

    @property
    def _metadata(self) -> DiffractionMetadata:
        return self._diffractionDataset.getMetadata()

    def canSyncDetectorPixelCount(self) -> bool:
        return (self._metadata.detectorExtent is not None)

    def syncDetectorPixelCount(self) -> None:
        detectorExtent = self._metadata.detectorExtent

        if detectorExtent:
            self._detector.setImageExtent(detectorExtent)

    def canSyncDetectorPixelSize(self) -> bool:
        return (self._metadata.detectorPixelGeometry is not None)

    def syncDetectorPixelSize(self) -> None:
        pixelGeometry = self._metadata.detectorPixelGeometry

        if pixelGeometry:
            self._detector.setPixelGeometry(pixelGeometry)

    def canSyncDetectorBitDepth(self) -> bool:
        return (self._metadata.detectorBitDepth is not None)

    def syncDetectorBitDepth(self) -> None:
        bitDepth = self._metadata.detectorBitDepth

        if bitDepth:
            self._detector.setBitDepth(bitDepth)

    def canSyncPatternCropCenter(self) -> bool:
        return (self._metadata.cropCenter is not None \
                or self._metadata.detectorExtent is not None)

    def canSyncPatternCropExtent(self) -> bool:
        return (self._metadata.detectorExtent is not None)

    def syncPatternCrop(self, syncCenter: bool, syncExtent: bool) -> None:
        if syncCenter:
            cropCenter = self._metadata.cropCenter

            if cropCenter:
                self._patternSettings.cropCenterXInPixels.value = cropCenter.positionXInPixels
                self._patternSettings.cropCenterYInPixels.value = cropCenter.positionYInPixels
            elif self._metadata.detectorExtent:
                self._patternSettings.cropCenterXInPixels.value = \
                        int(self._metadata.detectorExtent.widthInPixels) // 2
                self._patternSettings.cropCenterYInPixels.value = \
                        int(self._metadata.detectorExtent.heightInPixels) // 2

        if syncExtent and self._metadata.detectorExtent:
            centerX = self._patternSettings.cropCenterXInPixels.value
            centerY = self._patternSettings.cropCenterYInPixels.value

            extentX = int(self._metadata.detectorExtent.widthInPixels)
            extentY = int(self._metadata.detectorExtent.heightInPixels)

            maxRadiusX = min(centerX, extentX - centerX)
            maxRadiusY = min(centerY, extentY - centerY)
            maxRadius = min(maxRadiusX, maxRadiusY)
            cropDiameterInPixels = 1

            while cropDiameterInPixels < maxRadius:
                cropDiameterInPixels <<= 1

            self._patternSettings.cropWidthInPixels.value = cropDiameterInPixels
            self._patternSettings.cropHeightInPixels.value = cropDiameterInPixels

    def canSyncProbeEnergy(self) -> bool:
        return (self._metadata.probeEnergyInElectronVolts is not None)

    def syncProbeEnergy(self) -> None:
        energyInElectronVolts = self._metadata.probeEnergyInElectronVolts

        if energyInElectronVolts:
            self._metadataSettings.probeEnergyInElectronVolts.value = \
                    Decimal.from_float(energyInElectronVolts)

    def canSyncDetectorDistance(self) -> bool:
        return (self._metadata.detectorDistanceInMeters is not None)

    def syncDetectorDistance(self) -> None:
        distanceInMeters = self._metadata.detectorDistanceInMeters

        if distanceInMeters:
            self._metadataSettings.detectorDistanceInMeters.value = \
                    Decimal.from_float(distanceInMeters)

    def update(self, observable: Observable) -> None:
        if observable is self._diffractionDataset:
            self.notifyObservers()


class MetadataCore:

    def __init__(self, settingsRegistry: SettingsRegistry, diffractionDataset: DiffractionDataset,
                 detector: Detector, patternSettings: DiffractionPatternSettings) -> None:
        self._settings = MetadataSettings.createInstance(settingsRegistry)
        self.builder = MetadataBuilder(self._settings)
        self.presenter = MetadataPresenter(diffractionDataset, detector, patternSettings,
                                           self._settings)
