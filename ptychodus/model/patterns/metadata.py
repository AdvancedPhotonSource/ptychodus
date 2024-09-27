from __future__ import annotations
from decimal import Decimal

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.patterns import DiffractionDataset, DiffractionMetadata

from .detector import Detector
from .settings import PatternSettings, ProductSettings


class DiffractionMetadataPresenter(Observable, Observer):
    def __init__(
        self,
        diffractionDataset: DiffractionDataset,
        detector: Detector,
        patternSettings: PatternSettings,
        productSettings: ProductSettings,
    ) -> None:
        super().__init__()
        self._diffractionDataset = diffractionDataset
        self._detector = detector
        self._patternSettings = patternSettings
        self._productSettings = productSettings

        diffractionDataset.addObserver(self)

    @property
    def _metadata(self) -> DiffractionMetadata:
        return self._diffractionDataset.getMetadata()

    def canSyncDetectorPixelCount(self) -> bool:
        return self._metadata.detectorExtent is not None

    def syncDetectorPixelCount(self) -> None:
        detectorExtent = self._metadata.detectorExtent

        if detectorExtent:
            self._detector.setImageExtent(detectorExtent)

    def canSyncDetectorPixelSize(self) -> bool:
        return self._metadata.detectorPixelGeometry is not None

    def syncDetectorPixelSize(self) -> None:
        pixelGeometry = self._metadata.detectorPixelGeometry

        if pixelGeometry:
            self._detector.setPixelGeometry(pixelGeometry)

    def canSyncDetectorBitDepth(self) -> bool:
        return self._metadata.detectorBitDepth is not None

    def syncDetectorBitDepth(self) -> None:
        bitDepth = self._metadata.detectorBitDepth

        if bitDepth:
            self._detector.setBitDepth(bitDepth)

    def canSyncPatternCropCenter(self) -> bool:
        return self._metadata.cropCenter is not None or self._metadata.detectorExtent is not None

    def canSyncPatternCropExtent(self) -> bool:
        return self._metadata.detectorExtent is not None

    def syncPatternCrop(self, syncCenter: bool, syncExtent: bool) -> None:
        if syncCenter:
            cropCenter = self._metadata.cropCenter

            if cropCenter:
                self._patternSettings.cropCenterXInPixels.setValue(cropCenter.positionXInPixels)
                self._patternSettings.cropCenterYInPixels.setValue(cropCenter.positionYInPixels)
            elif self._metadata.detectorExtent:
                self._patternSettings.cropCenterXInPixels.setValue(
                    int(self._metadata.detectorExtent.widthInPixels) // 2
                )
                self._patternSettings.cropCenterYInPixels.setValue(
                    int(self._metadata.detectorExtent.heightInPixels) // 2
                )

        if syncExtent and self._metadata.detectorExtent:
            centerX = self._patternSettings.cropCenterXInPixels.getValue()
            centerY = self._patternSettings.cropCenterYInPixels.getValue()

            extentX = int(self._metadata.detectorExtent.widthInPixels)
            extentY = int(self._metadata.detectorExtent.heightInPixels)

            maxRadiusX = min(centerX, extentX - centerX)
            maxRadiusY = min(centerY, extentY - centerY)
            maxRadius = min(maxRadiusX, maxRadiusY)
            cropDiameterInPixels = 1

            while cropDiameterInPixels < maxRadius:
                cropDiameterInPixels <<= 1

            self._patternSettings.cropWidthInPixels.setValue(cropDiameterInPixels)
            self._patternSettings.cropHeightInPixels.setValue(cropDiameterInPixels)

    def canSyncProbeEnergy(self) -> bool:
        return self._metadata.probeEnergyInElectronVolts is not None

    def syncProbeEnergy(self) -> None:
        energyInElectronVolts = self._metadata.probeEnergyInElectronVolts

        if energyInElectronVolts:
            self._productSettings.probeEnergyInElectronVolts.setValue(
                Decimal.from_float(energyInElectronVolts)
            )

    def canSyncDetectorDistance(self) -> bool:
        return self._metadata.detectorDistanceInMeters is not None

    def syncDetectorDistance(self) -> None:
        distanceInMeters = self._metadata.detectorDistanceInMeters

        if distanceInMeters:
            self._productSettings.detectorDistanceInMeters.setValue(
                Decimal.from_float(distanceInMeters)
            )

    def update(self, observable: Observable) -> None:
        if observable is self._diffractionDataset:
            self.notifyObservers()
