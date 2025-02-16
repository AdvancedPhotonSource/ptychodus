from __future__ import annotations

from ptychodus.api.observer import Observable
from ptychodus.api.patterns import DiffractionMetadata

from .patterns import (
    DetectorSettings,
    DiffractionDatasetObserver,
    ObservableDiffractionDataset,
    PatternSettings,
)
from .product import ProductSettings


class MetadataPresenter(Observable, DiffractionDatasetObserver):
    def __init__(
        self,
        detectorSettings: DetectorSettings,
        patternSettings: PatternSettings,
        diffractionDataset: ObservableDiffractionDataset,
        productSettings: ProductSettings,
    ) -> None:
        super().__init__()
        self._detectorSettings = detectorSettings
        self._patternSettings = patternSettings
        self._diffractionDataset = diffractionDataset
        self._productSettings = productSettings

        diffractionDataset.add_observer(self)

    @property
    def _metadata(self) -> DiffractionMetadata:
        return self._diffractionDataset.getMetadata()

    def canSyncDetectorPixelCount(self) -> bool:
        return self._metadata.detectorExtent is not None

    def syncDetectorPixelCount(self) -> None:
        detectorExtent = self._metadata.detectorExtent

        if detectorExtent:
            self._detectorSettings.widthInPixels.setValue(detectorExtent.widthInPixels)
            self._detectorSettings.heightInPixels.setValue(detectorExtent.heightInPixels)

    def canSyncDetectorPixelSize(self) -> bool:
        return self._metadata.detectorPixelGeometry is not None

    def syncDetectorPixelSize(self) -> None:
        pixelGeometry = self._metadata.detectorPixelGeometry

        if pixelGeometry:
            self._detectorSettings.pixelWidthInMeters.setValue(pixelGeometry.widthInMeters)
            self._detectorSettings.pixelHeightInMeters.setValue(pixelGeometry.heightInMeters)

    def canSyncDetectorBitDepth(self) -> bool:
        return self._metadata.detectorBitDepth is not None

    def syncDetectorBitDepth(self) -> None:
        bitDepth = self._metadata.detectorBitDepth

        if bitDepth:
            self._detectorSettings.bitDepth.setValue(bitDepth)

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
            self._productSettings.probeEnergyInElectronVolts.setValue(energyInElectronVolts)

    def canSyncDetectorDistance(self) -> bool:
        return self._metadata.detectorDistanceInMeters is not None

    def syncDetectorDistance(self) -> None:
        distanceInMeters = self._metadata.detectorDistanceInMeters

        if distanceInMeters:
            self._productSettings.detectorDistanceInMeters.setValue(distanceInMeters)

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self.notifyObservers()
