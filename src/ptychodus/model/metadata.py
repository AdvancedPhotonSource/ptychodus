from __future__ import annotations

from ptychodus.api.observer import Observable
from ptychodus.api.patterns import DiffractionMetadata

from .patterns import (
    DetectorSettings,
    DiffractionDatasetObserver,
    AssembledDiffractionDataset,
    PatternSettings,
)
from .product import ProductSettings


class MetadataPresenter(Observable, DiffractionDatasetObserver):
    def __init__(
        self,
        detectorSettings: DetectorSettings,
        patternSettings: PatternSettings,
        diffractionDataset: AssembledDiffractionDataset,
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
        return self._diffractionDataset.get_metadata()

    def canSyncDetectorExtent(self) -> bool:
        return self._metadata.detector_extent is not None

    def syncDetectorExtent(self) -> None:
        detectorExtent = self._metadata.detector_extent

        if detectorExtent:
            self._detectorSettings.width_px.set_value(detectorExtent.width_px)
            self._detectorSettings.height_px.set_value(detectorExtent.height_px)

    def canSyncDetectorPixelSize(self) -> bool:
        return self._metadata.detector_pixel_geometry is not None

    def syncDetectorPixelSize(self) -> None:
        pixelGeometry = self._metadata.detector_pixel_geometry

        if pixelGeometry:
            self._detectorSettings.pixel_width_m.set_value(pixelGeometry.width_m)
            self._detectorSettings.pixel_height_m.set_value(pixelGeometry.height_m)

    def canSyncDetectorBitDepth(self) -> bool:
        return self._metadata.detector_bit_depth is not None

    def syncDetectorBitDepth(self) -> None:
        bitDepth = self._metadata.detector_bit_depth

        if bitDepth:
            self._detectorSettings.bit_depth.set_value(bitDepth)

    def canSyncPatternCropCenter(self) -> bool:
        return self._metadata.crop_center is not None or self._metadata.detector_extent is not None

    def canSyncPatternCropExtent(self) -> bool:
        return self._metadata.detector_extent is not None

    def syncPatternCrop(self, syncCenter: bool, syncExtent: bool) -> None:
        if syncCenter:
            cropCenter = self._metadata.crop_center

            if cropCenter:
                self._patternSettings.cropCenterXInPixels.set_value(cropCenter.position_x_px)
                self._patternSettings.cropCenterYInPixels.set_value(cropCenter.position_y_px)
            elif self._metadata.detector_extent:
                self._patternSettings.cropCenterXInPixels.set_value(
                    int(self._metadata.detector_extent.width_px) // 2
                )
                self._patternSettings.cropCenterYInPixels.set_value(
                    int(self._metadata.detector_extent.height_px) // 2
                )

        if syncExtent and self._metadata.detector_extent:
            centerX = self._patternSettings.cropCenterXInPixels.get_value()
            centerY = self._patternSettings.cropCenterYInPixels.get_value()

            extentX = int(self._metadata.detector_extent.width_px)
            extentY = int(self._metadata.detector_extent.height_px)

            maxRadiusX = min(centerX, extentX - centerX)
            maxRadiusY = min(centerY, extentY - centerY)
            maxRadius = min(maxRadiusX, maxRadiusY)
            cropDiameterInPixels = 1

            while cropDiameterInPixels < maxRadius:
                cropDiameterInPixels <<= 1

            self._patternSettings.cropWidthInPixels.set_value(cropDiameterInPixels)
            self._patternSettings.cropHeightInPixels.set_value(cropDiameterInPixels)

    def canSyncProbePhotonCount(self) -> bool:
        return self._metadata.probe_photon_count is not None

    def syncProbePhotonCount(self) -> None:
        photonCount = self._metadata.probe_photon_count

        if photonCount:
            self._productSettings.probe_photon_count.set_value(photonCount)

    def canSyncProbeEnergy(self) -> bool:
        return self._metadata.probe_energy_eV is not None

    def syncProbeEnergy(self) -> None:
        energyInElectronVolts = self._metadata.probe_energy_eV

        if energyInElectronVolts:
            self._productSettings.probe_energy_eV.set_value(energyInElectronVolts)

    def canSyncDetectorDistance(self) -> bool:
        return self._metadata.detector_distance_m is not None

    def syncDetectorDistance(self) -> None:
        distanceInMeters = self._metadata.detector_distance_m

        if distanceInMeters:
            self._productSettings.detector_distance_m.set_value(distanceInMeters)

    def handle_array_inserted(self, index: int) -> None:
        pass

    def handle_array_changed(self, index: int) -> None:
        pass

    def handle_dataset_reloaded(self) -> None:
        self.notify_observers()
