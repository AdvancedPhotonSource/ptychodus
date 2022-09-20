from __future__ import annotations
from pathlib import Path

import numpy

from ..api.data import DiffractionArrayState, DiffractionDataset, DiffractionMetadata
from ..api.observer import Observable, Observer
from .data import CropSettings
from .detector import DetectorSettings
from .probe import ProbeSettings
from .scan import ScanCore


class MetadataPresenter(Observable, Observer):

    def __init__(self, diffractionDataset: DiffractionDataset, detectorSettings: DetectorSettings,
                 cropSettings: CropSettings, probeSettings: ProbeSettings,
                 scanCore: ScanCore) -> None:
        super().__init__()
        self._diffractionDataset = diffractionDataset
        self._detectorSettings = detectorSettings
        self._cropSettings = cropSettings
        self._probeSettings = probeSettings
        self._scanCore = scanCore

    @classmethod
    def createInstance(cls, diffractionDataset: DiffractionDataset,
                       detectorSettings: DetectorSettings, cropSettings: CropSettings,
                       probeSettings: ProbeSettings, scanCore: ScanCore) -> MetadataPresenter:
        presenter = cls(diffractionDataset, detectorSettings, cropSettings, probeSettings,
                        scanCore)
        diffractionDataset.addObserver(presenter)
        return presenter

    @property
    def _metadata(self) -> DiffractionMetadata:
        return self._diffractionDataset.getMetadata()

    def syncDetectorPixelCount(self) -> None:
        if self._metadata.detectorNumberOfPixelsX and self._metadata.detectorNumberOfPixelsY:
            self._detectorSettings.numberOfPixelsX.value = \
                self._metadata.detectorNumberOfPixelsX
            self._detectorSettings.numberOfPixelsY.value = \
                self._metadata.detectorNumberOfPixelsY

    def syncDetectorPixelSize(self) -> None:
        if self._metadata.detectorPixelSizeXInMeters and self._metadata.detectorPixelSizeYInMeters:
            self._detectorSettings.pixelSizeXInMeters.value = \
                self._metadata.detectorPixelSizeXInMeters
            self._detectorSettings.pixelSizeYInMeters.value = \
                self._metadata.detectorPixelSizeYInMeters

    def syncDetectorDistance(self) -> None:
        if self._metadata.detectorDistanceInMeters:
            self._detectorSettings.detectorDistanceInMeters.value = \
                self._metadata.detectorDistanceInMeters

    def syncImageCrop(self, syncCenter: bool, syncExtent: bool) -> None:
        if syncCenter and self._metadata.cropCenterXInPixels and self._metadata.cropCenterYInPixels:
            self._cropSettings.centerXInPixels.value = \
                    self._metadata.cropCenterXInPixels
            self._cropSettings.centerYInPixels.value = \
                    self._metadata.cropCenterYInPixels

        if syncExtent and self._metadata.detectorNumberOfPixelsX and self._metadata.detectorNumberOfPixelsY:
            centerX = self._cropSettings.centerXInPixels.value
            centerY = self._cropSettings.centerYInPixels.value

            extentX = int(self._metadata.detectorNumberOfPixelsX)
            extentY = int(self._metadata.detectorNumberOfPixelsY)

            maxRadiusX = min(centerX, extentX - centerX)
            maxRadiusY = min(centerY, extentY - centerY)
            maxRadius = min(maxRadiusX, maxRadiusY)
            cropDiameterInPixels = 1

            while cropDiameterInPixels < maxRadius:
                cropDiameterInPixels <<= 1

            self._cropSettings.extentXInPixels.value = cropDiameterInPixels
            self._cropSettings.extentYInPixels.value = cropDiameterInPixels

    def syncProbeEnergy(self) -> None:
        if self._metadata.probeEnergyInElectronVolts:
            self._probeSettings.probeEnergyInElectronVolts.value = \
                    self._metadata.probeEnergyInElectronVolts

    def loadScanFile(self) -> None:  # TODO velociprobe only
        filePathMaster = self._metadata.filePath
        fileName = filePathMaster.stem.replace('master', 'pos') + '.csv'
        filePath = filePathMaster.parents[2] / 'positions' / fileName
        fileFilter = 'Comma-Separated Values Files (*.csv)'  # TODO refactor; get from somewhere
        self._scanCore.openScan(filePath, fileFilter)

    def update(self, observable: Observable) -> None:
        if observable is self._diffractionDataset:
            self.notifyObservers()
