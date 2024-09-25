from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import ObjectGeometry
from ptychodus.api.observer import Observable
from ptychodus.api.scan import ScanPoint
from ptychodus.api.visualization import RealArrayType

from ..reconstructor import DiffractionPatternPositionMatcher

__all__ = [
    "STXMSimulator",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class STXMImage:
    intensity: RealArrayType
    pixel_width_m: float
    pixel_height_m: float
    center_x_m: float
    center_y_m: float

    @property
    def pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(self.pixel_width_m, self.pixel_height_m)


class STXMStitcher:
    def __init__(self, geometry: ObjectGeometry) -> None:
        self._geometry = geometry
        self._weights = numpy.zeros((geometry.heightInPixels, geometry.widthInPixels))
        self._intensity = numpy.zeros_like(self._weights)

    def _addPatchPart(
        self, ixSlice: slice, iySlice: slice, intensity: float, probeProfile: RealArrayType
    ) -> None:
        idx = numpy.s_[iySlice, ixSlice]
        self._weights[idx] += probeProfile
        self._intensity[idx] += intensity * probeProfile

    def addMeasurement(
        self, point: ScanPoint, intensity: float, probeProfile: RealArrayType
    ) -> None:
        geometry = self._geometry

        patchWidth = probeProfile.shape[-1]
        patchRadiusXInMeters = geometry.pixelWidthInMeters * patchWidth / 2
        patchMinimumXInMeters = point.positionXInMeters - patchRadiusXInMeters
        ixBeginF, xi = divmod(
            patchMinimumXInMeters - geometry.minimumXInMeters, geometry.pixelWidthInMeters
        )
        ixBegin = int(ixBeginF)
        ixEnd = ixBegin + patchWidth
        ixSlice0 = slice(ixBegin, ixEnd)
        ixSlice1 = slice(ixBegin + 1, ixEnd + 1)

        patchHeight = probeProfile.shape[-2]
        patchRadiusYInMeters = geometry.pixelHeightInMeters * patchHeight / 2
        patchMinimumYInMeters = point.positionYInMeters - patchRadiusYInMeters
        iyBeginF, eta = divmod(
            patchMinimumYInMeters - geometry.minimumYInMeters, geometry.pixelHeightInMeters
        )
        iyBegin = int(iyBeginF)
        iyEnd = iyBegin + patchHeight
        iySlice0 = slice(iyBegin, iyEnd)
        iySlice1 = slice(iyBegin + 1, iyEnd + 1)

        xiC = 1.0 - xi
        etaC = 1.0 - eta

        self._addPatchPart(ixSlice0, iySlice0, xiC * etaC, probeProfile)
        self._addPatchPart(ixSlice1, iySlice0, xi * etaC, probeProfile)
        self._addPatchPart(ixSlice0, iySlice1, xiC * eta, probeProfile)
        self._addPatchPart(ixSlice1, iySlice1, xi * eta, probeProfile)

    def build(self) -> STXMImage:
        intensity = numpy.divide(
            self._intensity,
            self._weights,
            out=numpy.zeros_like(self._weights),
            where=(self._weights > 0),
        )
        return STXMImage(
            intensity=intensity,
            pixel_width_m=self._geometry.pixelWidthInMeters,
            pixel_height_m=self._geometry.pixelHeightInMeters,
            center_x_m=self._geometry.centerXInMeters,
            center_y_m=self._geometry.centerYInMeters,
        )


class STXMSimulator(Observable):
    def __init__(self, dataMatcher: DiffractionPatternPositionMatcher) -> None:
        super().__init__()
        self._dataMatcher = dataMatcher

        self._productIndex = -1
        self._image: STXMImage | None = None

    def setProduct(self, productIndex: int) -> None:
        if self._productIndex != productIndex:
            self._productIndex = productIndex
            self._image = None
            self.notifyObservers()

    def getProductName(self) -> str:
        return self._dataMatcher.getProductName(self._productIndex)

    def simulate(self) -> None:
        reconstructInput = self._dataMatcher.matchDiffractionPatternsWithPositions(
            self._productIndex
        )
        product = reconstructInput.product
        stitcher = STXMStitcher(product.object_.getGeometry())

        probeIntensity = product.probe.getIntensity()
        probeProfile = probeIntensity / numpy.sqrt(numpy.sum(numpy.abs(probeIntensity) ** 2))

        for pattern, scanPoint in zip(reconstructInput.patterns, product.scan):
            patternIntensity = pattern.sum()
            stitcher.addMeasurement(scanPoint, patternIntensity, probeProfile)

        self._image = stitcher.build()
        self.notifyObservers()

    def getImage(self) -> STXMImage:
        if self._image is None:
            raise ValueError("No simulated image!")

        return self._image

    def getSaveFileFilterList(self) -> Sequence[str]:
        return [self.getSaveFileFilter()]

    def getSaveFileFilter(self) -> str:
        return "NumPy Zipped Archive (*.npz)"

    def saveImage(self, filePath: Path) -> None:
        if self._image is None:
            raise ValueError("No simulated image!")

        numpy.savez(
            filePath,
            "pixel_height_m",
            self._image.pixel_height_m,
            "pixel_width_m",
            self._image.pixel_width_m,
            "center_x_m",
            self._image.center_x_m,
            "center_y_m",
            self._image.center_y_m,
            "intensity",
            self._image.intensity,
        )
