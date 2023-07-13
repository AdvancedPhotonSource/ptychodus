from __future__ import annotations
from decimal import Decimal

from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
from ...api.scan import ScanPoint
from ..probe import Apparatus, ProbeSizer
from ..scan import ScanSizer
from .settings import ObjectSettings


class ObjectSizer(Observable, Observer):

    def __init__(self, settings: ObjectSettings, apparatus: Apparatus, scanSizer: ScanSizer,
                 probeSizer: ProbeSizer) -> None:
        super().__init__()
        self._settings = settings
        self._apparatus = apparatus
        self._scanSizer = scanSizer
        self._probeSizer = probeSizer

    @classmethod
    def createInstance(cls, settings: ObjectSettings, apparatus: Apparatus, scanSizer: ScanSizer,
                       probeSizer: ProbeSizer) -> ObjectSizer:
        sizer = cls(settings, apparatus, scanSizer, probeSizer)
        apparatus.addObserver(sizer)
        scanSizer.addObserver(sizer)
        probeSizer.addObserver(sizer)
        return sizer

    def getPixelSizeXInMeters(self) -> Decimal:
        return self._apparatus.getObjectPlanePixelSizeXInMeters()

    def getPixelSizeYInMeters(self) -> Decimal:
        return self._apparatus.getObjectPlanePixelSizeYInMeters()

    def getMidpointInMeters(self) -> ScanPoint:
        bbox = self._scanSizer.getBoundingBoxInMeters()
        return ScanPoint(0., 0.) if bbox is None else bbox.midpoint

    @staticmethod
    def _getExtentInPixels(center: float, width: float, pixelSize: float) -> int:
        radius = width / 2
        lowerPx = +int((center - radius) / +pixelSize)
        upperPx = -int((center + radius) / -pixelSize)
        radiusPx = (upperPx - lowerPx + 1) // 2
        return 2 * radiusPx

    def getScanExtentInPixels(self) -> ImageExtent:
        bbox = self._scanSizer.getBoundingBoxInMeters()

        if bbox is None:
            return ImageExtent(0, 0)

        midpoint = bbox.midpoint

        return ImageExtent(
            width=self._getExtentInPixels(midpoint.x, bbox.rangeX.width,
                                          float(self.getPixelSizeXInMeters())),
            height=self._getExtentInPixels(midpoint.y, bbox.rangeY.width,
                                           float(self.getPixelSizeYInMeters())),
        )

        return ImageExtent(width=bbox.rangeX.width, height=bbox.rangeY.width)

    def getProbeExtentInPixels(self) -> ImageExtent:
        return self._probeSizer.getExtentInPixels()

    def getObjectExtentInPixels(self) -> ImageExtent:
        return self.getScanExtentInPixels() + self.getProbeExtentInPixels()

    def update(self, observable: Observable) -> None:
        if observable is self._apparatus:
            self.notifyObservers()
        elif observable is self._scanSizer:
            self.notifyObservers()
        elif observable is self._probeSizer:
            self.notifyObservers()
