from __future__ import annotations
from decimal import Decimal, ROUND_CEILING

from ...api.geometry import Interval
from ...api.image import ImageExtent
from ...api.observer import Observable, Observer
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

    @staticmethod
    def _getIntegerCeilingOfQuotient(upper: Decimal, lower: Decimal) -> int:
        return int((upper / lower).to_integral_exact(rounding=ROUND_CEILING))

    def getScanExtent(self) -> ImageExtent:
        extent = ImageExtent(0, 0)
        boundingBoxInMeters = self._scanSizer.getBoundingBoxInMeters()

        if boundingBoxInMeters is not None:
            extentX = self._getIntegerCeilingOfQuotient(
                boundingBoxInMeters.rangeX.length,
                self._apparatus.getObjectPlanePixelSizeXInMeters())
            extentY = self._getIntegerCeilingOfQuotient(
                boundingBoxInMeters.rangeY.length,
                self._apparatus.getObjectPlanePixelSizeYInMeters())
            extent = ImageExtent(width=extentX, height=extentY)

        return extent

    def getProbeExtent(self) -> ImageExtent:
        return 2 * (self._probeSizer.getProbeExtent() // 2)

    def getObjectExtent(self) -> ImageExtent:
        return self.getScanExtent() + self.getProbeExtent()

    def update(self, observable: Observable) -> None:
        if observable is self._apparatus:
            self.notifyObservers()
        elif observable is self._scanSizer:
            self.notifyObservers()
        elif observable is self._probeSizer:
            self.notifyObservers()
