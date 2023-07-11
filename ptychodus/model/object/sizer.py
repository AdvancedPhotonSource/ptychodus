from __future__ import annotations
from decimal import Decimal, ROUND_CEILING

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
        return self._scanSizer.getMidpointInMeters()

    @staticmethod
    def _getIntegerCeilingOfQuotient(upper: Decimal, lower: Decimal) -> int:
        return int((upper / lower).to_integral_exact(rounding=ROUND_CEILING))

    def getScanExtent(self) -> ImageExtent:
        bbox = self._scanSizer.getBoundingBoxInMeters()

        if bbox is None:
            return ImageExtent(width=0, height=0)

        dx = self._apparatus.getObjectPlanePixelSizeXInMeters()
        dy = self._apparatus.getObjectPlanePixelSizeYInMeters()
        extentX = self._getIntegerCeilingOfQuotient(bbox.rangeX.width, dx)
        extentY = self._getIntegerCeilingOfQuotient(bbox.rangeY.width, dy)

        return ImageExtent(width=extentX, height=extentY)

    def getProbeExtent(self) -> ImageExtent:
        return self._probeSizer.getProbeExtent()

    def getObjectExtent(self) -> ImageExtent:
        return self.getScanExtent() + self.getProbeExtent()

    def update(self, observable: Observable) -> None:
        if observable is self._apparatus:
            self.notifyObservers()
        elif observable is self._scanSizer:
            self.notifyObservers()
        elif observable is self._probeSizer:
            self.notifyObservers()
