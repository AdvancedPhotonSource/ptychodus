from __future__ import annotations
from decimal import Decimal

import numpy

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

    def getScanExtentInPixels(self) -> ImageExtent:
        bbox = self._scanSizer.getBoundingBoxInMeters()
        extentX = 0
        extentY = 0

        if bbox is not None:
            extentX = int(numpy.ceil(bbox.rangeX.width / float(self.getPixelSizeXInMeters())))
            extentY = int(numpy.ceil(bbox.rangeY.width / float(self.getPixelSizeYInMeters())))

        return ImageExtent(width=extentX, height=extentY)

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
