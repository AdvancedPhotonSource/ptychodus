from itertools import pairwise
import logging
import sys

import numpy

from ...api.observer import Observable
from ...api.parametric import ParameterRepository
from ...api.scan import Scan, ScanBoundingBox, ScanPoint
from .boundingBox import ScanBoundingBoxBuilder
from .builder import ScanBuilder
from .transform import ScanPointTransform

logger = logging.getLogger(__name__)


class ScanRepositoryItem(ParameterRepository):

    def __init__(self, builder: ScanBuilder, transform: ScanPointTransform) -> None:
        super().__init__('Scan')
        self._builder = builder
        self._transform = transform

        self._untransformedScan = Scan()
        self._transformedScan = Scan()
        self._boundingBoxBuilder = ScanBoundingBoxBuilder()
        self._lengthInMeters = 0.
        self._sizeInBytes = 0

        self._addParameterRepository(builder, observe=True)
        self._addParameterRepository(transform, observe=True)

        self._rebuild()

    def getUntransformedScan(self) -> Scan:
        return self._untransformedScan

    def getScan(self) -> Scan:
        return self._transformedScan

    def getBuilder(self) -> ScanBuilder:
        return self._builder

    def setBuilder(self, builder: ScanBuilder) -> None:
        self._removeParameterRepository(self._builder)
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addParameterRepository(self._builder)
        self._rebuild()

    def getBoundingBox(self) -> ScanBoundingBox | None:
        return self._boundingBoxBuilder.getBoundingBox()

    def getLengthInMeters(self) -> float:
        return self._lengthInMeters

    def getSizeInBytes(self) -> int:
        return self._sizeInBytes

    def _transformScan(self) -> None:
        transformedPoints: list[ScanPoint] = list()
        boundingBoxBuilder = ScanBoundingBoxBuilder()
        lengthInMeters = 0.

        for untransformedPoint in self._untransformedScan:
            point = self._transform(untransformedPoint)
            transformedPoints.append(point)
            boundingBoxBuilder.hull(point)

        for pointL, pointR in pairwise(transformedPoints):
            dx = pointR.positionXInMeters - pointL.positionXInMeters
            dy = pointR.positionYInMeters - pointL.positionYInMeters
            lengthInMeters += numpy.hypot(dx, dy)

        self._transformedScan = Scan(transformedPoints)
        self._boundingBoxBuilder = boundingBoxBuilder
        self._lengthInMeters = lengthInMeters
        self._sizeInBytes = sys.getsizeof(self._untransformedScan) \
                + sys.getsizeof(self._transformedScan)
        self.notifyObservers()

    def _rebuild(self) -> None:
        try:
            scan = self._builder.build()
        except Exception:
            logger.exception('Failed to reinitialize scan!')
        else:
            self._untransformedScan = scan
            self._transformScan()

    def getTransform(self) -> ScanPointTransform:
        return self._transform

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._transform:
            self._transformScan()
        else:
            super().update(observable)