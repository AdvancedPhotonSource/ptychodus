from __future__ import annotations
from itertools import pairwise
import logging

import numpy

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterRepository
from ptychodus.api.scan import Scan, ScanBoundingBox, ScanPoint

from .boundingBox import ScanBoundingBoxBuilder
from .builder import ScanBuilder
from .settings import ScanSettings
from .transform import ScanPointTransform

logger = logging.getLogger(__name__)


class ScanRepositoryItem(ParameterRepository):

    def __init__(self, settings: ScanSettings, builder: ScanBuilder,
                 transform: ScanPointTransform) -> None:
        super().__init__('Scan')
        self._builder = builder
        self._transform = transform

        self._untransformedScan = Scan()
        self._transformedScan = Scan()
        self._boundingBoxBuilder = ScanBoundingBoxBuilder()
        self._lengthInMeters = 0.

        self._addParameterRepository(builder, observe=True)
        self._addParameterRepository(transform, observe=True)

        self.expandBoundingBox = self._registerBooleanParameter('expand_bbox',
                                                                settings.expandBoundingBox.value)
        self.expandedBoundingBoxMinimumXInMeters = self._registerRealParameter(
            'expanded_bbox_xmin_m', float(settings.expandedBoundingBoxMinimumXInMeters.value))
        self.expandedBoundingBoxMaximumXInMeters = self._registerRealParameter(
            'expanded_bbox_xmax_m', float(settings.expandedBoundingBoxMaximumXInMeters.value))
        self.expandedBoundingBoxMinimumYInMeters = self._registerRealParameter(
            'expanded_bbox_ymin_m', float(settings.expandedBoundingBoxMinimumYInMeters.value))
        self.expandedBoundingBoxMaximumYInMeters = self._registerRealParameter(
            'expanded_bbox_ymax_m', float(settings.expandedBoundingBoxMaximumYInMeters.value))

        self._rebuild()

    def assign(self, item: ScanRepositoryItem) -> None:
        self._removeParameterRepository(self._transform)
        self._transform.removeObserver(self)
        self._transform = item.getTransform().copy()
        self._transform.addObserver(self)
        self._addParameterRepository(self._transform)

        self.setBuilder(item.getBuilder().copy())

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
        bbox = self._boundingBoxBuilder.getBoundingBox()

        if self.expandBoundingBox.getValue():
            expandedBoundingBox = ScanBoundingBox(
                minimumXInMeters=self.expandedBoundingBoxMinimumXInMeters.getValue(),
                maximumXInMeters=self.expandedBoundingBoxMaximumXInMeters.getValue(),
                minimumYInMeters=self.expandedBoundingBoxMinimumYInMeters.getValue(),
                maximumYInMeters=self.expandedBoundingBoxMaximumYInMeters.getValue(),
            )
            bbox = expandedBoundingBox if bbox is None else bbox.hull(expandedBoundingBox)

        return bbox

    def getLengthInMeters(self) -> float:
        return self._lengthInMeters

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
        self.notifyObservers()

    def _rebuild(self) -> None:
        try:
            scan = self._builder.build()
        except Exception as exc:
            logger.error(''.join(exc.args))
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
