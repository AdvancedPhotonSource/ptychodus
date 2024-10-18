from __future__ import annotations
from itertools import pairwise
import logging

import numpy

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.scan import Scan, ScanBoundingBox, ScanPoint

from .boundingBox import ScanBoundingBoxBuilder
from .builder import FromMemoryScanBuilder, ScanBuilder
from .settings import ScanSettings
from .transform import ScanPointTransform

logger = logging.getLogger(__name__)


class ScanRepositoryItem(ParameterGroup):
    def __init__(
        self,
        settings: ScanSettings,
        builder: ScanBuilder,
        transform: ScanPointTransform,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._builder = builder
        self._transform = transform

        self._untransformedScan = Scan()
        self._transformedScan = Scan()
        self._boundingBoxBuilder = ScanBoundingBoxBuilder()
        self._lengthInMeters = 0.0

        self._addGroup('builder', builder, observe=True)
        self._addGroup('transform', transform, observe=True)

        self.expandBoundingBox = settings.expandBoundingBox.copy()
        self._addParameter('expand_bbox', self.expandBoundingBox)

        self.expandedBoundingBoxMinimumXInMeters = (
            settings.expandedBoundingBoxMinimumXInMeters.copy()
        )
        self._addParameter('expanded_bbox_xmin_m', self.expandedBoundingBoxMinimumXInMeters)

        self.expandedBoundingBoxMaximumXInMeters = (
            settings.expandedBoundingBoxMaximumXInMeters.copy()
        )
        self._addParameter('expanded_bbox_xmax_m', self.expandedBoundingBoxMaximumXInMeters)

        self.expandedBoundingBoxMinimumYInMeters = (
            settings.expandedBoundingBoxMinimumYInMeters.copy()
        )
        self._addParameter('expanded_bbox_ymin_m', self.expandedBoundingBoxMinimumYInMeters)

        self.expandedBoundingBoxMaximumYInMeters = (
            settings.expandedBoundingBoxMaximumYInMeters.copy()
        )
        self._addParameter('expanded_bbox_ymax_m', self.expandedBoundingBoxMaximumYInMeters)

        self._rebuild()

    def assignItem(self, item: ScanRepositoryItem) -> None:
        self._removeGroup('transform')
        self._transform.removeObserver(self)
        self._transform = item.getTransform().copy()
        self._transform.addObserver(self)
        self._addGroup('transform', self._transform)

        self.setBuilder(item.getBuilder().copy())

    def assign(self, scan: Scan) -> None:
        builder = FromMemoryScanBuilder(self._settings, scan)
        self.setBuilder(builder)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.syncValueToParent()

        self._builder.syncToSettings()
        self._transform.syncToSettings()

    def getScan(self) -> Scan:
        return self._transformedScan

    def getBuilder(self) -> ScanBuilder:
        return self._builder

    def setBuilder(self, builder: ScanBuilder) -> None:
        self._removeGroup('builder')
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addGroup('builder', self._builder)
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
        lengthInMeters = 0.0

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
