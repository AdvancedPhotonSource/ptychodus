from __future__ import annotations
from itertools import pairwise
import logging

import numpy

from ptychodus.api.observer import Observable
from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.scan import PositionSequence, ScanBoundingBox, ScanPoint

from .bounding_box import ScanBoundingBoxBuilder
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

        self._untransformedScan = PositionSequence()
        self._transformedScan = PositionSequence()
        self._boundingBoxBuilder = ScanBoundingBoxBuilder()
        self._lengthInMeters = 0.0

        self._add_group('builder', builder, observe=True)
        self._add_group('transform', transform, observe=True)

        self.expandBoundingBox = settings.expandBoundingBox.copy()
        self._add_parameter('expand_bbox', self.expandBoundingBox)

        self.expandedBoundingBoxMinimumXInMeters = (
            settings.expandedBoundingBoxMinimumXInMeters.copy()
        )
        self._add_parameter('expanded_bbox_xmin_m', self.expandedBoundingBoxMinimumXInMeters)

        self.expandedBoundingBoxMaximumXInMeters = (
            settings.expandedBoundingBoxMaximumXInMeters.copy()
        )
        self._add_parameter('expanded_bbox_xmax_m', self.expandedBoundingBoxMaximumXInMeters)

        self.expandedBoundingBoxMinimumYInMeters = (
            settings.expandedBoundingBoxMinimumYInMeters.copy()
        )
        self._add_parameter('expanded_bbox_ymin_m', self.expandedBoundingBoxMinimumYInMeters)

        self.expandedBoundingBoxMaximumYInMeters = (
            settings.expandedBoundingBoxMaximumYInMeters.copy()
        )
        self._add_parameter('expanded_bbox_ymax_m', self.expandedBoundingBoxMaximumYInMeters)

        self._rebuild()

    def assignItem(self, item: ScanRepositoryItem) -> None:
        self._remove_group('transform')
        self._transform.remove_observer(self)
        self._transform = item.getTransform().copy()
        self._transform.add_observer(self)
        self._add_group('transform', self._transform)

        self.setBuilder(item.getBuilder().copy())

    def assign(self, scan: PositionSequence, *, mutable: bool = True) -> None:
        builder = FromMemoryScanBuilder(self._settings, scan)
        self.setBuilder(builder, mutable=mutable)

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.syncToSettings()
        self._transform.syncToSettings()

    def getScan(self) -> PositionSequence:
        return self._transformedScan

    def getBuilder(self) -> ScanBuilder:
        return self._builder

    def setBuilder(self, builder: ScanBuilder, *, mutable: bool = True) -> None:
        self._remove_group('builder')
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group('builder', self._builder)
        self._rebuild(mutable=mutable)

    def getBoundingBox(self) -> ScanBoundingBox | None:
        bbox = self._boundingBoxBuilder.getBoundingBox()

        if self.expandBoundingBox.get_value():
            expandedBoundingBox = ScanBoundingBox(
                minimum_x_m=self.expandedBoundingBoxMinimumXInMeters.get_value(),
                maximum_x_m=self.expandedBoundingBoxMaximumXInMeters.get_value(),
                minimum_y_m=self.expandedBoundingBoxMinimumYInMeters.get_value(),
                maximum_y_m=self.expandedBoundingBoxMaximumYInMeters.get_value(),
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
            dx = pointR.position_x_m - pointL.position_x_m
            dy = pointR.position_y_m - pointL.position_y_m
            lengthInMeters += numpy.hypot(dx, dy)

        self._transformedScan = PositionSequence(transformedPoints)
        self._boundingBoxBuilder = boundingBoxBuilder
        self._lengthInMeters = lengthInMeters
        self.notify_observers()

    def _rebuild(self, *, mutable: bool = True) -> None:
        try:
            scan = self._builder.build()
        except Exception as exc:
            logger.error(''.join(exc.args))
        else:
            self._untransformedScan = scan
            self._transformScan()  # FIXME mutable

    def getTransform(self) -> ScanPointTransform:
        return self._transform

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._transform:
            self._transformScan()
        else:
            super()._update(observable)
