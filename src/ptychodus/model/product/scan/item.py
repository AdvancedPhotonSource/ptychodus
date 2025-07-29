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
        self._untransformed_scan = PositionSequence()
        self._transformed_scan = PositionSequence()
        self._bbox_builder = ScanBoundingBoxBuilder()
        self._length_m = 0.0

        self._add_group('builder', builder, observe=True)
        self._add_group('transform', transform, observe=True)

        self.expand_bbox = settings.expand_bbox.copy()
        self._add_parameter('expand_bbox', self.expand_bbox)

        self.expand_bbox_xmin_m = settings.expand_bbox_xmin_m.copy()
        self._add_parameter('expand_bbox_xmin_m', self.expand_bbox_xmin_m)

        self.expand_bbox_xmax_m = settings.expand_bbox_xmax_m.copy()
        self._add_parameter('expand_bbox_xmax_m', self.expand_bbox_xmax_m)

        self.expand_bbox_ymin_m = settings.expand_bbox_ymin_m.copy()
        self._add_parameter('expand_bbox_ymin_m', self.expand_bbox_ymin_m)

        self.expand_bbox_ymax_m = settings.expand_bbox_ymax_m.copy()
        self._add_parameter('expand_bbox_ymax_m', self.expand_bbox_ymax_m)

        self._rebuild()

    def assign_item(self, item: ScanRepositoryItem) -> None:
        group = 'transform'

        self._remove_group(group)
        self._transform.remove_observer(self)

        transform = item.get_transform()

        self._transform = transform.copy()
        self._transform.add_observer(self)
        self._add_group(group, self._transform, observe=True)

        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, scan: PositionSequence) -> None:
        builder = FromMemoryScanBuilder(self._settings, scan)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()
        self._transform.sync_to_settings()

    def get_scan(self) -> PositionSequence:
        return self._transformed_scan

    def get_builder(self) -> ScanBuilder:
        return self._builder

    def set_builder(self, builder: ScanBuilder) -> None:
        group = 'builder'
        self._remove_group(group)
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group(group, self._builder, observe=True)
        self._rebuild()

    def get_bounding_box(self) -> ScanBoundingBox | None:
        bbox = self._bbox_builder.get_bounding_box()

        if self.expand_bbox.get_value():
            expanded_bbox = ScanBoundingBox(
                minimum_x_m=self.expand_bbox_xmin_m.get_value(),
                maximum_x_m=self.expand_bbox_xmax_m.get_value(),
                minimum_y_m=self.expand_bbox_ymin_m.get_value(),
                maximum_y_m=self.expand_bbox_ymax_m.get_value(),
            )
            bbox = expanded_bbox if bbox is None else bbox.hull(expanded_bbox)

        return bbox

    def get_length_m(self) -> float:
        return self._length_m

    def _transform_scan(self) -> None:
        transformed_points: list[ScanPoint] = list()
        bbox_builder = ScanBoundingBoxBuilder()
        length_m = 0.0

        for untransformed_point in self._untransformed_scan:
            transformed_point = self._transform(untransformed_point)
            transformed_points.append(transformed_point)
            bbox_builder.hull(transformed_point)

        for point_l, point_r in pairwise(transformed_points):
            dx = point_r.position_x_m - point_l.position_x_m
            dy = point_r.position_y_m - point_l.position_y_m
            length_m += numpy.hypot(dx, dy)

        self._transformed_scan = PositionSequence(transformed_points)
        self._bbox_builder = bbox_builder
        self._length_m = length_m
        self.notify_observers()

    def _rebuild(self) -> None:
        try:
            scan = self._builder.build()
        except Exception:
            logger.exception('Failed to rebuild scan!')
            return

        self._untransformed_scan = scan
        self._transform_scan()

    def get_transform(self) -> ScanPointTransform:
        return self._transform

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        elif observable is self._transform:
            self._transform_scan()
        else:
            super()._update(observable)
