from __future__ import annotations
import logging

import numpy

from ptychodus.api.observer import Observable
from ptychodus.api.parameters import ParameterGroup
from ptychodus.api.probe_positions import (
    ProbePositionSequence,
    ScanGeometry,
    calculate_scan_geometry,
)

from .builder import FromMemoryProbePositionsBuilder, ProbePositionsBuilder
from .settings import ProbePositionsSettings

logger = logging.getLogger(__name__)


class ProbePositionsRepositoryItem(ParameterGroup):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbePositionsSettings,
        builder: ProbePositionsBuilder,
    ) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._builder = builder
        self._probe_positions = ProbePositionSequence()
        self._geometry: ScanGeometry | None = None

        self._add_group('builder', builder, observe=True)

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

    def assign_item(self, item: ProbePositionsRepositoryItem) -> None:
        self.set_builder(item.get_builder().copy())
        self._rebuild()

    def assign(self, scan: ProbePositionSequence) -> None:
        builder = FromMemoryProbePositionsBuilder(self._rng, self._settings, scan)
        self.set_builder(builder)

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

        self._builder.sync_to_settings()

    def get_probe_positions(self) -> ProbePositionSequence:
        return self._probe_positions

    def get_builder(self) -> ProbePositionsBuilder:
        return self._builder

    def set_builder(self, builder: ProbePositionsBuilder) -> None:
        group = 'builder'
        self._remove_group(group)
        self._builder.remove_observer(self)
        self._builder = builder
        self._builder.add_observer(self)
        self._add_group(group, self._builder, observe=True)
        self._rebuild()

    def get_geometry(self) -> ScanGeometry | None:
        if self.expand_bbox.get_value():
            minimum_x_m = self.expand_bbox_xmin_m.get_value()
            maximum_x_m = self.expand_bbox_xmax_m.get_value()
            minimum_y_m = self.expand_bbox_ymin_m.get_value()
            maximum_y_m = self.expand_bbox_ymax_m.get_value()
            lenth_m = 0.0

            if self._geometry is not None:
                minimum_x_m = min(minimum_x_m, self._geometry.minimum_x_m)
                maximum_x_m = max(maximum_x_m, self._geometry.maximum_x_m)
                minimum_y_m = min(minimum_y_m, self._geometry.minimum_y_m)
                maximum_y_m = max(maximum_y_m, self._geometry.maximum_y_m)
                lenth_m = self._geometry.length_m

            return ScanGeometry(
                minimum_x_m=minimum_x_m,
                maximum_x_m=maximum_x_m,
                minimum_y_m=minimum_y_m,
                maximum_y_m=maximum_y_m,
                length_m=lenth_m,
            )

        return self._geometry

    def _rebuild(self) -> None:
        try:
            probe_positions = self._builder.build()
        except Exception:
            logger.exception('Failed to rebuild scan!')
            return

        # build() always returns a ProbePositionSequence, and the class has no
        # mutators, so there is nothing to defend against by copying. This path
        # runs once per reconstructor iteration; the old round-trip through
        # Python dataclasses was O(N) every time.
        self._probe_positions = probe_positions
        self._geometry = calculate_scan_geometry(probe_positions)
        self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
        else:
            super()._update(observable)
