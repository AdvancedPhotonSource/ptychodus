from __future__ import annotations
import logging

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.simulate.object import generate_dead_leaves_object

from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class DeadLeavesObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'dead_leaves')
        self._rng = rng
        self._settings = settings

        self.leaf_radius_lower_px = settings.leaf_radius_lower_px.copy()
        self._add_parameter('leaf_radius_lower_px', self.leaf_radius_lower_px)
        self.leaf_radius_upper_px = settings.leaf_radius_upper_px.copy()
        self._add_parameter('leaf_radius_upper_px', self.leaf_radius_upper_px)
        self.leaf_radius_power_law_exponent = settings.leaf_radius_power_law_exponent.copy()
        self._add_parameter('leaf_radius_power_law_exponent', self.leaf_radius_power_law_exponent)

        self.leaf_amplitude_lower = settings.leaf_amplitude_lower.copy()
        self._add_parameter('leaf_amplitude_lower', self.leaf_amplitude_lower)
        self.leaf_amplitude_upper = settings.leaf_amplitude_upper.copy()
        self._add_parameter('leaf_amplitude_upper', self.leaf_amplitude_upper)

        self.leaf_phase_lower_tr = settings.leaf_phase_lower_tr.copy()
        self._add_parameter('leaf_phase_lower_tr', self.leaf_phase_lower_tr)
        self.leaf_phase_upper_tr = settings.leaf_phase_upper_tr.copy()
        self._add_parameter('leaf_phase_upper_tr', self.leaf_phase_upper_tr)

    def copy(self) -> DeadLeavesObjectBuilder:
        builder = DeadLeavesObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        object_ = generate_dead_leaves_object(
            self._rng,
            geometry_provider.get_object_geometry(),
            leaf_radius_lower_px=self.leaf_radius_lower_px.get_value(),
            leaf_radius_upper_px=self.leaf_radius_upper_px.get_value(),
            leaf_radius_power_law_exponent=self.leaf_radius_power_law_exponent.get_value(),
            leaf_amplitude_lower=self.leaf_amplitude_lower.get_value(),
            leaf_amplitude_upper=self.leaf_amplitude_upper.get_value(),
            leaf_phase_lower_tr=self.leaf_phase_lower_tr.get_value(),
            leaf_phase_upper_tr=self.leaf_phase_upper_tr.get_value(),
        )
        return self._pad_object(object_)
