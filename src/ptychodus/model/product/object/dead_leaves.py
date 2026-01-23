from __future__ import annotations
from collections.abc import Sequence
import logging

import numpy

from ptychodus.api.common import RealArrayType, TWO_PI_J, lerp
from ptychodus.api.object import Object, ObjectGeometryProvider

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

        self.leaf_phase_lower_turns = settings.leaf_phase_lower_turns.copy()
        self._add_parameter('leaf_phase_lower_turns', self.leaf_phase_lower_turns)
        self.leaf_phase_upper_turns = settings.leaf_phase_upper_turns.copy()
        self._add_parameter('leaf_phase_upper_turns', self.leaf_phase_upper_turns)

        # TODO consider using Poisson disk sampling or CVT for sample positions
        # TODO include pixel aspect ratio in sample position calculation
        self._sample_positions = [
            (1.0 / 3, 1.0 / 3),
            (1.0 / 3, 2.0 / 3),
            (2.0 / 3, 1.0 / 3),
            (2.0 / 3, 2.0 / 3),
        ]

    def copy(self) -> DeadLeavesObjectBuilder:
        builder = DeadLeavesObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _sample_leaf_radius_px(self) -> float:
        rmin_px = self.leaf_radius_lower_px.get_value()
        rmax_px = self.leaf_radius_upper_px.get_value()
        beta = 1.0 - self.leaf_radius_power_law_exponent.get_value()
        coef = numpy.power(rmax_px / rmin_px, beta) - 1.0
        return rmin_px * numpy.power(1.0 + self._rng.uniform() * coef, 1.0 / beta)

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        if self.leaf_radius_lower_px.get_value() >= self.leaf_radius_upper_px.get_value():
            raise ValueError('leaf_radius_lower_px must be less than leaf_radius_upper_px')

        if self.leaf_amplitude_lower.get_value() >= self.leaf_amplitude_upper.get_value():
            raise ValueError('leaf_amplitude_lower must be less than leaf_amplitude_upper')

        if self.leaf_phase_lower_turns.get_value() >= self.leaf_phase_upper_turns.get_value():
            raise ValueError('leaf_phase_lower_turns must be less than leaf_phase_upper_turns')

        geometry = geometry_provider.get_object_geometry()
        object_shape = (geometry.height_px, geometry.width_px)
        is_covered = numpy.zeros(object_shape, dtype=bool)
        amplitude: RealArrayType = numpy.zeros(object_shape, dtype=float)
        phase_turns: RealArrayType = numpy.zeros(object_shape, dtype=float)

        position_y_px, position_x_px = numpy.indices(object_shape)
        num_covered_pixels = 0

        for leaf in range(100000):  # large value eliminates possiblity of infinite loop
            leaf_radius_px = self._sample_leaf_radius_px()
            leaf_position_x_px = self._rng.uniform(
                -leaf_radius_px, geometry.width_px + leaf_radius_px
            )
            leaf_position_y_px = self._rng.uniform(
                -leaf_radius_px, geometry.height_px + leaf_radius_px
            )
            leaf_phase_turns = self._rng.uniform(
                self.leaf_phase_lower_turns.get_value(),
                self.leaf_phase_upper_turns.get_value(),
                size=(1, 1),
            )
            leaf_amplitude = numpy.sqrt(
                self._rng.uniform(
                    self.leaf_amplitude_lower.get_value() ** 2,
                    self.leaf_amplitude_upper.get_value() ** 2,
                    size=(1, 1),
                )
            )

            leaf_counts = numpy.zeros_like(is_covered, dtype=int)
            dx = position_x_px - leaf_position_x_px
            dy = position_y_px - leaf_position_y_px

            # multi-sample anti-aliasing
            for u_y, u_x in self._sample_positions:
                sample_distance_px = numpy.hypot(dy + u_y, dx + u_x)
                leaf_counts[sample_distance_px <= leaf_radius_px] += 1

            num_samples = len(self._sample_positions)
            leaf_coverage = leaf_counts / num_samples
            amplitude = lerp(amplitude, leaf_amplitude, leaf_coverage)
            phase_turns = lerp(phase_turns, leaf_phase_turns, leaf_coverage)

            is_covered |= leaf_counts == num_samples
            num_covered_pixels = numpy.count_nonzero(is_covered).item()

            covered_pct = 100 * num_covered_pixels / is_covered.size
            logger.info(f'leaves = {leaf}, covered = {covered_pct:.2f}%')

            if num_covered_pixels == is_covered.size:
                break

        array = amplitude * numpy.exp(TWO_PI_J * phase_turns)

        return self._create_object(
            array=array.astype('complex'),
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
