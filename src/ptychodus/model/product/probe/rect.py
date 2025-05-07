from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.propagator import AngularSpectrumPropagator, PropagatorParameters

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class RectangularProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'rectangular')
        self._settings = settings

        self.width_m = settings.rectangle_width_m.copy()
        self._add_parameter('width_m', self.width_m)

        self.height_m = settings.rectangle_height_m.copy()
        self._add_parameter('height_m', self.height_m)

        # from sample to the focal plane
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

    def copy(self) -> RectangularProbeBuilder:
        builder = RectangularProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        geometry = geometry_provider.get_probe_geometry()
        coords = self.get_transverse_coordinates(geometry)

        aX_m = numpy.fabs(coords.position_x_m)  # noqa: N806
        rx_m = self.width_m.get_value() / 2.0
        aY_m = numpy.fabs(coords.position_y_m)  # noqa: N806
        ry_m = self.height_m.get_value() / 2.0

        is_inside = numpy.logical_and(aX_m < rx_m, aY_m < ry_m)
        rect = numpy.where(is_inside, 1 + 0j, 0j)

        propagator_parameters = PropagatorParameters(
            wavelength_m=geometry_provider.probe_wavelength_m,
            width_px=rect.shape[-1],
            height_px=rect.shape[-2],
            pixel_width_m=geometry.pixel_width_m,
            pixel_height_m=geometry.pixel_height_m,
            propagation_distance_m=self.defocus_distance_m.get_value(),
        )
        propagator = AngularSpectrumPropagator(propagator_parameters)
        array = propagator.propagate(rect)

        return ProbeSequence(
            array=self.normalize(array),
            opr_weights=None,
            pixel_geometry=geometry.get_pixel_geometry(),
        )
