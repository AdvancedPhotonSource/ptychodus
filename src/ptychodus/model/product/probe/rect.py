from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.propagator import AngularSpectrumPropagator, PropagatorParameters

from .builder import ProbeBuilder
from .settings import ProbeSettings


class RectangularProbeBuilder(ProbeBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'rectangular')
        self._settings = settings

        self.widthInMeters = settings.rectangleWidthInMeters.copy()
        self._add_parameter('width_m', self.widthInMeters)

        self.heightInMeters = settings.rectangleHeightInMeters.copy()
        self._add_parameter('height_m', self.heightInMeters)

        # from sample to the focal plane
        self.defocusDistanceInMeters = settings.defocusDistanceInMeters.copy()
        self._add_parameter('defocus_distance_m', self.defocusDistanceInMeters)

    def copy(self) -> RectangularProbeBuilder:
        builder = RectangularProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.get_probe_geometry()
        coords = self.get_transverse_coordinates(geometry)

        aX_m = numpy.fabs(coords.position_x_m)
        rx_m = self.widthInMeters.get_value() / 2.0
        aY_m = numpy.fabs(coords.position_y_m)
        ry_m = self.heightInMeters.get_value() / 2.0

        isInside = numpy.logical_and(aX_m < rx_m, aY_m < ry_m)
        rect = numpy.where(isInside, 1 + 0j, 0j)

        propagatorParameters = PropagatorParameters(
            wavelength_m=geometryProvider.probe_wavelength_m,
            width_px=rect.shape[-1],
            height_px=rect.shape[-2],
            pixel_width_m=geometry.pixel_width_m,
            pixel_height_m=geometry.pixel_height_m,
            propagation_distance_m=self.defocusDistanceInMeters.get_value(),
        )
        propagator = AngularSpectrumPropagator(propagatorParameters)
        array = propagator.propagate(rect)

        return Probe(
            array=self.normalize(array),
            pixel_geometry=geometry.get_pixel_geometry(),
        )
