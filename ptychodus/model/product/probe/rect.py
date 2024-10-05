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
        self._addParameter('width_m', self.widthInMeters)

        self.heightInMeters = settings.rectangleHeightInMeters.copy()
        self._addParameter('height_m', self.heightInMeters)

        # from sample to the focal plane
        self.defocusDistanceInMeters = settings.defocusDistanceInMeters.copy()
        self._addParameter('defocus_distance_m', self.defocusDistanceInMeters)

    def copy(self) -> RectangularProbeBuilder:
        builder = RectangularProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].setValue(value)

        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        aX_m = numpy.fabs(coords.positionXInMeters)
        rx_m = self.widthInMeters.getValue() / 2.0
        aY_m = numpy.fabs(coords.positionYInMeters)
        ry_m = self.heightInMeters.getValue() / 2.0

        isInside = numpy.logical_and(aX_m < rx_m, aY_m < ry_m)
        rect = numpy.where(isInside, 1 + 0j, 0j)

        propagatorParameters = PropagatorParameters(
            wavelength_m=geometryProvider.probeWavelengthInMeters,
            width_px=rect.shape[-1],
            height_px=rect.shape[-2],
            pixel_width_m=geometry.pixelWidthInMeters,
            pixel_height_m=geometry.pixelHeightInMeters,
            propagation_distance_m=self.defocusDistanceInMeters.getValue(),
        )
        propagator = AngularSpectrumPropagator(propagatorParameters)
        array = propagator.propagate(rect)

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
