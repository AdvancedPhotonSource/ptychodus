from __future__ import annotations

import numpy

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.propagator import AngularSpectrumPropagator, PropagatorParameters

from .builder import ProbeBuilder
from .settings import ProbeSettings


class DiskProbeBuilder(ProbeBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'disk')
        self._settings = settings

        self.diameterInMeters = settings.diskDiameterInMeters.copy()
        self._addParameter('diameter_m', self.diameterInMeters)

        # from sample to the focal plane
        self.defocusDistanceInMeters = settings.defocusDistanceInMeters.copy()
        self._addParameter('defocus_distance_m', self.defocusDistanceInMeters)

    def copy(self) -> DiskProbeBuilder:
        builder = DiskProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].setValue(value)

        return builder

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()
        coords = self.getTransverseCoordinates(geometry)

        R_m = coords.positionRInMeters
        r_m = self.diameterInMeters.getValue() / 2.0
        disk = numpy.where(R_m < r_m, 1 + 0j, 0j)

        propagatorParameters = PropagatorParameters(
            wavelength_m=geometryProvider.probeWavelengthInMeters,
            width_px=disk.shape[-1],
            height_px=disk.shape[-2],
            pixel_width_m=geometry.pixelWidthInMeters,
            pixel_height_m=geometry.pixelHeightInMeters,
            propagation_distance_m=self.defocusDistanceInMeters.getValue(),
        )
        propagator = AngularSpectrumPropagator(propagatorParameters)
        array = propagator.propagate(disk)

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
