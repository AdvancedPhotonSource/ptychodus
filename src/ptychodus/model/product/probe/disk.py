from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.propagator import AngularSpectrumPropagator, PropagatorParameters

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class DiskProbeBuilder(ProbeSequenceBuilder):
    def __init__(self, settings: ProbeSettings) -> None:
        super().__init__(settings, 'disk')
        self._settings = settings

        self.diameter_m = settings.disk_diameter_m.copy()
        self._add_parameter('diameter_m', self.diameter_m)

        # from sample to the focal plane
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

    def copy(self) -> DiskProbeBuilder:
        builder = DiskProbeBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        geometry = geometry_provider.get_probe_geometry()
        coords = self.get_transverse_coordinates(geometry)

        R_m = coords.position_r_m  # noqa: N806
        r_m = self.diameter_m.get_value() / 2.0
        disk = numpy.where(R_m < r_m, 1 + 0j, 0j)

        propagator_parameters = PropagatorParameters(
            wavelength_m=geometry_provider.probe_wavelength_m,
            width_px=disk.shape[-1],
            height_px=disk.shape[-2],
            pixel_width_m=geometry.pixel_width_m,
            pixel_height_m=geometry.pixel_height_m,
            propagation_distance_m=self.defocus_distance_m.get_value(),
        )
        propagator = AngularSpectrumPropagator(propagator_parameters)
        array = propagator.propagate(disk)

        return ProbeSequence(
            array=self.normalize(array),
            opr_weights=None,
            pixel_geometry=geometry.get_pixel_geometry(),
        )
