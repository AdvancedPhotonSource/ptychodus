from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeEnsemble, ProbeGeometryProvider
from ptychodus.api.simulate.probe import generate_disk_probe

from .builder import ProbeEnsembleBuilder
from .settings import ProbeSettings


class DiskProbeBuilder(ProbeEnsembleBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(rng, settings, 'disk')
        self._settings = settings

        self.diameter_m = settings.disk_diameter_m.copy()
        self._add_parameter('diameter_m', self.diameter_m)

        # from the focus to the sample, setting the curvature of the beam at the aperture
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

        # from sample to the focal plane
        self.propagation_distance_m = settings.propagation_distance_m.copy()
        self._add_parameter('propagation_distance_m', self.propagation_distance_m)

    def copy(self) -> DiskProbeBuilder:
        builder = DiskProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def can_estimate_size_from_diffraction(self, geometry_provider: ProbeGeometryProvider) -> bool:
        return geometry_provider.get_assembled_data() is not None

    def estimate_size_from_diffraction(self, geometry_provider: ProbeGeometryProvider) -> bool:
        extent = self._estimate_extent(geometry_provider, self.defocus_distance_m.get_value())

        if extent is None:
            return False

        self.diameter_m.set_value(extent.disk_diameter_m)
        return True

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeEnsemble:
        probe = generate_disk_probe(
            geometry_provider.get_probe_geometry(),
            radius_m=self.diameter_m.get_value() / 2.0,
        )
        probe = self._shape_parametric_field(
            probe,
            geometry_provider,
            defocus_distance_m=self.defocus_distance_m.get_value(),
            propagation_distance_m=self.propagation_distance_m.get_value(),
        )
        return self._rescale_to_photon_count(probe, geometry_provider)
