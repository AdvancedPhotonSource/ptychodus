from __future__ import annotations

import numpy

from ptychodus.api.probe import ProbeEnsemble, ProbeGeometryProvider
from ptychodus.api.simulate.probe import generate_super_gaussian_probe

from .builder import ProbeEnsembleBuilder
from .settings import ProbeSettings


class SuperGaussianProbeBuilder(ProbeEnsembleBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(rng, settings, 'super_gaussian')
        self._settings = settings

        self.annular_radius_m = settings.super_gaussian_annular_radius_m.copy()
        self._add_parameter('annular_radius_m', self.annular_radius_m)

        self.fwhm_m = settings.super_gaussian_width_m.copy()
        self._add_parameter('full_width_at_half_maximum_m', self.fwhm_m)

        self.order_parameter = settings.super_gaussian_order_parameter.copy()
        self._add_parameter('order_parameter', self.order_parameter)

        # from the focus to the sample, setting the curvature of the beam at the profile
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

        # from sample to the focal plane
        self.propagation_distance_m = settings.propagation_distance_m.copy()
        self._add_parameter('propagation_distance_m', self.propagation_distance_m)

    def copy(self) -> SuperGaussianProbeBuilder:
        builder = SuperGaussianProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def can_estimate_size_from_diffraction(self, geometry_provider: ProbeGeometryProvider) -> bool:
        return geometry_provider.get_assembled_data() is not None

    def estimate_size_from_diffraction(self, geometry_provider: ProbeGeometryProvider) -> bool:
        extent = self._estimate_extent(geometry_provider, self.defocus_distance_m.get_value())

        if extent is None:
            return False

        # The second-moment match is to a Gaussian, so only the width is set; the
        # annular radius and order keep whatever shape the user chose.
        self.fwhm_m.set_value(extent.gaussian_fwhm_m)
        return True

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeEnsemble:
        probe = generate_super_gaussian_probe(
            geometry_provider.get_probe_geometry(),
            annular_radius_m=self.annular_radius_m.get_value(),
            fwhm_m=self.fwhm_m.get_value(),
            order_parameter=self.order_parameter.get_value(),
        )
        probe = self._shape_parametric_field(
            probe,
            geometry_provider,
            defocus_distance_m=self.defocus_distance_m.get_value(),
            propagation_distance_m=self.propagation_distance_m.get_value(),
        )
        return self._rescale_to_photon_count(probe, geometry_provider)
