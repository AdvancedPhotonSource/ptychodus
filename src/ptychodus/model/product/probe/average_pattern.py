from __future__ import annotations

import numpy

from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.probe import ProbeEnsemble, ProbeGeometryProvider
from ptychodus.api.simulate.probe import generate_average_pattern_probe

from .builder import ProbeEnsembleBuilder, resolve_defocus_distance_m
from .settings import ProbeSettings


class AveragePatternProbeBuilder(ProbeEnsembleBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ProbeSettings) -> None:
        super().__init__(rng, settings, 'average_pattern')
        self._settings = settings

        # from the focus to the sample: the back-propagated waist is carried this far
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

        self.propagation_distance_m = settings.propagation_distance_m.copy()
        self._add_parameter('propagation_distance_m', self.propagation_distance_m)

        self.bright_field_threshold = settings.average_pattern_bright_field_threshold.copy()
        self._add_parameter('bright_field_threshold', self.bright_field_threshold)

    def copy(self) -> AveragePatternProbeBuilder:
        builder = AveragePatternProbeBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    @property
    def reads_diffraction_data(self) -> bool:
        return True

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeEnsemble:
        assembled_data = geometry_provider.get_assembled_data()
        moments = geometry_provider.get_diffraction_moments()

        if assembled_data is None or moments is None:
            # Not yet bound, or the moments are still being computed; the build is
            # retried once they are available.
            raise GeometryNotDefinedError('No diffraction moments to estimate the probe from.')

        # A flat-phase square root back-propagates to the beam at its waist. The defocus
        # carries it to the object, deferring to the product's focus as the focusing
        # builders do; the propagation distance then refines the plane without moving
        # the object pixel size, which editing the geometry would. Both act on the same
        # field, so they add.
        distance_m = (
            resolve_defocus_distance_m(geometry_provider, self.defocus_distance_m.get_value())
            + self.propagation_distance_m.get_value()
        )
        # The moments' mean frame is already corrected for the background and flat field,
        # and their mask includes the pixels those maps leave unusable.
        probe = generate_average_pattern_probe(
            geometry_provider.get_probe_geometry(),
            moments.mean_frame,
            detector_pixel_geometry=assembled_data.get_pixel_geometry(),
            bad_pixels=moments.bad_pixels,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            detector_distance_m=geometry_provider.detector_distance_m,
            far_field=geometry_provider.far_field,
            defocus_distance_m=distance_m,
            bright_field_mad_threshold=4.5 if self.bright_field_threshold.get_value() else None,
        )
        return self._rescale_to_photon_count(probe, geometry_provider)
