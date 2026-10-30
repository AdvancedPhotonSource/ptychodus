"""A probe estimated from the measured data: the back-propagated mean diffraction pattern."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    ProbeGenerator,
    generate_average_pattern_probe,
    resolve_defocus_distance_m,
)

from ._options import defocus_distance_field, expect_options, propagation_distance_field


class AveragePatternOptions(BaseModel):
    defocus_distance_m: float = defocus_distance_field()
    propagation_distance_m: float = propagation_distance_field()
    bright_field_threshold: bool = Field(
        False,
        title='Bright Field Only',
        description=(
            'Remove the background pedestal and the noise around the bright field before '
            'back-propagating. Far field only.'
        ),
    )


class AveragePatternProbeGenerator(ProbeGenerator):
    """The square root of the mean pattern, back-propagated to the object plane."""

    def get_default_options(self) -> BaseModel:
        return AveragePatternOptions()

    @property
    def reads_diffraction_data(self) -> bool:
        return True

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        estimate = expect_options(options, AveragePatternOptions)
        assembled_data = geometry_provider.get_assembled_data()
        moments = geometry_provider.get_diffraction_moments()

        if assembled_data is None or moments is None:
            # Not yet bound, or the moments are still being computed.
            raise GeometryNotDefinedError('No diffraction moments to estimate the probe from.')

        # A flat-phase square root back-propagates to the beam at its waist. The defocus
        # carries it to the object, deferring to the product's focus as the focusing
        # generators do; the propagation distance then refines the plane without moving
        # the object pixel size, which editing the geometry would. Both act on the same
        # field, so they add.
        distance_m = (
            resolve_defocus_distance_m(geometry_provider, estimate.defocus_distance_m)
            + estimate.propagation_distance_m
        )
        # The moments' mean frame is already corrected for the background and flat field,
        # and their mask includes the pixels those maps leave unusable.
        return generate_average_pattern_probe(
            geometry_provider.get_probe_geometry(),
            moments.mean_frame,
            detector_pixel_geometry=assembled_data.get_pixel_geometry(),
            bad_pixels=moments.bad_pixels,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            detector_distance_m=geometry_provider.detector_distance_m,
            far_field=geometry_provider.far_field,
            defocus_distance_m=distance_m,
            bright_field_mad_threshold=4.5 if estimate.bright_field_threshold else None,
        )
