"""A probe estimated from the measured data: the back-propagated mean diffraction pattern."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ptychodus.api.geometry import GeometryNotDefinedError
from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    ProbeGenerator,
    generate_average_pattern_probe,
)
from ptychodus.api.options import expect_options

from ._options import propagation_distance_field


class AveragePatternOptions(BaseModel):
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
        moments = geometry_provider.get_diffraction_moments()

        if moments is None:
            # Not yet bound, or the moments are still being computed.
            raise GeometryNotDefinedError('No diffraction moments to estimate the probe from.')

        # In far field a flat-phase square root back-propagates to the beam at its waist,
        # and the product's defocus carries it to the object. In near field the hologram is
        # back-propagated over the equivalent distance, which already lands it in the
        # object plane of the equivalent parallel-beam geometry, so the defocus is not
        # applied again. The propagation distance then refines the plane without moving
        # the object pixel size, which editing the geometry would; acting on the same
        # field, it adds.
        defocus_distance_m = (
            geometry_provider.defocus_distance_m if geometry_provider.far_field else 0.0
        )
        distance_m = defocus_distance_m + estimate.propagation_distance_m
        # The moments' mean frame is already corrected for the background and flat field,
        # and their mask includes the pixels those maps leave unusable.
        return generate_average_pattern_probe(
            geometry_provider.get_probe_geometry(),
            moments.mean_pattern,
            detector_pixel_geometry=geometry_provider.get_detector_pixel_geometry(),
            bad_pixels=moments.bad_pixels,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            detector_distance_m=geometry_provider.detector_distance_m,
            far_field=geometry_provider.far_field,
            defocus_distance_m=distance_m,
            bright_field_mad_threshold=4.5 if estimate.bright_field_threshold else None,
        )
