"""Flat-phased apertures: a disk, a rectangle and a super-Gaussian profile.

Each forms an amplitude at the object plane, takes the quadratic phase of a beam sampled
its defocus past the focus, and is then carried by its propagation distance.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    ProbeGenerator,
    apply_defocus_phase,
    generate_disk_probe,
    generate_rectangular_probe,
    generate_super_gaussian_probe,
    propagate_probe,
    resolve_defocus_distance_m,
)

from ._options import defocus_distance_field, expect_options, propagation_distance_field


def _shape_parametric_field(
    probe: Probe,
    geometry_provider: ProbeGeometryProvider,
    *,
    defocus_distance_m: float,
    propagation_distance_m: float,
) -> Probe:
    """Curve a flat-phased amplitude to its defocus, then carry it to another plane.

    The defocus is the curvature of the beam the amplitude models, reconciled with the
    product's focus by `resolve_defocus_distance_m`; the propagation moves the plane
    afterwards. A zero propagation distance skips the propagator, whose FFT round trip
    would otherwise perturb the field at round-off level.
    """
    photon_wavelength_m = geometry_provider.photon_wavelength_m
    probe = apply_defocus_phase(
        probe,
        photon_wavelength_m=photon_wavelength_m,
        defocus_distance_m=resolve_defocus_distance_m(geometry_provider, defocus_distance_m),
    )

    if propagation_distance_m != 0.0:
        probe = propagate_probe(
            probe,
            photon_wavelength_m=photon_wavelength_m,
            propagation_distance_m=propagation_distance_m,
        )

    return probe


class DiskOptions(BaseModel):
    diameter_m: float = Field(
        1e-6, ge=0.0, title='Diameter:', json_schema_extra={'probe_extent': 'disk_diameter_m'}
    )
    defocus_distance_m: float = defocus_distance_field()
    propagation_distance_m: float = propagation_distance_field()


class DiskProbeGenerator(ProbeGenerator):
    """A binary circular aperture."""

    def get_default_options(self) -> BaseModel:
        return DiskOptions()

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        disk = expect_options(options, DiskOptions)
        probe = generate_disk_probe(
            geometry_provider.get_probe_geometry(), radius_m=disk.diameter_m / 2.0
        )
        return _shape_parametric_field(
            probe,
            geometry_provider,
            defocus_distance_m=disk.defocus_distance_m,
            propagation_distance_m=disk.propagation_distance_m,
        )


class RectangularOptions(BaseModel):
    width_m: float = Field(
        1e-6, ge=0.0, title='Width:', json_schema_extra={'probe_extent': 'rectangle_width_m'}
    )
    height_m: float = Field(
        1e-6, ge=0.0, title='Height:', json_schema_extra={'probe_extent': 'rectangle_height_m'}
    )
    defocus_distance_m: float = defocus_distance_field()
    propagation_distance_m: float = propagation_distance_field()


class RectangularProbeGenerator(ProbeGenerator):
    """A binary rectangular aperture."""

    def get_default_options(self) -> BaseModel:
        return RectangularOptions()

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        rectangle = expect_options(options, RectangularOptions)
        probe = generate_rectangular_probe(
            geometry_provider.get_probe_geometry(),
            width_m=rectangle.width_m,
            height_m=rectangle.height_m,
        )
        return _shape_parametric_field(
            probe,
            geometry_provider,
            defocus_distance_m=rectangle.defocus_distance_m,
            propagation_distance_m=rectangle.propagation_distance_m,
        )


class SuperGaussianOptions(BaseModel):
    annular_radius_m: float = Field(
        0.0,
        ge=0.0,
        title='Annular Radius:',
        description='Zero gives a spot; a positive radius gives a ring.',
        json_schema_extra={'display_unit': 'MICROMETER'},
    )
    full_width_at_half_maximum_m: float = Field(
        400e-6,
        ge=0.0,
        title='Full Width at Half Maximum:',
        json_schema_extra={'probe_extent': 'gaussian_fwhm_m'},
    )
    order_parameter: float = Field(1.0, ge=1.0, title='Order Parameter:')
    defocus_distance_m: float = defocus_distance_field()
    propagation_distance_m: float = propagation_distance_field()


class SuperGaussianProbeGenerator(ProbeGenerator):
    """An analytic super-Gaussian amplitude, a spot or a ring."""

    def get_default_options(self) -> BaseModel:
        return SuperGaussianOptions()

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        profile = expect_options(options, SuperGaussianOptions)
        probe = generate_super_gaussian_probe(
            geometry_provider.get_probe_geometry(),
            annular_radius_m=profile.annular_radius_m,
            fwhm_m=profile.full_width_at_half_maximum_m,
            order_parameter=profile.order_parameter,
        )
        return _shape_parametric_field(
            probe,
            geometry_provider,
            defocus_distance_m=profile.defocus_distance_m,
            propagation_distance_m=profile.propagation_distance_m,
        )
