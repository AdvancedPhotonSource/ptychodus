"""Focusing optics: a Fresnel zone plate and a Kirkpatrick-Baez mirror pair.

Each simulates the optic and carries its field to an object sampled its defocus past the
focus, deferring to the product's focus when it states one.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    KirkpatrickBaezMirror,
    KirkpatrickBaezMirrorPair,
    ProbeGenerator,
    generate_fresnel_zone_plate_probe,
    generate_kb_mirror_probe,
    resolve_defocus_distance_m,
)

from ._options import defocus_distance_field, expect_options


class FresnelZonePlateOptions(BaseModel):
    zone_plate_diameter_m: float = Field(180e-6, ge=0.0, title='Zone Plate Diameter:')
    outermost_zone_width_m: float = Field(50e-9, ge=0.0, title='Outermost Zone Width:')
    central_beamstop_diameter_m: float = Field(60e-6, ge=0.0, title='Central Beamstop Diameter:')
    defocus_distance_m: float = defocus_distance_field()


class FresnelZonePlateProbeGenerator(ProbeGenerator):
    """A Fresnel zone plate, propagated to the object with the Fresnel transform."""

    def get_default_options(self) -> BaseModel:
        return FresnelZonePlateOptions()

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        optic = expect_options(options, FresnelZonePlateOptions)
        zone_plate = FresnelZonePlate(
            zone_plate_diameter_m=optic.zone_plate_diameter_m,
            outermost_zone_width_m=optic.outermost_zone_width_m,
            central_beamstop_diameter_m=optic.central_beamstop_diameter_m,
        )
        return generate_fresnel_zone_plate_probe(
            geometry=geometry_provider.get_probe_geometry(),
            zone_plate=zone_plate,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            defocus_distance_m=resolve_defocus_distance_m(
                geometry_provider, optic.defocus_distance_m
            ),
        )


def _mirror_length_field(default: float, title: str) -> Any:
    # Shown in millimeters, the unit an optics log records them in.
    return Field(default, ge=0.0, title=title, json_schema_extra={'display_unit': 'MILLIMETER'})


def _grazing_angle_field(title: str) -> Any:
    # A floor only: with both ends set the editor offers a slider, and a slider to a
    # quarter turn cannot reach the few milliradians a grazing angle is.
    return Field(3e-3, ge=0.0, title=title)


class KBMirrorOptions(BaseModel):
    horizontal_acceptance_length_m: float = _mirror_length_field(
        0.1, 'Horizontal Acceptance Length:'
    )
    horizontal_grazing_angle_rad: float = _grazing_angle_field('Horizontal Grazing Angle [rad]:')
    horizontal_focus_distance_m: float = _mirror_length_field(50e-3, 'Horizontal Focus Distance:')
    vertical_acceptance_length_m: float = _mirror_length_field(0.1, 'Vertical Acceptance Length:')
    vertical_grazing_angle_rad: float = _grazing_angle_field('Vertical Grazing Angle [rad]:')
    vertical_focus_distance_m: float = _mirror_length_field(50e-3, 'Vertical Focus Distance:')
    astigmatism_m: float = Field(
        0.0,
        title='Astigmatism:',
        description='Separation of the two axis foci along the beam; zero is an aligned pair.',
        json_schema_extra={'display_unit': 'MICROMETER'},
    )
    defocus_distance_m: float = defocus_distance_field()


class KBMirrorProbeGenerator(ProbeGenerator):
    """A Kirkpatrick-Baez mirror pair.

    The source distance each mirror accepts is left out: it feeds only a focal length
    the probe simulation never reads, so an option for it would do nothing.
    """

    def get_default_options(self) -> BaseModel:
        return KBMirrorOptions()

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        optic = expect_options(options, KBMirrorOptions)
        mirrors = KirkpatrickBaezMirrorPair(
            horizontal=KirkpatrickBaezMirror(
                acceptance_length_m=optic.horizontal_acceptance_length_m,
                grazing_angle_rad=optic.horizontal_grazing_angle_rad,
                focus_distance_m=optic.horizontal_focus_distance_m,
            ),
            vertical=KirkpatrickBaezMirror(
                acceptance_length_m=optic.vertical_acceptance_length_m,
                grazing_angle_rad=optic.vertical_grazing_angle_rad,
                focus_distance_m=optic.vertical_focus_distance_m,
            ),
        )
        return generate_kb_mirror_probe(
            geometry_provider.get_probe_geometry(),
            mirrors,
            photon_wavelength_m=geometry_provider.photon_wavelength_m,
            defocus_distance_m=resolve_defocus_distance_m(
                geometry_provider, optic.defocus_distance_m
            ),
            astigmatism_m=optic.astigmatism_m,
        )
