"""Mode expansions: Hermite-Gauss modes and Zernike polynomials.

Each takes a list of complex coefficients in a fixed single-index order, so the length of
the list is the number of modes and a coefficient's position says which mode it weights.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ptychodus.api.geometry import HermiteMode, ZernikeMode
from ptychodus.api.probe import Probe, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    ProbeGenerator,
    generate_hermite_gauss_probe,
    generate_zernike_probe,
)

from ._options import expect_options


class HermiteGaussOptions(BaseModel):
    beam_waist_x_m: float = Field(
        1e-6,
        gt=0.0,
        title='Beam Waist X:',
        description='Radius along x at which the intensity of mode (0, 0) falls to 1/e² of its peak.',
        json_schema_extra={'display_unit': 'MICROMETER'},
    )
    beam_waist_y_m: float = Field(
        1e-6,
        gt=0.0,
        title='Beam Waist Y:',
        description='Radius along y at which the intensity of mode (0, 0) falls to 1/e² of its peak.',
        json_schema_extra={'display_unit': 'MICROMETER'},
    )
    hermite_gauss_coefficients: list[complex] = Field(
        default_factory=lambda: [1 + 0j],
        min_length=1,
        title='Coefficients:',
        description=(
            'Complex amplitude of each unit-power Hermite-Gauss mode, in total-degree '
            'order: (0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2), ...'
        ),
    )


class HermiteGaussProbeGenerator(ProbeGenerator):
    """A superposition of Hermite-Gauss modes."""

    def get_default_options(self) -> BaseModel:
        return HermiteGaussOptions()

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        expansion = expect_options(options, HermiteGaussOptions)
        coefficients = expansion.hermite_gauss_coefficients

        if not coefficients:
            raise ValueError('A Hermite-Gauss probe needs at least one coefficient.')

        return generate_hermite_gauss_probe(
            geometry_provider.get_probe_geometry(),
            [HermiteMode.from_index(index, value) for index, value in enumerate(coefficients)],
            beam_waist_x_m=expansion.beam_waist_x_m,
            beam_waist_y_m=expansion.beam_waist_y_m,
        )


class ZernikeOptions(BaseModel):
    diameter_m: float = Field(1e-6, ge=0.0, title='Diameter:')
    zernike_coefficients: list[complex] = Field(
        default_factory=lambda: [1 + 0j],
        min_length=1,
        title='Coefficients:',
        description=(
            'Complex amplitude of each Zernike polynomial, in OSA/ANSI order: '
            '(0, 0), (1, -1), (1, 1), (2, -2), (2, 0), (2, 2), ...'
        ),
    )


class ZernikeProbeGenerator(ProbeGenerator):
    """A superposition of Zernike polynomials inside a circular pupil."""

    def get_default_options(self) -> BaseModel:
        return ZernikeOptions()

    def generate(self, options: BaseModel, geometry_provider: ProbeGeometryProvider) -> Probe:
        expansion = expect_options(options, ZernikeOptions)
        coefficients = expansion.zernike_coefficients

        if not coefficients:
            raise ValueError('A Zernike probe needs at least one coefficient.')

        return generate_zernike_probe(
            geometry_provider.get_probe_geometry(),
            [ZernikeMode.from_osa_index(index, value) for index, value in enumerate(coefficients)],
            radius_m=expansion.diameter_m / 2.0,
        )
