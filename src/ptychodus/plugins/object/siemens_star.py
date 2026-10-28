"""A Siemens star resolution target."""

from __future__ import annotations

import numpy
from pydantic import BaseModel, Field

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.probe import Probe
from ptychodus.api.simulate.object import ObjectGenerator, generate_siemens_star_object

from ._options import expect_options


class SiemensStarOptions(BaseModel):
    num_spokes: int = Field(32, ge=2, title='Number of Spokes:')
    outer_radius_fraction: float = Field(
        0.45,
        gt=0.0,
        le=1.0,
        title='Outer Radius Fraction:',
        description="The star's radius as a fraction of the canvas's smaller half-extent.",
    )
    spoke_amplitude: float = Field(0.0, ge=0.0, le=1.0, title='Spoke Amplitude:')
    background_amplitude: float = Field(1.0, ge=0.0, le=1.0, title='Background Amplitude:')
    spoke_phase_tr: float = Field(-0.25, title='Spoke Phase:')
    background_phase_tr: float = Field(0.0, title='Background Phase:')


class SiemensStarObjectGenerator(ObjectGenerator):
    """Alternating angular wedges of two transmissions."""

    def get_default_options(self) -> BaseModel:
        return SiemensStarOptions()

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        star = expect_options(options, SiemensStarOptions)
        return generate_siemens_star_object(
            geometry_provider.get_object_geometry(),
            num_spokes=star.num_spokes,
            outer_radius_fraction=star.outer_radius_fraction,
            spoke_amplitude=star.spoke_amplitude,
            background_amplitude=star.background_amplitude,
            spoke_phase_tr=star.spoke_phase_tr,
            background_phase_tr=star.background_phase_tr,
        )
