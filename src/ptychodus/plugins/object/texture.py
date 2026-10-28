"""Random textures: Gaussian fields, a spectral random field, fractal noise and dead leaves.

Each draws every random number it uses from the generator it is handed, so the same
draws give the same texture and the generator itself holds nothing between calls.
"""

from __future__ import annotations

import numpy
from pydantic import BaseModel, Field

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.probe import Probe
from ptychodus.api.simulate.object import (
    ObjectGenerator,
    generate_dead_leaves_object,
    generate_fractal_noise_object,
    generate_gaussian_random_field_object,
    generate_random_object,
)

from ._options import expect_options


class RandomOptions(BaseModel):
    amplitude_mean: float = Field(1.0, ge=0.0, le=1.0, title='Amplitude Mean:')
    amplitude_deviation: float = Field(0.0, ge=0.0, le=1.0, title='Amplitude Deviation:')
    phase_deviation_tr: float = Field(0.0, ge=0.0, le=1.0, title='Phase Deviation:')
    blur_deviation_px: float = Field(
        0.0,
        ge=0.0,
        title='Blur Deviation:',
        description='Width of the Gaussian blur applied to both fields; zero leaves them sharp.',
    )


class RandomObjectGenerator(ObjectGenerator):
    """Independent Gaussian amplitude and phase fields, optionally blurred."""

    def get_default_options(self) -> BaseModel:
        return RandomOptions()

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        gaussian = expect_options(options, RandomOptions)
        return generate_random_object(
            rng,
            geometry_provider.get_object_geometry(),
            amplitude_mean=gaussian.amplitude_mean,
            amplitude_deviation=gaussian.amplitude_deviation,
            phase_mean_tr=0.0,
            phase_deviation_tr=gaussian.phase_deviation_tr,
            blur_deviation_px=gaussian.blur_deviation_px,
        )


class GaussianRandomFieldOptions(BaseModel):
    correlation_length_px: float = Field(30.0, ge=0.0, title='Correlation Length:')


class GaussianRandomFieldObjectGenerator(ObjectGenerator):
    """A complex Gaussian random field, synthesized under a Gaussian power spectrum."""

    def get_default_options(self) -> BaseModel:
        return GaussianRandomFieldOptions()

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        field = expect_options(options, GaussianRandomFieldOptions)
        return generate_gaussian_random_field_object(
            rng,
            geometry_provider.get_object_geometry(),
            correlation_length_px=field.correlation_length_px,
        )


class FractalNoiseOptions(BaseModel):
    grid_scale_px: float = Field(
        30.0, gt=0.0, title='Grid Scale:', description='Simplex grid scale of the first octave.'
    )
    num_octaves: int = Field(1, ge=1, le=8, title='Number of Octaves:')
    gain: float = Field(
        0.5, ge=0.0, title='Gain:', description='Amplitude ratio of each octave to the one before.'
    )
    lacunarity: float = Field(
        2.0,
        gt=0.0,
        title='Lacunarity:',
        description='Factor each octave divides the grid scale by.',
    )


class FractalNoiseObjectGenerator(ObjectGenerator):
    """Octaves of complex simplex noise, summed."""

    def get_default_options(self) -> BaseModel:
        return FractalNoiseOptions()

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        noise = expect_options(options, FractalNoiseOptions)
        return generate_fractal_noise_object(
            rng,
            geometry_provider.get_object_geometry(),
            grid_scale_px=noise.grid_scale_px,
            num_octaves=noise.num_octaves,
            gain=noise.gain,
            lacunarity=noise.lacunarity,
        )


class DeadLeavesOptions(BaseModel):
    leaf_radius_lower_px: float = Field(2.0, ge=0.0, title='Leaf Radius Lower:')
    leaf_radius_upper_px: float = Field(999.0, ge=0.0, title='Leaf Radius Upper:')
    leaf_radius_power_law_exponent: float = Field(
        3.0, ge=1.0, title='Leaf Radius Power Law Exponent:'
    )
    leaf_amplitude_lower: float = Field(0.0, ge=0.0, le=1.0, title='Leaf Amplitude Lower:')
    leaf_amplitude_upper: float = Field(1.0, ge=0.0, le=1.0, title='Leaf Amplitude Upper:')
    leaf_phase_lower_tr: float = Field(-0.5, title='Leaf Phase Lower:')
    leaf_phase_upper_tr: float = Field(0.5, title='Leaf Phase Upper:')


class DeadLeavesObjectGenerator(ObjectGenerator):
    """Disks of power-law radii and uniform transmissions, layered over one another."""

    def get_default_options(self) -> BaseModel:
        return DeadLeavesOptions()

    def generate(
        self,
        options: BaseModel,
        geometry_provider: ObjectGeometryProvider,
        *,
        rng: numpy.random.Generator,
        probe: Probe | None,
    ) -> Object:
        leaves = expect_options(options, DeadLeavesOptions)
        return generate_dead_leaves_object(
            rng,
            geometry_provider.get_object_geometry(),
            leaf_radius_lower_px=leaves.leaf_radius_lower_px,
            leaf_radius_upper_px=leaves.leaf_radius_upper_px,
            leaf_radius_power_law_exponent=leaves.leaf_radius_power_law_exponent,
            leaf_amplitude_lower=leaves.leaf_amplitude_lower,
            leaf_amplitude_upper=leaves.leaf_amplitude_upper,
            leaf_phase_lower_tr=leaves.leaf_phase_lower_tr,
            leaf_phase_upper_tr=leaves.leaf_phase_upper_tr,
        )
