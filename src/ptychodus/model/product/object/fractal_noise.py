from __future__ import annotations

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.simulate.object import generate_fractal_noise_object

from .builder import ObjectBuilder
from .settings import ObjectSettings


class FractalNoiseObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'fractal_noise')
        self._rng = rng
        self._settings = settings

        self.grid_scale_px = settings.simplex_grid_scale_px.copy()
        self._add_parameter('grid_scale_px', self.grid_scale_px)

        self.num_octaves = settings.fractal_num_octaves.copy()
        self._add_parameter('num_octaves', self.num_octaves)

        self.gain = settings.fractal_gain.copy()
        self._add_parameter('gain', self.gain)

        self.lacunarity = settings.fractal_lacunarity.copy()
        self._add_parameter('lacunarity', self.lacunarity)

    def copy(self) -> FractalNoiseObjectBuilder:
        builder = FractalNoiseObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        object_ = generate_fractal_noise_object(
            self._rng,
            geometry_provider.get_object_geometry(),
            grid_scale_px=self.grid_scale_px.get_value(),
            num_octaves=self.num_octaves.get_value(),
            gain=self.gain.get_value(),
            lacunarity=self.lacunarity.get_value(),
        )
        return self._pad_object(object_)
