from __future__ import annotations
from collections.abc import Sequence

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.object_gen import generate_fractal_noise_object

from .builder import ObjectBuilder
from .settings import ObjectSettings


class FractalNoiseObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'fractal_noise')
        self._rng = rng
        self._settings = settings

        self.grid_scale_m = settings.simplex_grid_scale_m.copy()
        self._add_parameter('grid_scale_m', self.grid_scale_m)

        self.vertex_support_px2 = settings.simplex_vertex_support_px2.copy()
        self._add_parameter('vertex_support_px2', self.vertex_support_px2)

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

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        # FIXME add to view controller
        object_ = generate_fractal_noise_object(
            self._rng,
            geometry_provider.get_object_geometry(),
            grid_scale_m=self.grid_scale_m.get_value(),
            vertex_support_px2=self.vertex_support_px2.get_value(),
            num_octaves=self.num_octaves.get_value(),
            gain=self.gain.get_value(),
            lacunarity=self.lacunarity.get_value(),
        )
        return self._create_object(object_, layer_spacing_m)
