from __future__ import annotations
from collections.abc import Sequence

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import ObjectBuilder
from .settings import ObjectSettings


class RandomObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'random')
        self._rng = rng
        self._settings = settings

        self.amplitude_mean = settings.amplitude_mean.copy()
        self._add_parameter('amplitude_mean', self.amplitude_mean)
        self.amplitude_deviation = settings.amplitude_deviation.copy()
        self._add_parameter('amplitude_deviation', self.amplitude_deviation)
        self.phase_deviation = settings.phase_deviation.copy()
        self._add_parameter('phase_deviation', self.phase_deviation)

    def copy(self) -> RandomObjectBuilder:
        builder = RandomObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        geometry = geometry_provider.get_object_geometry()
        object_shape = (1, geometry.height_px, geometry.width_px)

        amplitude = self._rng.normal(
            self.amplitude_mean.get_value(),
            self.amplitude_deviation.get_value(),
            object_shape,
        )
        phase = self._rng.normal(
            0.0,
            self.phase_deviation.get_value(),
            object_shape,
        )
        array = numpy.clip(amplitude, 0.0, 1.0) * numpy.exp(1j * phase)

        return self._create_object(
            array=array.astype('complex'),
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
