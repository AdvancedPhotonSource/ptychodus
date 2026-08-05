from __future__ import annotations

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.object_gen import generate_random_object

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
        self.phase_deviation_tr = settings.phase_deviation_tr.copy()
        self._add_parameter('phase_deviation_turns', self.phase_deviation_tr)
        self.blur_deviation_px = settings.blur_deviation_px.copy()
        self._add_parameter('blur_deviation_px', self.blur_deviation_px)

    def copy(self) -> RandomObjectBuilder:
        builder = RandomObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        object_ = generate_random_object(
            self._rng,
            geometry_provider.get_object_geometry(),
            amplitude_mean=self.amplitude_mean.get_value(),
            amplitude_deviation=self.amplitude_deviation.get_value(),
            phase_mean=0.0,
            phase_deviation_tr=self.phase_deviation_tr.get_value(),
            blur_deviation_px=self.blur_deviation_px.get_value(),
        )
        return self._pad_object(object_)
