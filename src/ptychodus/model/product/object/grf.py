from __future__ import annotations

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.object_gen import generate_gaussian_random_field_object

from .builder import ObjectBuilder
from .settings import ObjectSettings


class GaussianRandomFieldObjectBuilder(ObjectBuilder):
    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings) -> None:
        super().__init__(settings, 'gaussian_random_field')
        self._rng = rng
        self._settings = settings

        self.correlation_length_px = settings.correlation_length_px.copy()
        self._add_parameter('correlation_length_px', self.correlation_length_px)

    def copy(self) -> GaussianRandomFieldObjectBuilder:
        builder = GaussianRandomFieldObjectBuilder(self._rng, self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        object_ = generate_gaussian_random_field_object(
            self._rng,
            geometry_provider.get_object_geometry(),
            correlation_length_px=self.correlation_length_px.get_value(),
        )
        return self._pad_object(object_)
