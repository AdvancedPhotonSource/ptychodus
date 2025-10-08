from __future__ import annotations
from collections.abc import Sequence

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import ObjectBuilder
from .settings import ObjectSettings


class STXMObjectBuilder(ObjectBuilder):
    def __init__(self, settings: ObjectSettings) -> None:
        super().__init__(settings, 'stxm')
        self._settings = settings

    def copy(self) -> STXMObjectBuilder:
        builder = STXMObjectBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        geometry = geometry_provider.get_object_geometry()
        height_px = geometry.height_px  # FIXME + 2 * self.extra_padding_y.get_value()
        width_px = geometry.width_px  # FIXME + 2 * self.extra_padding_x.get_value()
        object_shape = (1 + len(layer_spacing_m), height_px, width_px)
        array = numpy.zeros(object_shape) + 0j

        return Object(
            array=array,
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
