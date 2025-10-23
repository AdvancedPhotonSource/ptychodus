from __future__ import annotations
from collections.abc import Sequence

from scipy.interpolate import griddata
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
        coordinates_px: list[float] = list()
        values: list[float] = list()

        for scan_point in geometry_provider.get_probe_positions():
            object_point = geometry.map_coordinates_probe_to_object(scan_point)
            coordinates_px.append(object_point.coordinate_y_px)
            coordinates_px.append(object_point.coordinate_x_px)
            values.append(scan_point.index % 10)  # FIXME from patterns

        points = numpy.reshape(coordinates_px, (-1, 2))
        YY, XX = numpy.mgrid[: geometry.height_px, : geometry.width_px]  # noqa: N806
        query_points = numpy.transpose((YY.flat, XX.flat))

        # FIXME add padding and extra

        intensity = griddata(points, values, query_points, method='linear', fill_value=0.0).reshape(
            XX.shape
        )
        array = numpy.tile(numpy.sqrt(intensity), (1 + len(layer_spacing_m), 1, 1)) + 0j

        return Object(
            array=array,
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
