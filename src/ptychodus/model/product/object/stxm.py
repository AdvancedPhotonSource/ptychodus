from __future__ import annotations
from collections.abc import Sequence
import logging

from scipy.interpolate import griddata
import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from ...diffraction import AssembledDiffractionDataset
from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class STXMObjectBuilder(ObjectBuilder):
    def __init__(
        self,
        settings: ObjectSettings,
        dataset: AssembledDiffractionDataset,
    ) -> None:
        super().__init__(settings, 'stxm')
        self._settings = settings
        self._dataset = dataset

    def copy(self) -> STXMObjectBuilder:
        builder = STXMObjectBuilder(self._settings, self._dataset)

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

        pattern_counts_lut = self._dataset.get_pattern_counts_lut()

        for scan_point in geometry_provider.get_probe_positions():
            try:
                value = pattern_counts_lut[scan_point.index]
            except KeyError:
                logger.debug(f'Skipping missing scan point index={scan_point.index}!')
            else:
                object_point = geometry.map_coordinates_probe_to_object(scan_point)
                coordinates_px.append(object_point.coordinate_y_px)
                coordinates_px.append(object_point.coordinate_x_px)
                values.append(value)

        points = numpy.reshape(coordinates_px, (-1, 2))
        YY, XX = numpy.mgrid[: geometry.height_px, : geometry.width_px]  # noqa: N806
        query_points = numpy.transpose((YY.flat, XX.flat))
        intensity = griddata(points, values, query_points, method='linear', fill_value=0.0).reshape(
            XX.shape
        )

        return self._create_object(
            array=numpy.sqrt(intensity[numpy.newaxis, :, :]).astype('complex'),
            layer_spacing_m=layer_spacing_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
