from scipy.interpolate import griddata, RBFInterpolator
import numpy

from ptychodus.api.fluorescence import ElementMap, UpscalingStrategy
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import Product


class IdentityUpscaling(UpscalingStrategy):
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        return emap


class GridDataUpscaling(UpscalingStrategy):
    def __init__(self, method: str) -> None:
        self._method = method

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        object_geometry = product.object_.get_geometry()
        scan_coords_px: list[float] = list()

        for scan_point in product.positions:
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            scan_coords_px.append(object_point.position_y_px)
            scan_coords_px.append(object_point.position_x_px)

        points = numpy.reshape(scan_coords_px, (-1, 2))
        values = emap.counts_per_second.flat
        YY, XX = numpy.mgrid[: object_geometry.height_px, : object_geometry.width_px]  # noqa: N806
        query_points = numpy.transpose((YY.flat, XX.flat))

        cps = griddata(points, values, query_points, method=self._method, fill_value=0.0).reshape(
            XX.shape
        )

        return ElementMap(emap.name, cps.astype(emap.counts_per_second.dtype))


class RadialBasisFunctionUpscaling(UpscalingStrategy):
    def __init__(
        self,
        kernel: str,
        *,
        neighbors: int | None = 25,
        epsilon: float | None = None,
        degree: int | None = None,
    ) -> None:
        self._kernel = kernel
        self._neighbors = neighbors
        self._epsilon = epsilon
        self._degree = degree

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        object_geometry = product.object_.get_geometry()
        scan_coords_px: list[float] = list()

        for scan_point in product.positions:
            object_point = object_geometry.map_scan_point_to_object_point(scan_point)
            scan_coords_px.append(object_point.position_y_px)
            scan_coords_px.append(object_point.position_x_px)

        interpolator = RBFInterpolator(
            numpy.reshape(scan_coords_px, (-1, 2)),
            emap.counts_per_second.flat,
            kernel=self._kernel,
            neighbors=self._neighbors,
            epsilon=self._epsilon,
            degree=self._degree,
        )
        YY, XX = numpy.mgrid[: object_geometry.height_px, : object_geometry.width_px]  # noqa: N806
        cps = interpolator(numpy.transpose((YY.flat, XX.flat)))
        return ElementMap(emap.name, cps.astype(emap.counts_per_second.dtype).reshape(XX.shape))


def register_plugins(registry: PluginRegistry) -> None:
    # TODO natural neighbor
    # TODO kriging
    # TODO inverse distance weighting

    registry.upscaling_strategies.register_plugin(
        IdentityUpscaling(),
        display_name='Identity',
    )
    registry.upscaling_strategies.register_plugin(
        GridDataUpscaling('nearest'),
        display_name='Nearest Neighbor',
    )
    registry.upscaling_strategies.register_plugin(
        GridDataUpscaling('linear'),
        display_name='Linear',
    )
    registry.upscaling_strategies.register_plugin(
        GridDataUpscaling('cubic'),
        display_name='Cubic',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('linear'),
        display_name='Linear RBF',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('thin_plate_spline'),
        display_name='Thin Plate Spline RBF',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('cubic'),
        display_name='Cubic RBF',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('quintic'),
        display_name='Quintic RBF',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('multiquadric'),
        display_name='Multiquadric RBF',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('inverse_multiquadric'),
        display_name='Inverse Multiquadric RBF',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('inverse_quadratic'),
        display_name='Inverse Quadratic RBF',
    )
    registry.upscaling_strategies.register_plugin(
        RadialBasisFunctionUpscaling('gaussian'),
        display_name='Gaussian RBF',
    )
