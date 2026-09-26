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
        points = numpy.column_stack(
            (
                object_geometry.map_probe_positions_to_object_y_px(product.probe_positions),
                object_geometry.map_probe_positions_to_object_x_px(product.probe_positions),
            )
        )
        values = emap.counts_per_second.flat
        shape = (object_geometry.height_px, object_geometry.width_px)
        query_points = numpy.indices(shape).reshape(2, -1).T

        cps = griddata(points, values, query_points, method=self._method, fill_value=0.0).reshape(
            shape
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
        scan_coords_px = numpy.column_stack(
            (
                object_geometry.map_probe_positions_to_object_y_px(product.probe_positions),
                object_geometry.map_probe_positions_to_object_x_px(product.probe_positions),
            )
        )

        interpolator = RBFInterpolator(
            scan_coords_px,
            emap.counts_per_second.flat,
            kernel=self._kernel,
            neighbors=self._neighbors,
            epsilon=self._epsilon,
            degree=self._degree,
        )
        shape = (object_geometry.height_px, object_geometry.width_px)
        cps = interpolator(numpy.indices(shape).reshape(2, -1).T)
        return ElementMap(emap.name, cps.astype(emap.counts_per_second.dtype).reshape(shape))


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
