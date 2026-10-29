"""Remap element concentrations from the scan grid onto the object grid."""

from __future__ import annotations
from enum import StrEnum

from scipy.interpolate import griddata, RBFInterpolator
import numpy

from ptychodus.api.fluorescence import ElementMap
from ptychodus.api.product import Product
from ptychodus.api.typing import RealArrayType

__all__ = [
    'UpscalingMethod',
    'upscale',
]


class UpscalingMethod(StrEnum):
    """How measured values at the scan positions are spread onto the object pixels.

    Only the radial basis kernels that need no shape parameter are offered: the
    others take an ``epsilon`` whose scale nothing here can choose.
    """

    # TODO natural neighbor
    # TODO kriging
    # TODO inverse distance weighting
    IDENTITY = 'identity'
    NEAREST = 'nearest'
    LINEAR = 'linear'
    CUBIC = 'cubic'
    LINEAR_RBF = 'linear_rbf'
    THIN_PLATE_SPLINE_RBF = 'thin_plate_spline_rbf'
    CUBIC_RBF = 'cubic_rbf'
    QUINTIC_RBF = 'quintic_rbf'


def _scan_points_px(product: Product) -> RealArrayType:
    object_geometry = product.object_.get_geometry()
    return numpy.column_stack(
        (
            object_geometry.map_probe_positions_to_object_y_px(product.probe_positions),
            object_geometry.map_probe_positions_to_object_x_px(product.probe_positions),
        )
    )


def _upscale_with_griddata(emap: ElementMap, product: Product, method: str) -> ElementMap:
    object_geometry = product.object_.get_geometry()
    points = _scan_points_px(product)
    values = emap.counts_per_second.flat
    shape = (object_geometry.height_px, object_geometry.width_px)
    query_points = numpy.indices(shape).reshape(2, -1).T

    cps = griddata(points, values, query_points, method=method, fill_value=0.0).reshape(shape)

    return ElementMap(emap.name, cps.astype(emap.counts_per_second.dtype))


def _upscale_with_rbf(
    emap: ElementMap, product: Product, kernel: str, *, neighbors: int | None
) -> ElementMap:
    object_geometry = product.object_.get_geometry()
    interpolator = RBFInterpolator(
        _scan_points_px(product),
        emap.counts_per_second.flat,
        kernel=kernel,
        neighbors=neighbors,
    )
    shape = (object_geometry.height_px, object_geometry.width_px)
    cps = interpolator(numpy.indices(shape).reshape(2, -1).T)
    return ElementMap(emap.name, cps.astype(emap.counts_per_second.dtype).reshape(shape))


def upscale(
    emap: ElementMap,
    product: Product,
    method: UpscalingMethod,
    *,
    rbf_neighbors: int | None = 25,
) -> ElementMap:
    """Return *emap* sampled on the object grid of *product* by *method*.

    *rbf_neighbors* is how many nearest scan points each radial basis evaluation
    draws on; None uses every point, which is exact but scales with the square of
    the scan size. The grid-data methods ignore it.

    Raises:
        ValueError: for a method this function does not know.
    """
    match method:
        case UpscalingMethod.IDENTITY:
            return emap
        case UpscalingMethod.NEAREST | UpscalingMethod.LINEAR | UpscalingMethod.CUBIC:
            return _upscale_with_griddata(emap, product, method.value)
        case UpscalingMethod.LINEAR_RBF:
            return _upscale_with_rbf(emap, product, 'linear', neighbors=rbf_neighbors)
        case UpscalingMethod.THIN_PLATE_SPLINE_RBF:
            return _upscale_with_rbf(emap, product, 'thin_plate_spline', neighbors=rbf_neighbors)
        case UpscalingMethod.CUBIC_RBF:
            return _upscale_with_rbf(emap, product, 'cubic', neighbors=rbf_neighbors)
        case UpscalingMethod.QUINTIC_RBF:
            return _upscale_with_rbf(emap, product, 'quintic', neighbors=rbf_neighbors)

    raise ValueError(f'Unknown upscaling method "{method}"!')
