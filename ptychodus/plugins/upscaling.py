from scipy.interpolate import griddata, RBFInterpolator
import numpy

from ptychodus.api.fluorescence import ElementMap, UpscalingStrategy
from ptychodus.api.object import Object
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.product import Product
from ptychodus.api.scan import Scan
from ptychodus.api.typing import RealArrayType


def _scan_to_array(scan: Scan) -> RealArrayType:  # FIXME REMOVE
    coords: list[float] = list()

    for point in scan:
        coords.append(point.positionYInMeters)
        coords.append(point.positionXInMeters)

    return numpy.reshape(coords, (-1, 2))


def _object_coordinates(numberOfPixels: int, pixelSizeInMeters: float,
                        centerInMeters: float) -> RealArrayType:  # FIXME REMOVE
    positionInPixels = numpy.arange(numberOfPixels) - numberOfPixels / 2
    return centerInMeters + positionInPixels * pixelSizeInMeters


def _object_coordinates_yx(object_: Object) -> tuple[RealArrayType, RealArrayType]:  # FIXME REMOVE
    axisXInMeters = _object_coordinates(object_.widthInPixels, object_.pixelWidthInMeters,
                                        object_.centerXInMeters)
    axisYInMeters = _object_coordinates(object_.heightInPixels, object_.pixelHeightInMeters,
                                        object_.centerYInMeters)
    gridYInMeters, gridXInMeters = numpy.meshgrid(axisYInMeters, axisXInMeters)
    return gridYInMeters, gridXInMeters


class IdentityUpscaling(UpscalingStrategy):

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        return emap


class GridDataUpscaling(UpscalingStrategy):

    def __init__(self, method: str) -> None:
        self._method = method

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        objectGeometry = product.object_.getGeometry()
        scanCoordinatesInPixels: list[float] = list()

        for scanPoint in product.scan:
            objectPoint = objectGeometry.mapScanPointToObjectPoint(scanPoint)
            scanCoordinatesInPixels.append(objectPoint.positionYInPixels)
            scanCoordinatesInPixels.append(objectPoint.positionXInPixels)

        points = numpy.reshape(scanCoordinatesInPixels, (-1, 2))
        values = emap.counts_per_second.flat
        YY, XX = numpy.mgrid[:objectGeometry.heightInPixels, :objectGeometry.widthInPixels]
        query_points = numpy.transpose((YY.flat, XX.flat))

        cps = griddata(points, values, query_points, method=self._method, fill_value=0.).reshape(XX.shape)

        return ElementMap(emap.name, cps.astype(emap.counts_per_second.dtype))


class RadialBasisFunctionUpscaling(UpscalingStrategy):

    def __init__(self,
                 kernel: str,
                 *,
                 neighbors: int | None = 25,
                 epsilon: float | None = None,
                 degree: int | None = None) -> None:
        self._kernel = kernel
        self._neighbors = neighbors
        self._epsilon = epsilon
        self._degree = degree

    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        interpolator = RBFInterpolator(
            _scan_to_array(product.scan),
            emap.counts_per_second.flat,
            kernel=self._kernel,
            neighbors=self._neighbors,
            epsilon=self._epsilon,
            degree=self._degree,
        )
        grid_y, grid_x = _object_coordinates_yx(product.object_)
        cps = interpolator(numpy.transpose((grid_y.flat, grid_x.flat)))
        return ElementMap(emap.name,
                          cps.astype(emap.counts_per_second.dtype).reshape(grid_x.shape))


def registerPlugins(registry: PluginRegistry) -> None:
    # TODO natural neighbor
    # TODO kriging
    # TODO inverse distance weighting

    registry.upscalingStrategies.registerPlugin(
        IdentityUpscaling(),
        displayName='Identity',
    )
    registry.upscalingStrategies.registerPlugin(
        GridDataUpscaling('nearest'),
        displayName='Nearest Neighbor',
    )
    registry.upscalingStrategies.registerPlugin(
        GridDataUpscaling('linear'),
        displayName='Linear',
    )
    registry.upscalingStrategies.registerPlugin(
        GridDataUpscaling('cubic'),
        displayName='Cubic',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('linear'),
        displayName='Linear RBF',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('thin_plate_spline'),
        displayName='Thin Plate Spline RBF',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('cubic'),
        displayName='Cubic RBF',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('quintic'),
        displayName='Quintic RBF',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('multiquadric'),
        displayName='Multiquadric RBF',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('inverse_multiquadric'),
        displayName='Inverse Multiquadric RBF',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('inverse_quadratic'),
        displayName='Inverse Quadratic RBF',
    )
    registry.upscalingStrategies.registerPlugin(
        RadialBasisFunctionUpscaling('gaussian'),
        displayName='Gaussian RBF',
    )
