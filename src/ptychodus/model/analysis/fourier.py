from dataclasses import dataclass
import logging

from scipy.fft import fft2, fftshift, ifftshift

from ptychodus.api.typing import ComplexArrayType
from ptychodus.api.geometry import Box2D, PixelGeometry
from ptychodus.api.interpolate import NearestNeighborArrayInterpolator
from ptychodus.api.object import Object

from ..product import ProductRepository


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FourierAnalysisResult:
    transformed_roi: ComplexArrayType
    pixel_geometry: PixelGeometry


class FourierAnalyzer:
    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def get_object(self, product_index: int) -> Object:
        return self._repository[product_index].get_object_item().get_object()

    def analyze_layer(self, product_index: int) -> FourierAnalysisResult:
        object_ = self.get_object(product_index)
        return self._analyze(object_.get_layer(0), object_.get_pixel_geometry())

    def analyze_roi(self, product_index: int, bounding_box: Box2D) -> FourierAnalysisResult:
        logger.debug(f'bounding_box: {bounding_box}')
        object_ = self.get_object(product_index)
        interpolator = NearestNeighborArrayInterpolator(object_.get_layer(0))

        width = int(bounding_box.width + 0.5)
        height = int(bounding_box.height + 0.5)
        roi = interpolator.get_patch(bounding_box.x_center, bounding_box.y_center, width, height)
        logger.debug(f'roi: {roi.dtype}{roi.shape}')
        return self._analyze(roi, object_.get_pixel_geometry())

    @staticmethod
    def _analyze(array: ComplexArrayType, pixel_geometry: PixelGeometry) -> FourierAnalysisResult:
        norm = 'forward'  # TODO let user choose norm
        return FourierAnalysisResult(
            transformed_roi=fftshift(fft2(ifftshift(array), norm=norm)),
            pixel_geometry=pixel_geometry,
        )
