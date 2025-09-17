from dataclasses import dataclass
import logging

from scipy.fft import fft2, fftshift, ifftshift

from ptychodus.api.geometry import Box2D, PixelGeometry
from ptychodus.api.object import Object
from ptychodus.api.observer import Observable
from ptychodus.api.typing import ComplexArrayType

from ..product import ProductRepository
from .interpolators import NearestNeighborArrayInterpolator


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FourierAnalysisResult:
    transformed_roi: ComplexArrayType
    pixel_geometry: PixelGeometry


class FourierAnalyzer(Observable):
    def __init__(self, repository: ProductRepository) -> None:
        super().__init__()
        self._repository = repository

        self._product_index = -1
        self._result: FourierAnalysisResult | None = None

    def set_product(self, product_index: int) -> None:
        if self._product_index != product_index:
            self._product_index = product_index
            self._result = None
            self.notify_observers()

    def get_product_name(self) -> str:
        product = self._repository[self._product_index]
        return product.get_name()

    def get_object(self) -> Object:
        product = self._repository[self._product_index]
        return product.get_object_item().get_object()

    def analyze_roi(self, bounding_box: Box2D) -> None:
        logger.debug(f'bounding_box: {bounding_box}')
        object_ = self.get_object()
        interpolator = NearestNeighborArrayInterpolator(object_.get_layer(0))

        width = int(bounding_box.width + 0.5)
        height = int(bounding_box.height + 0.5)
        roi = interpolator.get_patch(bounding_box.x_center, bounding_box.y_center, width, height)
        logger.debug(f'roi: {roi.dtype}{roi.shape}')

        # FIXME choose fft scaling
        self._result = FourierAnalysisResult(
            transformed_roi=fftshift(fft2(ifftshift(roi))),
            pixel_geometry=object_.get_pixel_geometry(),
        )
        self.notify_observers()

    def get_result(self) -> FourierAnalysisResult:
        if self._result is None:
            raise ValueError('Fourier analysis has not been performed yet.')

        return self._result
