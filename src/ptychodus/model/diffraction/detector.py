import logging

import numpy

from ptychodus.api.diffraction import BadPixels
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.observer import Observable, Observer
from .settings import DetectorSettings

logger = logging.getLogger(__name__)


class Detector(Observable, Observer):
    def __init__(self, settings: DetectorSettings) -> None:
        super().__init__()
        self._settings = settings
        self._bad_pixels: BadPixels | None = None

        settings.add_observer(self)

    def set_extent(self, extent: ImageExtent) -> None:
        logger.debug(f'Detector {extent=}')
        self._settings.height_px.set_value(extent.height_px)
        self._settings.width_px.set_value(extent.width_px)

    def get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self._settings.pixel_width_m.get_value(),
            height_m=self._settings.pixel_height_m.get_value(),
        )

    def set_bad_pixels(self, bad_pixels: BadPixels | None) -> None:
        if bad_pixels is not None and bad_pixels.ndim != 2:
            raise ValueError(f'Bad pixels array must be 2D, got {bad_pixels.ndim}D.')

        self._bad_pixels = bad_pixels
        self.notify_observers()

    def get_bad_pixels(self) -> BadPixels:
        if self._bad_pixels is None:
            detector_height_px = self._settings.height_px.get_value()
            detector_width_px = self._settings.width_px.get_value()
            return numpy.full((detector_height_px, detector_width_px), False)

        return self._bad_pixels

    def get_num_bad_pixels(self) -> int:
        return 0 if self._bad_pixels is None else int(numpy.count_nonzero(self._bad_pixels))

    def _update(self, observable: Observable) -> None:
        if observable is self._settings:
            self.notify_observers()
