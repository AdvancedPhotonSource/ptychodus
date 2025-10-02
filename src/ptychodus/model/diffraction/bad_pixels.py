import logging

import numpy

from ptychodus.api.diffraction import BadPixels
from ptychodus.api.observer import Observable, Observer

from .settings import DetectorSettings

logger = logging.getLogger(__name__)


class BadPixelsProvider(Observable, Observer):
    def __init__(self, settings: DetectorSettings) -> None:
        self._settings = settings
        self._bad_pixels: BadPixels | None = None

        settings.height_px.add_observer(self)
        settings.width_px.add_observer(self)

    def clear_bad_pixels(self) -> None:
        self._bad_pixels = None
        self.notify_observers()

    def set_bad_pixels(self, bad_pixels: BadPixels) -> None:
        if bad_pixels.ndim != 2:
            raise ValueError(f'Bad pixels array must be 2D, got {bad_pixels.ndim}D.')

        num_bad_pixels = int(numpy.count_nonzero(bad_pixels))
        self._bad_pixels = bad_pixels if num_bad_pixels > 0 else None
        self._settings.height_px.set_value(bad_pixels.shape[-2])
        self._settings.width_px.set_value(bad_pixels.shape[-1])
        self.notify_observers()

    def get_bad_pixels(self) -> BadPixels:
        if self._bad_pixels is None:
            height_px = self._settings.height_px.get_value()
            width_px = self._settings.width_px.get_value()
            return numpy.full((height_px, width_px), False)

        return self._bad_pixels

    def get_num_bad_pixels(self) -> int:
        if self._bad_pixels is None:
            return 0

        return int(numpy.count_nonzero(self._bad_pixels))

    def _clear_bad_pixels_if_mismatched(self, actual: int, expected: int) -> None:
        if actual != expected:
            self.clear_bad_pixels()

    def _update(self, observable: Observable) -> None:
        if self._bad_pixels is not None:
            match observable:
                case self._settings.height_px:
                    self._clear_bad_pixels_if_mismatched(
                        self._bad_pixels.shape[-2], self._settings.height_px.get_value()
                    )
                case self._settings.width_px:
                    self._clear_bad_pixels_if_mismatched(
                        self._bad_pixels.shape[-1], self._settings.width_px.get_value()
                    )
