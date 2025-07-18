import logging

import numpy
import scipy.fft

from ptychodus.api.observer import Observable

from ..product import ObjectRepository

logger = logging.getLogger(__name__)


class FourierAnalyzer(Observable):
    def __init__(self, object_repository: ObjectRepository) -> None:
        super().__init__()
        self.object_repository = object_repository

        self._product_index = -1

    def set_product(self, product_index: int) -> None:  # FIXME
        if self._product_index != product_index:
            self._product_index = product_index
            self._product_data = None
            self.notify_observers()

    def analyze(self) -> None:
        # Implement Fourier analysis logic here
        pass
