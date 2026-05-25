from __future__ import annotations
import logging

import numpy

from ptychodus.api.metrics import FourierRingCorrelation, compute_fourier_ring_correlation

from ..product import ObjectRepository

__all__ = ['FourierRingCorrelation', 'FourierRingCorrelator']

logger = logging.getLogger(__name__)


class FourierRingCorrelator:
    def __init__(self, repository: ObjectRepository) -> None:
        self._repository = repository

    def correlate(self, product_index_1: int, product_index_2: int) -> FourierRingCorrelation:
        object1 = self._repository[product_index_1].get_object()
        object2 = self._repository[product_index_2].get_object()

        # FIXME support multilayer objects
        array1 = object1.get_layer(0)
        array2 = object2.get_layer(0)

        if numpy.ndim(array1) != 2 or numpy.ndim(array2) != 2:
            raise ValueError('Arrays must be 2D!')

        if numpy.shape(array1) != numpy.shape(array2):
            raise ValueError('Arrays must have same shape!')

        # FIXME verify compatible pixel geometry
        # FIXME subpixel image registration: skimage.registration.phase_cross_correlation
        # FIXME remove phase offset and ramp
        # FIXME apply soft-edged mask
        # FIXME stats: SSNR, area under FRC curve, average SNR, etc.
        pixel_geometry = object2.get_pixel_geometry()

        return compute_fourier_ring_correlation(
            array1,
            array2,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )
