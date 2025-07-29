from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
import logging

import numpy
import scipy.fft

from ptychodus.api.typing import ComplexArrayType, IntegerArrayType
from ptychodus.api.visualization import Plot2D, PlotAxis, PlotSeries

from ..product import ObjectRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FourierRingCorrelation:
    spatial_frequency_per_m: Sequence[float]
    correlation: Sequence[float]

    def get_resolution_m(self, threshold: float) -> float:
        # TODO threshold from bits
        for freq_rm, frc in zip(self.spatial_frequency_per_m, self.correlation):
            if frc < threshold:
                return 1.0 / freq_rm

        return numpy.nan

    def get_plot(self) -> Plot2D:
        freq_series = PlotSeries('freq', [1.0e-9 * freq for freq in self.spatial_frequency_per_m])
        frc_series = PlotSeries('frc', self.correlation)

        return Plot2D(
            axis_x=PlotAxis('Spatial Frequency [1/nm]', [freq_series]),
            axis_y=PlotAxis('Fourier Ring Correlation', [frc_series]),
        )


class FourierRingCorrelator:
    def __init__(self, repository: ObjectRepository) -> None:
        self._repository = repository

    @staticmethod
    def _integrate_rings(rings: IntegerArrayType, array: ComplexArrayType) -> ComplexArrayType:
        total = numpy.zeros(numpy.max(rings) + 1, dtype=complex)

        for index, value in zip(rings.flat, array.flat):
            total[index] += value

        return total

    def correlate(self, product_index_1: int, product_index_2: int) -> FourierRingCorrelation:
        """
        See: Joan Vila-Comamala, Ana Diaz, Manuel Guizar-Sicairos, Alexandre Mantion,
        Cameron M. Kewish, Andreas Menzel, Oliver Bunk, and Christian David,
        "Characterization of high-resolution diffractive X-ray optics by ptychographic
        coherent diffractive imaging," Opt. Express 19, 21333-21344 (2011)
        """

        object1 = self._repository[product_index_1].get_object()
        object2 = self._repository[product_index_2].get_object()

        # TODO support multilayer objects
        array1 = object1.get_layer(0)
        array2 = object2.get_layer(0)

        if numpy.ndim(array1) != 2 or numpy.ndim(array2) != 2:
            raise ValueError('Arrays must be 2D!')

        if numpy.shape(array1) != numpy.shape(array2):
            raise ValueError('Arrays must have same shape!')

        # TODO verify compatible pixel geometry
        pixel_geometry = object2.get_pixel_geometry()

        # TODO subpixel image registration: skimage.registration.phase_cross_correlation
        # TODO remove phase offset and ramp
        # TODO apply soft-edged mask
        # TODO stats: SSNR, area under FRC curve, average SNR, etc.

        x_rm = scipy.fft.fftfreq(array1.shape[-1], d=pixel_geometry.width_m)
        y_rm = scipy.fft.fftfreq(array1.shape[-2], d=pixel_geometry.height_m)
        radial_bin_size_per_m = max(x_rm[1], y_rm[1])

        xx_rm, yy_rm = numpy.meshgrid(x_rm, y_rm)
        rr_rm = numpy.hypot(xx_rm, yy_rm)

        rings = numpy.divide(rr_rm, radial_bin_size_per_m).astype(int)
        spatial_frequency_per_m = numpy.arange(numpy.max(rings) + 1) * radial_bin_size_per_m

        sf1 = scipy.fft.fft2(array1)
        sf2 = scipy.fft.fft2(array2)

        c11 = self._integrate_rings(rings, numpy.multiply(sf1, numpy.conj(sf1)))
        c12 = self._integrate_rings(rings, numpy.multiply(sf1, numpy.conj(sf2)))
        c22 = self._integrate_rings(rings, numpy.multiply(sf2, numpy.conj(sf2)))

        correlation = numpy.absolute(c12) / numpy.sqrt(numpy.absolute(numpy.multiply(c11, c22)))

        # TODO replace NaNs with interpolated values

        rnyquist = numpy.min(array1.shape) // 2 + 1
        return FourierRingCorrelation(spatial_frequency_per_m[:rnyquist], correlation[:rnyquist])
