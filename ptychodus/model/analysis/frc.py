from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias
import logging

import numpy
import numpy.typing
import scipy.fft

from ...api.object import ObjectArrayType
from ...api.visualize import Plot2D, PlotAxis, PlotSeries
from ..product import ObjectRepository

IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FourierRingCorrelation:
    spatialFrequency_rm: Sequence[float]
    correlation: Sequence[float]

    def getResolutionInMeters(self, threshold: float) -> float:
        # TODO threshold from bits
        for freq_rm, frc in zip(self.spatialFrequency_rm, self.correlation):
            if frc < threshold:
                return 1. / freq_rm

        return numpy.nan

    def getPlot(self) -> Plot2D:
        freqSeries = PlotSeries('freq', [1.e-9 * freq for freq in self.spatialFrequency_rm])
        frcSeries = PlotSeries('frc', self.correlation)

        return Plot2D(
            axisX=PlotAxis('Spatial Frequency [1/nm]', [freqSeries]),
            axisY=PlotAxis('Fourier Ring Correlation', [frcSeries]),
        )


class FourierRingCorrelator:

    def __init__(self, repository: ObjectRepository) -> None:
        self._repository = repository

    @staticmethod
    def _integrateRings(rings: IntegerArrayType, array: ObjectArrayType) -> ObjectArrayType:
        total = numpy.zeros(numpy.max(rings) + 1, dtype=complex)

        for index, value in zip(rings.flat, array.flat):
            total[index] += value

        return total

    def correlate(self, itemIndex1: int, itemIndex2: int) -> FourierRingCorrelation:
        '''
        See: Joan Vila-Comamala, Ana Diaz, Manuel Guizar-Sicairos, Alexandre Mantion,
        Cameron M. Kewish, Andreas Menzel, Oliver Bunk, and Christian David,
        "Characterization of high-resolution diffractive X-ray optics by ptychographic
        coherent diffractive imaging," Opt. Express 19, 21333-21344 (2011)
        '''

        object1 = self._repository[itemIndex1].getObject()
        object2 = self._repository[itemIndex2].getObject()

        # TODO support multilayer objects
        array1 = object1.getLayer(0)
        array2 = object2.getLayer(0)

        if numpy.ndim(array1) != 2 or numpy.ndim(array2) != 2:
            raise ValueError('Arrays must be 2D!')

        if numpy.shape(array1) != numpy.shape(array2):
            raise ValueError('Arrays must have same shape!')

        # TODO verify compatible pixel geometry
        pixelGeometry = object2.getPixelGeometry()

        # TODO subpixel image registration
        # TODO remove phase offset and ramp
        # TODO apply soft-edged mask
        # TODO stats: SSNR, area under FRC curve, average SNR, etc.

        x_rm = scipy.fft.fftfreq(array1.shape[-1], d=pixelGeometry.widthInMeters)
        y_rm = scipy.fft.fftfreq(array1.shape[-2], d=pixelGeometry.heightInMeters)
        radialBinSize_rm = max(x_rm[1], y_rm[1])

        xx_rm, yy_rm = numpy.meshgrid(x_rm, y_rm)
        rr_rm = numpy.hypot(xx_rm, yy_rm)

        rings = numpy.divide(rr_rm, radialBinSize_rm).astype(int)
        spatialFrequency_rm = numpy.arange(numpy.max(rings) + 1) * radialBinSize_rm

        sf1 = scipy.fft.fft2(array1)
        sf2 = scipy.fft.fft2(array2)

        c11 = self._integrateRings(rings, numpy.multiply(sf1, numpy.conj(sf1)))
        c12 = self._integrateRings(rings, numpy.multiply(sf1, numpy.conj(sf2)))
        c22 = self._integrateRings(rings, numpy.multiply(sf2, numpy.conj(sf2)))

        correlation = numpy.absolute(c12) / numpy.sqrt(numpy.absolute(numpy.multiply(c11, c22)))

        # TODO replace NaNs with interpolated values

        rnyquist = numpy.min(array1.shape) // 2 + 1
        return FourierRingCorrelation(spatialFrequency_rm[:rnyquist], correlation[:rnyquist])
