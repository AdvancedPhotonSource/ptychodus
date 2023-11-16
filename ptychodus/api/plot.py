from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy
import numpy.typing
import scipy.fft

from .apparatus import PixelGeometry

ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]
IntegerArrayType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]


@dataclass(frozen=True)
class PlotSeries:
    label: str
    values: Sequence[float]


@dataclass(frozen=True)
class PlotAxis:
    label: str
    series: Sequence[PlotSeries]

    @classmethod
    def createNull(cls) -> PlotAxis:
        return cls('', [])


@dataclass(frozen=True)
class Plot2D:
    axisX: PlotAxis
    axisY: PlotAxis

    @classmethod
    def createNull(cls) -> Plot2D:
        return cls(PlotAxis.createNull(), PlotAxis.createNull())


@dataclass(frozen=True)
class LineCut:
    distanceInMeters: Sequence[float]
    value: Sequence[float]
    valueLabel: str


@dataclass(frozen=True)
class FourierRingCorrelation:
    spatialFrequency_rm: Sequence[float]
    correlation: Sequence[float]

    @staticmethod
    def _integrateRings(rings: IntegerArrayType, array: ComplexArrayType) -> ComplexArrayType:
        total = numpy.zeros(numpy.max(rings) + 1, dtype=complex)

        for index, value in zip(rings.flat, array.flat):
            total[index] += value

        return total

    @classmethod
    def calculate(cls, image1: ComplexArrayType, image2: ComplexArrayType,
                  pixelGeometry: PixelGeometry) -> FourierRingCorrelation:
        '''
        See: Joan Vila-Comamala, Ana Diaz, Manuel Guizar-Sicairos, Alexandre Mantion,
        Cameron M. Kewish, Andreas Menzel, Oliver Bunk, and Christian David,
        "Characterization of high-resolution diffractive X-ray optics by ptychographic
        coherent diffractive imaging," Opt. Express 19, 21333-21344 (2011)
        '''

        if numpy.ndim(image1) != 2 or numpy.ndim(image2) != 2:
            raise ValueError('Images must be 2D!')

        if numpy.shape(image1) != numpy.shape(image2):
            raise ValueError('Images must have same shape!')

        # TODO subpixel image registration
        # TODO remove phase offset and ramp
        # TODO apply soft-edged mask
        # TODO stats: SSNR, area under FRC curve, average SNR, etc.

        x_rm = scipy.fft.fftfreq(image1.shape[-1], d=pixelGeometry.widthInMeters)
        y_rm = scipy.fft.fftfreq(image1.shape[-2], d=pixelGeometry.heightInMeters)
        radialBinSize_rm = max(x_rm.max(), y_rm.max())

        xx_rm, yy_rm = numpy.meshgrid(x_rm, y_rm)
        rr_rm = numpy.hypot(xx_rm, yy_rm)

        rings = numpy.divide(rr_rm, radialBinSize_rm).astype(int)
        spatialFrequency_rm = numpy.arange(numpy.max(rings) + 1) * radialBinSize_rm

        sf1 = scipy.fft.fft2(image1)
        sf2 = scipy.fft.fft2(image2)

        c11 = cls._integrateRings(rings, numpy.multiply(sf1, numpy.conj(sf1)))
        c12 = cls._integrateRings(rings, numpy.multiply(sf1, numpy.conj(sf2)))
        c22 = cls._integrateRings(rings, numpy.multiply(sf2, numpy.conj(sf2)))

        correlation = numpy.absolute(c12) / numpy.sqrt(numpy.absolute(numpy.multiply(c11, c22)))

        # TODO replace NaNs with interpolated values

        return cls(spatialFrequency_rm, correlation)

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
