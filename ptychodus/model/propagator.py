from typing import Final

from scipy.fft import fft2, fftshift, ifft2, ifftshift
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import WavefieldArrayType
from ptychodus.api.typing import RealArrayType


class FresnelPropagator:
    EPS: Final[float] = float(numpy.finfo(float).eps)

    @staticmethod
    def _create_coordinates(sz: int, pixelSizeInMeters: float) -> RealArrayType:
        return numpy.arange(-(sz // 2), (sz + 1) // 2) * pixelSizeInMeters

    def __init__(self, arrayShape: tuple[int, ...], pixelGeometry: PixelGeometry,
                 propagationDistanceInMeters: float, wavelengthInMeters: float) -> None:
        dx = pixelGeometry.widthInMeters
        dy = pixelGeometry.heightInMeters
        ik = 2j * numpy.pi / wavelengthInMeters

        try:
            ik_2z = ik / (2 * propagationDistanceInMeters)
        except ZeroDivisionError:
            ik_2z = 0

        # real space pixel size & coordinate grid
        x = self._create_coordinates(arrayShape[-1], dx)
        y = self._create_coordinates(arrayShape[-2], dy)
        YY, XX = numpy.meshgrid(y, x)

        # reciprocal space pixel size & coordinate grid
        lz = numpy.abs(wavelengthInMeters * propagationDistanceInMeters)
        fx = self._create_coordinates(arrayShape[-1], lz / (arrayShape[-1] * dx))
        fy = self._create_coordinates(arrayShape[-2], lz / (arrayShape[-2] * dy))
        FY, FX = numpy.meshgrid(fy, fx)

        # propagation quantities
        self._propagationDistanceInMeters = propagationDistanceInMeters
        self._A = numpy.exp(ik_2z * (XX**2 + YY**2))
        self._B = numpy.exp(ik_2z * (FX**2 + FY**2))
        self._eikz = numpy.exp(ik * propagationDistanceInMeters)

    def propagate(self, inputWavefield: WavefieldArrayType) -> WavefieldArrayType:
        if self._propagationDistanceInMeters < 0:
            Aeikz = self._A * self._eikz
            return Aeikz * fftshift(ifft2(ifftshift(self._B * inputWavefield), norm='ortho'))
        elif self._propagationDistanceInMeters > 0:
            Beikz = self._B * self._eikz
            return Beikz * fftshift(fft2(ifftshift(self._A * inputWavefield), norm='ortho'))

        return inputWavefield
