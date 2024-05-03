from typing import Any, Final, TypeAlias

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import WavefieldArrayType

RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]


class FresnelPropagator:
    EPS: Final[float] = float(numpy.finfo(float).eps)

    @staticmethod
    def _ifftshift_coords(sz: int, pixelSizeInMeters: float) -> RealArrayType:
        return ifftshift(numpy.arange(-(sz // 2), (sz + 1) // 2)) * pixelSizeInMeters

    def __init__(self, arrayShape: tuple[int, ...], pixelGeometry: PixelGeometry,
                 propagationDistanceInMeters: float, wavelengthInMeters: float) -> None:
        dx = pixelGeometry.widthInMeters
        dy = pixelGeometry.heightInMeters
        lz = wavelengthInMeters * numpy.abs(propagationDistanceInMeters)
        ik = 2j * numpy.pi / wavelengthInMeters
        ik_2z = ik / (2 * propagationDistanceInMeters)

        # real space pixel size & coordinate grid
        x = self._ifftshift_coords(arrayShape[-1], dx)
        y = self._ifftshift_coords(arrayShape[-2], dy)
        YY, XX = numpy.meshgrid(y, x)

        # reciprocal space pixel size & coordinate grid
        fx = fftfreq(arrayShape[-1], d=dx / lz)
        fy = fftfreq(arrayShape[-2], d=dy / lz)
        FY, FX = numpy.meshgrid(fy, fx)

        # propagation quantities
        self._propagationDistanceInMeters = propagationDistanceInMeters
        self._A = numpy.exp(ik_2z * (XX**2 + YY**2))
        self._B = numpy.exp(ik_2z * (FX**2 + FY**2))
        self._eikz = numpy.exp(ik * propagationDistanceInMeters)

    def propagate(self, inputWavefield: WavefieldArrayType) -> WavefieldArrayType:
        shiftedWavefield = ifftshift(inputWavefield)

        if self._propagationDistanceInMeters > FresnelPropagator.EPS:
            Beikz = self._B * self._eikz
            return fftshift(Beikz * fft2(self._A * shiftedWavefield, norm='ortho'))

        if self._propagationDistanceInMeters < -FresnelPropagator.EPS:
            Aeikz = self._A * self._eikz
            return fftshift(Aeikz * ifft2(self._B * shiftedWavefield, norm='ortho'))

        return inputWavefield
