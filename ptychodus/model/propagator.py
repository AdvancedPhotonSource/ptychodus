from typing import Any, Final, TypeAlias

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import WavefieldArrayType

__all__ = [
    'fresnel_propagate',
]

RealArrayType: TypeAlias = numpy.typing.NDArray[numpy.floating[Any]]


def _ifftshift_coords(sz: int, pixelSizeInMeters: float) -> RealArrayType:
    return ifftshift(numpy.arange(-(sz // 2), (sz + 1) // 2)) * pixelSizeInMeters


def fresnel_propagate(inputWavefield: WavefieldArrayType, pixelGeometry: PixelGeometry,
                      propagationDistanceInMeters: float,
                      wavelengthInMeters: float) -> WavefieldArrayType:
    EPS: Final[float] = float(numpy.finfo(float).eps)
    z = numpy.abs(propagationDistanceInMeters)

    if z < EPS:
        return inputWavefield

    dx = pixelGeometry.widthInMeters
    dy = pixelGeometry.heightInMeters
    lz = wavelengthInMeters * z

    # real space pixel size & coordinate grid
    x = _ifftshift_coords(inputWavefield.shape[-1], dx)
    y = _ifftshift_coords(inputWavefield.shape[-2], dy)
    YY, XX = numpy.meshgrid(y, x)

    # reciprocal space pixel size & coordinate grid
    fx = fftfreq(inputWavefield.shape[-1], d=dx / lz)
    fy = fftfreq(inputWavefield.shape[-2], d=dy / lz)
    FY, FX = numpy.meshgrid(fy, fx)

    # common quantities in result
    ik = 2j * numpy.pi / wavelengthInMeters
    ik_2z = ik / (2 * propagationDistanceInMeters)
    A = numpy.exp(ik_2z * (XX**2 + YY**2))
    B = numpy.exp(ik_2z * (FX**2 + FY**2))
    eikz = numpy.exp(ik * propagationDistanceInMeters)
    shiftedWavefield = ifftshift(inputWavefield)

    if propagationDistanceInMeters < 0:
        result = fftshift(A * eikz * ifft2(B * shiftedWavefield, norm='ortho'))
    else:
        result = fftshift(B * eikz * fft2(A * shiftedWavefield, norm='ortho'))

    return result
