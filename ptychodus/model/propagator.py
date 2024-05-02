from typing import Final

from scipy.fft import fft2, fftshift, ifft2, ifftshift
import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import WavefieldArrayType


def fresnel_propagate(inputWavefield: WavefieldArrayType, pixelGeometry: PixelGeometry,
                      propagationDistanceInMeters: float,
                      wavelengthInMeters: float) -> WavefieldArrayType:
    EPS: Final[float] = float(numpy.finfo(float).eps)
    z = numpy.abs(propagationDistanceInMeters)

    if z < EPS:
        return inputWavefield

    halfWidth = (inputWavefield.shape[-1] + 1) // 2
    halfHeight = (inputWavefield.shape[-2] + 1) // 2

    idx0 = numpy.arange(-halfWidth, halfWidth)
    idx1 = numpy.arange(-halfHeight, halfHeight)

    lambdaz = wavelengthInMeters * z

    # real space pixel size & coordinate grid
    dx = pixelGeometry.widthInMeters
    dy = pixelGeometry.heightInMeters
    YY, XX = numpy.meshgrid(idx1 * dy, idx0 * dx)

    # reciprocal space pixel size & coordinate grid
    rdx = lambdaz / dx / inputWavefield.shape[-1]
    rdy = lambdaz / dy / inputWavefield.shape[-2]
    FY, FX = numpy.meshgrid(idx1 * rdy, idx0 * rdx)

    # common quantities
    ik = 2j * numpy.pi / wavelengthInMeters
    ik_2z = ik / (2 * z)
    A = numpy.exp(ik * z) / (1j * lambdaz)
    B = numpy.exp(ik_2z * (FX**2 + FY**2))
    C = numpy.exp(ik_2z * (XX**2 + YY**2))
    D = A * B

    if propagationDistanceInMeters < 0:
        return fftshift(ifft2(ifftshift(inputWavefield / D), norm='ortho')) / C

    return D * fftshift(fft2(ifftshift(inputWavefield * C), norm='ortho'))
