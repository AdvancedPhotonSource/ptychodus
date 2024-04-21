from typing import Final

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import WavefieldArrayType


def fresnel_propagate(inputWavefield: WavefieldArrayType, pixelGeometry: PixelGeometry,
                      propagationDistanceInMeters: float,
                      wavelengthInMeters: float) -> WavefieldArrayType:
    EPS: Final[float] = float(numpy.finfo(float).eps)

    halfWidth = (inputWavefield.shape[-1] + 1) // 2
    halfHeight = (inputWavefield.shape[-2] + 1) // 2
    k = 2 * numpy.pi / wavelengthInMeters

    # the coordinate grid
    M_grid = numpy.arange(-halfWidth, halfWidth)
    N_grid = numpy.arange(-halfHeight, halfHeight)

    lx = M_grid * pixelGeometry.widthInMeters
    ly = N_grid * pixelGeometry.heightInMeters

    YY, XX = numpy.meshgrid(ly, lx)

    # the coordinate grid on the output plane
    fu = wavelengthInMeters * propagationDistanceInMeters / pixelGeometry.widthInMeters
    lu = M_grid * fu / inputWavefield.shape[-1]
    fv = wavelengthInMeters * propagationDistanceInMeters / pixelGeometry.heightInMeters
    lv = N_grid * fv / inputWavefield.shape[-2]
    Fy, Fx = numpy.meshgrid(lv, lu)

    z = propagationDistanceInMeters

    if z > EPS:
        pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)
        kern = inputWavefield * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.fft2(numpy.fft.fftshift(kern))
        outputWavefield = numpy.fft.fftshift(cgh * numpy.fft.fftshift(pf))
    elif z < -EPS:  # FIXME BROKEN
        pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.ifft2(
            numpy.fft.fftshift(inputWavefield * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)))
        outputWavefield = numpy.fft.fftshift(cgh) * pf
    else:
        outputWavefield = inputWavefield

    return outputWavefield
