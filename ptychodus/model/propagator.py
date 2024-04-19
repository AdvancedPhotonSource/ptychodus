import numpy

from ptychodus.api.probe import WavefieldArrayType


def fresnel_propagate(inputWavefield: WavefieldArrayType, inputPixelSizeInMeters: float,
                      propagationDistanceInMeters: float,
                      wavelengthInMeters: float) -> WavefieldArrayType:
    (M, N) = inputWavefield.shape
    k = 2 * numpy.pi / wavelengthInMeters

    # the coordinate grid
    M_grid = numpy.arange(-1 * numpy.floor(M / 2), numpy.ceil(M / 2))
    N_grid = numpy.arange(-1 * numpy.floor(N / 2), numpy.ceil(N / 2))
    lx = M_grid * inputPixelSizeInMeters
    ly = N_grid * inputPixelSizeInMeters

    XX, YY = numpy.meshgrid(lx, ly)

    # the coordinate grid on the output plane
    fu = wavelengthInMeters * propagationDistanceInMeters / inputPixelSizeInMeters
    lu = M_grid * fu / M
    lv = N_grid * fu / N
    Fx, Fy = numpy.meshgrid(lu, lv)

    z = propagationDistanceInMeters

    if z > 0:
        pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)
        kern = inputWavefield * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.fft2(numpy.fft.fftshift(kern))
        outputWavefield = numpy.fft.fftshift(cgh * numpy.fft.fftshift(pf))
    else:
        pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.ifft2(
            numpy.fft.fftshift(inputWavefield * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)))
        outputWavefield = numpy.fft.fftshift(cgh) * pf

    return outputWavefield
