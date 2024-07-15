from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy

from .typing import ComplexArrayType, RealArrayType

WavefieldArrayType: TypeAlias = ComplexArrayType


def intensity(wavefield: WavefieldArrayType) -> RealArrayType:
    return numpy.real(numpy.multiply(wavefield, numpy.conjugate(wavefield)))


@dataclass(frozen=True)
class PropagatorParameters:
    wavelength_m: float
    '''illumination wavelength in meters'''
    width_px: int
    '''number of pixels in the x-direction'''
    height_px: int
    '''number of pixels in the y-direction'''
    pixel_width_m: float
    '''source plane pixel width in meters'''
    pixel_height_m: float
    '''source plane pixel height in meters'''
    propagation_distance_m: float
    '''propagation distance in meters'''

    @property
    def dx(self) -> float:
        '''pixel width in wavelengths'''
        return self.pixel_width_m / self.wavelength_m

    @property
    def pixel_aspect_ratio(self) -> float:
        '''pixel aspect ratio (width / height)'''
        return self.pixel_width_m / self.pixel_height_m

    @property
    def z(self) -> float:
        '''propagation distance in wavelengths'''
        return self.propagation_distance_m / self.wavelength_m

    @property
    def fresnel_number(self) -> float:
        '''fresnel number'''
        return numpy.square(self.dx) / self.z

    def get_spatial_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        JJ, II = numpy.mgrid[:self.height_px, :self.width_px]
        XX = II - self.width_px // 2
        YY = JJ - self.height_px // 2
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        fx = fftshift(fftfreq(self.width_px))
        fy = fftshift(fftfreq(self.height_px))
        FY, FX = numpy.meshgrid(fy, fx)
        return FY, FX


class Propagator(ABC):

    @abstractmethod
    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        pass


class AngularSpectrumPropagator(Propagator):

    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * numpy.pi * parameters.z
        FY, FX = parameters.get_frequency_coordinates()
        F2 = numpy.square(FX) + numpy.square(ar * FY)
        ratio = F2 / numpy.square(parameters.dx)
        tf = numpy.exp(i2piz * numpy.sqrt(1 - ratio))

        self._transfer_function = numpy.where(ratio < 1, tf, 0)

    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransferFunctionPropagator(Propagator):

    def __init__(self, parameters: PropagatorParameters) -> None:
        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio

        i2pi = 2j * numpy.pi
        FY, FX = parameters.get_frequency_coordinates()
        F2 = numpy.square(FX) + numpy.square(ar * FY)

        self._transfer_function = numpy.exp(i2pi * (parameters.z - 0.5 * F2 / Fr))

    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransformPropagator(Propagator):

    def __init__(self, parameters: PropagatorParameters) -> None:
        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio

        N = parameters.width_px
        M = parameters.height_px
        C = numpy.exp(2j * numpy.pi * parameters.z) * Fr / (1j * ar)
        ipi = 1j * numpy.pi

        YY, XX = parameters.get_spatial_coordinates()

        self._A = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr) * C
        self._B = numpy.exp(ipi * Fr * (numpy.square(XX) + numpy.square(YY / ar)))

    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        return self._A * fftshift(fft2(ifftshift(wavefield * self._B)))


class FresnelTransformLegacyPropagator(Propagator):

    def __init__(self, parameters: PropagatorParameters) -> None:
        self._parameters = parameters

    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        dxy = self._parameters.pixel_width_m
        z = self._parameters.propagation_distance_m
        wavelength = self._parameters.wavelength_m

        (M, N) = wavefield.shape
        k = 2 * numpy.pi / wavelength

        # the coordinate grid
        M_grid = numpy.arange(-1 * numpy.floor(M / 2), numpy.ceil(M / 2))
        N_grid = numpy.arange(-1 * numpy.floor(N / 2), numpy.ceil(N / 2))
        lx = M_grid * dxy
        ly = N_grid * dxy

        XX, YY = numpy.meshgrid(lx, ly)

        # the coordinate grid on the output plane
        fc = 1 / dxy
        fu = wavelength * z * fc
        lu = M_grid * fu / M
        lv = N_grid * fu / N
        Fx, Fy = numpy.meshgrid(lu, lv)

        if z > 0:
            pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)
            kern = wavefield * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
            cgh = fft2(fftshift(kern))
            OUT = fftshift(cgh * fftshift(pf))
        else:
            pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
            cgh = ifft2(fftshift(wavefield * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)))
            OUT = fftshift(cgh) * pf

        return OUT


class FraunhoferPropagator(Propagator):

    def __init__(self, parameters: PropagatorParameters) -> None:
        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio

        N = parameters.width_px
        M = parameters.height_px
        C = numpy.exp(2j * numpy.pi * parameters.z) * Fr / (1j * ar)
        ipi = 1j * numpy.pi

        YY, XX = parameters.get_spatial_coordinates()

        self._A = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr) * C

    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        return self._A * fftshift(fft2(ifftshift(wavefield)))
