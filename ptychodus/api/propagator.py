from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy

from .typing import ComplexArrayType, RealArrayType

WavefieldArrayType: TypeAlias = ComplexArrayType


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

    @property
    def diffraction_plane_pixel_width(self) -> float:
        '''diffraction plane pixel width in wavelengths'''
        return self.z / (self.width_px * self.dx)

    def get_spatial_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        jj, ii = numpy.mgrid[:self.height_px, :self.width_px]
        xx = ii - (self.width_px - 1) / 2
        yy = jj - (self.height_px - 1) / 2
        return yy, xx

    def get_frequency_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        fx = fftfreq(self.width_px)
        fy = fftfreq(self.height_px)
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
        tf = numpy.exp(i2piz * numpy.sqrt(1 - ratio)),

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

        i2pi = 2j * numpy.pi
        ipiFr = 1j * numpy.pi * Fr
        iar = 1j * ar

        YY, XX = parameters.get_spatial_coordinates()
        FY, FX = parameters.get_frequency_coordinates()
        F2 = numpy.square(FX) + numpy.square(ar * FY)

        self._A = numpy.exp(-i2pi * (parameters.z - 0.5 * F2 / Fr)) * Fr / iar
        self._B = numpy.exp(-ipiFr * (numpy.square(XX) + numpy.square(YY) / numpy.square(ar)))

    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        return self._A * fftshift(fft2(ifftshift(wavefield * self._B)))


class FraunhoferPropagator(Propagator):

    def __init__(self, parameters: PropagatorParameters) -> None:
        Fr = parameters.fresnel_number
        ar = parameters.pixel_aspect_ratio

        i2pi = 2j * numpy.pi
        iar = 1j * ar

        FY, FX = parameters.get_frequency_coordinates()
        F2 = numpy.square(FX) + numpy.square(ar * FY)

        self._A = numpy.exp(-i2pi * (parameters.z - 0.5 * F2 / Fr)) * Fr / iar

    def propagate(self, wavefield: WavefieldArrayType) -> WavefieldArrayType:
        return self._A * fftshift(fft2(ifftshift(wavefield)))
