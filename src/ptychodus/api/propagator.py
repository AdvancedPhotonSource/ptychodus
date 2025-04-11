from abc import ABC, abstractmethod
from dataclasses import dataclass

from scipy.fft import fft2, fftfreq, fftshift, ifft2, ifftshift
import numpy

from .typing import ComplexArrayType, RealArrayType


def intensity(wavefield: ComplexArrayType) -> RealArrayType:
    return numpy.real(numpy.multiply(wavefield, numpy.conjugate(wavefield)))


@dataclass(frozen=True)
class PropagatorParameters:
    wavelength_m: float
    """illumination wavelength in meters"""
    width_px: int
    """number of pixels in the x-direction"""
    height_px: int
    """number of pixels in the y-direction"""
    pixel_width_m: float
    """source plane pixel width in meters"""
    pixel_height_m: float
    """source plane pixel height in meters"""
    propagation_distance_m: float
    """propagation distance in meters"""

    @property
    def dx(self) -> float:
        """pixel width in wavelengths"""
        return self.pixel_width_m / self.wavelength_m

    @property
    def pixel_aspect_ratio(self) -> float:
        """pixel aspect ratio (width / height)"""
        return self.pixel_width_m / self.pixel_height_m

    @property
    def z(self) -> float:
        """propagation distance in wavelengths"""
        return self.propagation_distance_m / self.wavelength_m

    @property
    def fresnel_number(self) -> float:
        """fresnel number"""
        return numpy.square(self.dx) / numpy.absolute(self.z)

    def get_spatial_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        JJ, II = numpy.mgrid[: self.height_px, : self.width_px]  # noqa: N806
        XX = II - self.width_px // 2  # noqa: N806
        YY = JJ - self.height_px // 2  # noqa: N806
        return YY, XX

    def get_frequency_coordinates(self) -> tuple[RealArrayType, RealArrayType]:
        fx = fftshift(fftfreq(self.width_px))
        fy = fftshift(fftfreq(self.height_px))
        FY, FX = numpy.meshgrid(fy, fx, indexing='ij')  # noqa: N806
        return FY, FX


class Propagator(ABC):
    @abstractmethod
    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        pass


class AngularSpectrumPropagator(Propagator):
    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * numpy.pi * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)
        tf = numpy.exp(i2piz * numpy.sqrt(1 - ratio))

        self._transfer_function = numpy.where(ratio < 1, tf, 0)

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransferFunctionPropagator(Propagator):
    def __init__(self, parameters: PropagatorParameters) -> None:
        ar = parameters.pixel_aspect_ratio

        i2piz = 2j * numpy.pi * parameters.z
        FY, FX = parameters.get_frequency_coordinates()  # noqa: N806
        F2 = numpy.square(FX) + numpy.square(ar * FY)  # noqa: N806
        ratio = F2 / numpy.square(parameters.dx)

        self._transfer_function = numpy.exp(i2piz * (1 - ratio / 2))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        return fftshift(ifft2(self._transfer_function * fft2(ifftshift(wavefield))))


class FresnelTransformPropagator(Propagator):
    def __init__(self, parameters: PropagatorParameters) -> None:
        ipi = 1j * numpy.pi

        Fr = parameters.fresnel_number  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        C0 = Fr / (1j * ar)  # noqa: N806
        C1 = numpy.exp(2j * numpy.pi * parameters.z)  # noqa: N806
        C2 = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr)  # noqa: N806
        is_forward = parameters.propagation_distance_m >= 0.0

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0
        self._B = numpy.exp(ipi * Fr * (numpy.square(XX) + numpy.square(YY / ar)))

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield * self._B)))
        else:
            return self._B * fftshift(ifft2(ifftshift(wavefield * self._A)))


class FraunhoferPropagator(Propagator):
    def __init__(self, parameters: PropagatorParameters) -> None:
        ipi = 1j * numpy.pi

        Fr = parameters.fresnel_number  # noqa: N806
        ar = parameters.pixel_aspect_ratio
        N = parameters.width_px  # noqa: N806
        M = parameters.height_px  # noqa: N806
        YY, XX = parameters.get_spatial_coordinates()  # noqa: N806

        C0 = Fr / (1j * ar)  # noqa: N806
        C1 = numpy.exp(2j * numpy.pi * parameters.z)  # noqa: N806
        C2 = numpy.exp((numpy.square(XX / N) + numpy.square(ar * YY / M)) * ipi / Fr)  # noqa: N806
        is_forward = parameters.propagation_distance_m >= 0.0

        self._is_forward = is_forward
        self._A = C2 * C1 * C0 if is_forward else C2 * C1 / C0

    def propagate(self, wavefield: ComplexArrayType) -> ComplexArrayType:
        if self._is_forward:
            return self._A * fftshift(fft2(ifftshift(wavefield)))
        else:
            return fftshift(ifft2(ifftshift(wavefield * self._A)))
