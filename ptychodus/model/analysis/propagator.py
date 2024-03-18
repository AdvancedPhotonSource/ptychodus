from __future__ import annotations
from dataclasses import dataclass
from typing import Any, TypeAlias
import logging

from scipy.fft import fft2, ifft2, fftshift, ifftshift
import numpy
import numpy.typing

from ..product import ProbeRepository

ComplexArrayType: TypeAlias = numpy.typing.NDArray[numpy.complexfloating[Any, Any]]

logger = logging.getLogger(__name__)


class WavefieldPropagator:

    # FIXME fftshift/ifftshift
    def __init__(self, wavelength_m: float, pixel_width_m: float, pixel_height_m: float) -> None:
        self._wavelength_m = wavelength_m
        self._pixel_width_m = pixel_width_m
        self._pixel_height_m = pixel_height_m

    @property
    def wave_number(self) -> float:
        return 2 * numpy.pi / self._wavelength_m

    def propagate_angular_spectrum(self, wavefield: ComplexArrayType,
                                   distance: float) -> ComplexArrayType:
        ikz = 1j * self.wave_number * distance
        wavefieldFT = fft2(wavefield)
        exactTF = numpy.exp(ikz * numpy.sqrt(1 - lambdaFr_sq))
        return ifft2(wavefieldFT * exactTF)

    def propagate_fresnel_transfer_function(self, wavefield: ComplexArrayType,
                                            distance: float) -> ComplexArrayType:
        ikz = 1j * self.wave_number * distance
        ipiz = 1j * numpy.pi * distance
        wavefieldFT = fft2(wavefield)
        fresnelTF = numpy.exp(ikz) * numpy.exp(-ipiz * lambdaFr_sq)
        return ifft2(wavefieldFT * fresnelTF)

    def propagate_fresnel_transform(self, wavefield: ComplexArrayType,
                                    distance: float) -> ComplexArrayType:

        # transverse plane indexes
        h, w = wavefield.shape
        H, W = numpy.mgrid[:h, :w]
        W = W - (w - 1) / 2
        H = H - (h - 1) / 2

        # input plane coordinates
        X0_m = W * self._pixel_width_m
        Y0_m = H * self._pixel_height_m

        FX0_rm = W * Q  # FIXME
        FY0_rm = H * Q  # FIXME

        # output plane coordinates
        X_m = W * QQ  # FIXME
        Y_m = H * QQ  # FIXME

        FX_rm = W * QQQ  # FIXME Fx = x_m / (wavelength_m * distance_m)
        FY_rm = H * QQQ  # FIXME Fy = y_m / (wavelength_m * distance_m)
        # FIXME Fr ** 2 = Fx ** 2 + Fy ** 2

        # the coordinate grid on the output plane
        fu = wavelength * z / dxy
        lu = M_grid * fu / M
        lv = N_grid * fu / N
        Fx, Fy = numpy.meshgrid(lu, lv)

        if distance > 0.:
            return  # FIXME

        if distance < 0.:
            return  # FIXME

        return wavefield

    def propagate_fourier(self, wavefield: ComplexArrayType, distance: float) -> ComplexArrayType:
        return  # FIXME


@dataclass(frozen=True)
class ProbePropagation:
    itemIndex: int  # FIXME


class ProbePropagator:

    def __init__(self, repository: ProbeRepository) -> None:
        self._repository = repository

    def getName(self, itemIndex: int) -> str:
        return self._repository.getName(itemIndex)

    def propagate(self, itemIndex: int) -> ProbePropagation:  # FIXME
        return ProbePropagation(itemIndex)
