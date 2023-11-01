from __future__ import annotations
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any, Final

import numpy
import numpy.typing

from ...api.apparatus import FresnelZonePlate
from ...api.probe import Probe
from .apparatus import Apparatus
from .repository import ProbeInitializer
from .settings import ProbeSettings
from .sizer import ProbeSizer


def gaussian_spectrum(lambda0: float, bandwidth: float, energy: int) -> numpy.typing.NDArray[Any]:
    spectrum = numpy.zeros((energy, 2))
    sigma = lambda0 * bandwidth / 2.355
    d_lam = sigma * 4 / (energy - 1)
    spectrum[:, 0] = numpy.arange(-1 * numpy.floor(energy / 2), numpy.ceil(
        energy / 2)) * d_lam + lambda0
    spectrum[:, 1] = numpy.exp(-(spectrum[:, 0] - lambda0)**2 / sigma**2)
    return spectrum


def fzp_calculate(wavelength: float, dis_defocus: float, probeSize: int, dx: float,
                  zonePlate: FresnelZonePlate) -> tuple[numpy.typing.NDArray[Any], float, float]:
    """
    this function can calculate the transfer function of zone plate
    return the transfer function, and the pixel sizes
    """

    FL = zonePlate.focalLengthInMeters(wavelength)

    # pixel size on FZP plane
    dx_fzp = wavelength * (FL + dis_defocus) / probeSize / dx

    # coordinate on FZP plane
    lx_fzp = -float(dx_fzp) * numpy.arange(-1 * numpy.floor(probeSize / 2),
                                           numpy.ceil(probeSize / 2))

    XX_FZP, YY_FZP = numpy.meshgrid(lx_fzp, lx_fzp)

    # transmission function of FZP
    T = numpy.exp(-1j * 2 * numpy.pi / float(wavelength) * (XX_FZP**2 + YY_FZP**2) / 2 / float(FL))
    C = numpy.sqrt(XX_FZP**2 + YY_FZP**2) <= zonePlate.zonePlateDiameterInMeters / 2
    H = numpy.sqrt(XX_FZP**2 + YY_FZP**2) >= zonePlate.centralBeamstopDiameterInMeters / 2

    return T * C * H, dx_fzp, FL


def fresnel_propagation(input: numpy.typing.NDArray[Any], dxy: float, z: float,
                        wavelength: float) -> numpy.typing.NDArray[Any]:
    """
    This is the python version code for fresnel propagation
    Summary of this function goes here
    Parameters:    dx,dy  -> the pixel pitch of the object
                z      -> the distance of the propagation
                lambda -> the wave length
                X,Y    -> meshgrid of coordinate
                input     -> input object
    """

    (M, N) = input.shape
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
        kern = input * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.fft2(numpy.fft.fftshift(kern))
        OUT = numpy.fft.fftshift(cgh * numpy.fft.fftshift(pf))
    else:
        pf = numpy.exp(1j * k * z) * numpy.exp(1j * k * (XX**2 + YY**2) / 2 / z)
        cgh = numpy.fft.ifft2(
            numpy.fft.fftshift(input * numpy.exp(1j * k * (Fx**2 + Fy**2) / 2 / z)))
        OUT = numpy.fft.fftshift(cgh) * pf
    return OUT


class FresnelZonePlateProbeInitializer(ProbeInitializer):
    SIMPLE_NAME: Final[str] = 'FresnelZonePlate'
    DISPLAY_NAME: Final[str] = 'Fresnel Zone Plate'
    CUSTOM_PRESET: Final[str] = 'Custom'

    def __init__(self, sizer: ProbeSizer, apparatus: Apparatus) -> None:
        super().__init__()
        self._sizer = sizer
        self._apparatus = apparatus
        self._fzpDict: Mapping[str, FresnelZonePlate] = {
            'Velociprobe': FresnelZonePlate(180e-6, 50e-9, 60e-6),
            '2-ID-D': FresnelZonePlate(160e-6, 70e-9, 60e-6),
            'LYNX': FresnelZonePlate(114.8e-6, 60e-9, 40e-6),
            'HXN': FresnelZonePlate(160e-6, 30e-9, 80e-6),
        }
        self._presets = self.CUSTOM_PRESET
        self._custom = FresnelZonePlate(180e-6, 50e-9, 60e-6)
        self._defocusDistanceInMeters = 800e-6  # from sample to the focal plane

    def syncFromSettings(self, settings: ProbeSettings) -> None:
        self._presets = settings.zonePlate.value
        self._custom = FresnelZonePlate(
            zonePlateDiameterInMeters=float(settings.zonePlateDiameterInMeters.value),
            outermostZoneWidthInMeters=float(settings.outermostZoneWidthInMeters.value),
            centralBeamstopDiameterInMeters=float(settings.centralBeamstopDiameterInMeters.value),
        )
        self._defocusDistanceInMeters = float(settings.defocusDistanceInMeters.value)
        super().syncFromSettings(settings)

    def syncToSettings(self, settings: ProbeSettings) -> None:
        settings.zonePlate.value = self._presets
        settings.zonePlateDiameterInMeters.value = \
                Decimal.from_float(self._custom.zonePlateDiameterInMeters)
        settings.outermostZoneWidthInMeters.value = \
                Decimal.from_float(self._custom.outermostZoneWidthInMeters)
        settings.centralBeamstopDiameterInMeters.value = \
                Decimal.from_float(self._custom.centralBeamstopDiameterInMeters)
        settings.defocusDistanceInMeters.value = \
                Decimal.from_float(self._defocusDistanceInMeters)
        super().syncToSettings(settings)

    @property
    def simpleName(self) -> str:
        return self.SIMPLE_NAME

    @property
    def displayName(self) -> str:
        return self.DISPLAY_NAME

    def __call__(self) -> Probe:
        arrayExtent = self._sizer.getExtentInPixels()
        array = numpy.zeros(arrayExtent.shape, dtype=complex)

        # central wavelength
        lambda0 = self._apparatus.getProbeWavelengthInMeters()

        # pixel size on sample plane (TODO non-square pixels are unsupported)
        dx = self._apparatus.getObjectPlanePixelGeometry().widthInMeters

        T, dx_fzp, FL0 = fzp_calculate(lambda0, self._defocusDistanceInMeters, arrayExtent.width,
                                       dx, self._custom)

        nprobe = fresnel_propagation(T, float(dx_fzp),
                                     (float(FL0) + float(self._defocusDistanceInMeters)),
                                     float(lambda0))

        # return probe sorted by the spectrum
        # return scale is the wavelength dependent pixel scaling factor
        array = nprobe / (numpy.sqrt(numpy.sum(numpy.abs(nprobe)**2)))

        return Probe(array)

    def getPresetsList(self) -> Sequence[str]:
        return [self.CUSTOM_PRESET, *self._fzpDict.keys()]

    def setPresets(self, presets: str) -> None:
        if self._presets != presets:
            self._presets = presets
            self.notifyObservers()

    def getPresets(self) -> str:
        return self._presets if self._presets in self._fzpDict else self.CUSTOM_PRESET

    @property
    def _fzp(self) -> FresnelZonePlate:
        try:
            return self._fzpDict[self._presets]
        except KeyError:
            return self._custom

    def setZonePlateDiameterInMeters(self, value: Decimal) -> None:
        if self._custom.zonePlateDiameterInMeters != value:
            self._custom = FresnelZonePlate(
                zonePlateDiameterInMeters=float(value),
                outermostZoneWidthInMeters=self._custom.outermostZoneWidthInMeters,
                centralBeamstopDiameterInMeters=self._custom.centralBeamstopDiameterInMeters,
            )
            self.notifyObservers()

    def getZonePlateDiameterInMeters(self) -> Decimal:
        return Decimal.from_float(self._fzp.zonePlateDiameterInMeters)

    def setOutermostZoneWidthInMeters(self, value: Decimal) -> None:
        if self._custom.outermostZoneWidthInMeters != value:
            self._custom = FresnelZonePlate(
                zonePlateDiameterInMeters=self._custom.zonePlateDiameterInMeters,
                outermostZoneWidthInMeters=float(value),
                centralBeamstopDiameterInMeters=self._custom.centralBeamstopDiameterInMeters,
            )
            self.notifyObservers()

    def getOutermostZoneWidthInMeters(self) -> Decimal:
        return Decimal.from_float(self._fzp.outermostZoneWidthInMeters)

    def setCentralBeamstopDiameterInMeters(self, value: Decimal) -> None:
        if self._custom.centralBeamstopDiameterInMeters != value:
            self._custom = FresnelZonePlate(
                zonePlateDiameterInMeters=self._custom.zonePlateDiameterInMeters,
                outermostZoneWidthInMeters=self._custom.outermostZoneWidthInMeters,
                centralBeamstopDiameterInMeters=float(value),
            )
            self.notifyObservers()

    def getCentralBeamstopDiameterInMeters(self) -> Decimal:
        return Decimal.from_float(self._fzp.centralBeamstopDiameterInMeters)

    def setDefocusDistanceInMeters(self, value: Decimal) -> None:
        if self._defocusDistanceInMeters != value:
            self._defocusDistanceInMeters = float(value)
            self.notifyObservers()

    def getDefocusDistanceInMeters(self) -> Decimal:
        return Decimal.from_float(self._defocusDistanceInMeters)
