from __future__ import annotations
from dataclasses import dataclass
from typing import Any

import numpy
import numpy.typing

from ...api.probe import Probe, ProbeGeometryProvider
from .builder import ProbeBuilder


@dataclass(frozen=True)
class FresnelZonePlate:
    zonePlateDiameterInMeters: float
    outermostZoneWidthInMeters: float
    centralBeamstopDiameterInMeters: float

    def getFocalLengthInMeters(self, centralWavelengthInMeters: float) -> float:
        return self.zonePlateDiameterInMeters * self.outermostZoneWidthInMeters \
                / centralWavelengthInMeters


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
    fu = wavelength * z / dxy
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


class FresnelZonePlateProbeBuilder(ProbeBuilder):

    def __init__(self, geometryProvider: ProbeGeometryProvider) -> None:
        super().__init__('Fresnel Zone Plate')
        self._geometryProvider = geometryProvider

        self.zonePlateDiameterInMeters = self._registerRealParameter(
            'zonePlateDiameterInMeters',
            180e-6,
            minimum=0.,
        )
        self.outermostZoneWidthInMeters = self._registerRealParameter(
            'outermostZoneWidthInMeters',
            50e-9,
            minimum=0.,
        )
        self.centralBeamstopDiameterInMeters = self._registerRealParameter(
            'centralBeamstopDiameterInMeters',
            60e-6,
            minimum=0.,
        )
        self.defocusDistanceInMeters = self._registerRealParameter(
            'defocusDistanceInMeters',
            800e-6,
        )  # from sample to the focal plane

    def build(self) -> Probe:
        geometry = self._geometryProvider.getProbeGeometry()

        # central wavelength
        wavelength = self._geometryProvider.getProbeWavelengthInMeters()

        # pixel size on sample plane (TODO non-square pixels are unsupported)
        dx = geometry.pixelWidthInMeters

        zonePlate = FresnelZonePlate(
            zonePlateDiameterInMeters=self.zonePlateDiameterInMeters.getValue(),
            outermostZoneWidthInMeters=self.outermostZoneWidthInMeters.getValue(),
            centralBeamstopDiameterInMeters=self.centralBeamstopDiameterInMeters.getValue(),
        )
        FL = zonePlate.getFocalLengthInMeters(wavelength)
        probeSize = geometry.widthInPixels  # FIXME verify

        # pixel size on FZP plane
        dis_defocus = self.defocusDistanceInMeters.getValue()
        dx_fzp = wavelength * (FL + dis_defocus) / probeSize / dx

        # coordinate on FZP plane
        lx_fzp = -dx_fzp * numpy.arange(-1 * numpy.floor(probeSize / 2), numpy.ceil(probeSize / 2))

        # FIXME y, x = numpy.mgrid[:ny,:nx]
        XX_FZP, YY_FZP = numpy.meshgrid(lx_fzp, lx_fzp)
        RR_FZP = numpy.hypot(XX_FZP, YY_FZP)

        # transmission function of FZP
        T = numpy.exp(-1j * 2 * numpy.pi / wavelength * (XX_FZP**2 + YY_FZP**2) / 2 / FL)
        C = RR_FZP <= zonePlate.zonePlateDiameterInMeters / 2
        H = RR_FZP >= zonePlate.centralBeamstopDiameterInMeters / 2
        fzpTransmissionFunction = T * C * H

        array = fresnel_propagation(fzpTransmissionFunction, dx_fzp,
                                    FL + self.defocusDistanceInMeters.getValue(), wavelength)

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )


# FIXME self._custom = FresnelZonePlate(180e-6, 50e-9, 60e-6)
# FIXME self._fzpDict: Mapping[str, FresnelZonePlate] = {
# FIXME     'Velociprobe': FresnelZonePlate(180e-6, 50e-9, 60e-6),
# FIXME     '2-ID-D': FresnelZonePlate(160e-6, 70e-9, 60e-6),
# FIXME     'LYNX': FresnelZonePlate(114.8e-6, 60e-9, 40e-6),
# FIXME     'HXN': FresnelZonePlate(160e-6, 30e-9, 80e-6),
# FIXME }
