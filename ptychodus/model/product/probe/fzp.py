from __future__ import annotations
from collections.abc import Iterator

import numpy
import numpy.typing

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, Probe, ProbeGeometryProvider

from ...propagator import fresnel_propagate
from .builder import ProbeBuilder


class FresnelZonePlateProbeBuilder(ProbeBuilder):

    def __init__(self, fresnelZonePlateChooser: PluginChooser[FresnelZonePlate]) -> None:
        super().__init__('fresnel_zone_plate')
        self._fresnelZonePlateChooser = fresnelZonePlateChooser

        self.zonePlateDiameterInMeters = self._registerRealParameter(
            'zone_plate_diameter_m',
            180e-6,
            minimum=0.,
        )
        self.outermostZoneWidthInMeters = self._registerRealParameter(
            'outermost_zone_width_m',
            50e-9,
            minimum=0.,
        )
        self.centralBeamstopDiameterInMeters = self._registerRealParameter(
            'central_beamstop_diameter_m',
            60e-6,
            minimum=0.,
        )
        self.defocusDistanceInMeters = self._registerRealParameter(
            'defocus_distance_m',
            800e-6,
        )  # from sample to the focal plane

    def copy(self) -> FresnelZonePlateProbeBuilder:
        builder = FresnelZonePlateProbeBuilder(self._fresnelZonePlateChooser)
        builder.zonePlateDiameterInMeters.setValue(self.zonePlateDiameterInMeters.getValue())
        builder.outermostZoneWidthInMeters.setValue(self.outermostZoneWidthInMeters.getValue())
        builder.centralBeamstopDiameterInMeters.setValue(
            self.centralBeamstopDiameterInMeters.getValue())
        builder.defocusDistanceInMeters.setValue(self.defocusDistanceInMeters.getValue())
        return builder

    def labelsForPresets(self) -> Iterator[str]:
        for entry in self._fresnelZonePlateChooser:
            yield entry.displayName

    def applyPresets(self, index: int) -> None:
        fzp = self._fresnelZonePlateChooser[index].strategy
        self.zonePlateDiameterInMeters.setValue(fzp.zonePlateDiameterInMeters)
        self.outermostZoneWidthInMeters.setValue(fzp.outermostZoneWidthInMeters)
        self.centralBeamstopDiameterInMeters.setValue(fzp.centralBeamstopDiameterInMeters)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        geometry = geometryProvider.getProbeGeometry()

        # central wavelength
        wavelength = geometryProvider.probeWavelengthInMeters

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

        array = fresnel_propagate(fzpTransmissionFunction, dx_fzp,
                                  FL + self.defocusDistanceInMeters.getValue(), wavelength)

        return Probe(
            array=self.normalize(array),
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
        )
