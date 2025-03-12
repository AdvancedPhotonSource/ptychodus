from __future__ import annotations
from collections.abc import Iterator

import numpy
import numpy.typing

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, Probe, ProbeGeometryProvider
from ptychodus.api.propagator import FresnelTransformPropagator, PropagatorParameters

from .builder import ProbeBuilder
from .settings import ProbeSettings


class FresnelZonePlateProbeBuilder(ProbeBuilder):
    def __init__(
        self,
        settings: ProbeSettings,
        fresnelZonePlateChooser: PluginChooser[FresnelZonePlate],
    ) -> None:
        super().__init__(settings, 'fresnel_zone_plate')
        self._settings = settings
        self._fresnelZonePlateChooser = fresnelZonePlateChooser

        self.zonePlateDiameterInMeters = settings.zonePlateDiameterInMeters.copy()
        self._add_parameter('zone_plate_diameter_m', self.zonePlateDiameterInMeters)

        self.outermostZoneWidthInMeters = settings.outermostZoneWidthInMeters.copy()
        self._add_parameter('outermost_zone_width_m', self.outermostZoneWidthInMeters)

        self.centralBeamstopDiameterInMeters = settings.centralBeamstopDiameterInMeters.copy()
        self._add_parameter('central_beamstop_diameter_m', self.centralBeamstopDiameterInMeters)

        # from sample to the focal plane
        self.defocusDistanceInMeters = settings.defocusDistanceInMeters.copy()
        self._add_parameter('defocus_distance_m', self.defocusDistanceInMeters)

    def copy(self) -> FresnelZonePlateProbeBuilder:
        builder = FresnelZonePlateProbeBuilder(self._settings, self._fresnelZonePlateChooser)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def labelsForPresets(self) -> Iterator[str]:
        for plugin in self._fresnelZonePlateChooser:
            yield plugin.display_name

    def applyPresets(self, display_name: str) -> None:
        self._fresnelZonePlateChooser.set_current_plugin(display_name)
        fzp = self._fresnelZonePlateChooser.get_current_plugin().strategy
        self.zonePlateDiameterInMeters.set_value(fzp.zone_plate_diameter_m)
        self.outermostZoneWidthInMeters.set_value(fzp.outermost_zone_width_m)
        self.centralBeamstopDiameterInMeters.set_value(fzp.central_beamstop_diameter_m)

    def build(self, geometryProvider: ProbeGeometryProvider) -> Probe:
        wavelengthInMeters = geometryProvider.probe_wavelength_m
        zonePlate = FresnelZonePlate(
            zone_plate_diameter_m=self.zonePlateDiameterInMeters.get_value(),
            outermost_zone_width_m=self.outermostZoneWidthInMeters.get_value(),
            central_beamstop_diameter_m=self.centralBeamstopDiameterInMeters.get_value(),
        )
        focalLengthInMeters = zonePlate.get_focal_length_m(wavelengthInMeters)
        distanceInMeters = focalLengthInMeters + self.defocusDistanceInMeters.get_value()
        samplePlaneGeometry = geometryProvider.get_probe_geometry()
        fzpHalfWidth = (samplePlaneGeometry.width_px + 1) // 2
        fzpHalfHeight = (samplePlaneGeometry.height_px + 1) // 2
        fzpPlanePixelSizeNumerator = wavelengthInMeters * distanceInMeters
        fzpPixelGeometry = PixelGeometry(
            width_m=fzpPlanePixelSizeNumerator / samplePlaneGeometry.width_m,
            height_m=fzpPlanePixelSizeNumerator / samplePlaneGeometry.height_m,
        )

        # coordinate on FZP plane
        lx_fzp = -fzpPixelGeometry.width_m * numpy.arange(-fzpHalfWidth, fzpHalfWidth)
        ly_fzp = -fzpPixelGeometry.height_m * numpy.arange(-fzpHalfHeight, fzpHalfHeight)

        YY_FZP, XX_FZP = numpy.meshgrid(ly_fzp, lx_fzp)
        RR_FZP = numpy.hypot(XX_FZP, YY_FZP)

        # transmission function of FZP
        T = numpy.exp(
            -2j * numpy.pi / wavelengthInMeters * (XX_FZP**2 + YY_FZP**2) / 2 / focalLengthInMeters
        )
        C = RR_FZP <= zonePlate.zone_plate_diameter_m / 2
        H = RR_FZP >= zonePlate.central_beamstop_diameter_m / 2
        fzpTransmissionFunction = T * C * H

        propagatorParameters = PropagatorParameters(
            wavelength_m=wavelengthInMeters,
            width_px=fzpTransmissionFunction.shape[-1],
            height_px=fzpTransmissionFunction.shape[-2],
            pixel_width_m=fzpPixelGeometry.width_m,
            pixel_height_m=fzpPixelGeometry.height_m,
            propagation_distance_m=distanceInMeters,
        )
        propagator = FresnelTransformPropagator(propagatorParameters)
        array = propagator.propagate(fzpTransmissionFunction)

        return Probe(
            array=self.normalize(array),
            pixel_geometry=samplePlaneGeometry.get_pixel_geometry(),
        )
