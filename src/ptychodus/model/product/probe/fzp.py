from __future__ import annotations
from collections.abc import Iterator

import numpy
import numpy.typing

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, ProbeSequence, ProbeGeometryProvider
from ptychodus.api.propagator import FresnelTransformPropagator, PropagatorParameters

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class FresnelZonePlateProbeBuilder(ProbeSequenceBuilder):
    def __init__(
        self,
        settings: ProbeSettings,
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
    ) -> None:
        super().__init__(settings, 'fresnel_zone_plate')
        self._settings = settings
        self._fresnel_zone_plate_chooser = fresnel_zone_plate_chooser

        self.zone_plate_diameter_m = settings.zone_plate_diameter_m.copy()
        self._add_parameter('zone_plate_diameter_m', self.zone_plate_diameter_m)

        self.outermost_zone_width_m = settings.outermost_zone_width_m.copy()
        self._add_parameter('outermost_zone_width_m', self.outermost_zone_width_m)

        self.central_beamstop_diameter_m = settings.central_beamstop_diameter_m.copy()
        self._add_parameter('central_beamstop_diameter_m', self.central_beamstop_diameter_m)

        # from sample to the focal plane
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

    def copy(self) -> FresnelZonePlateProbeBuilder:
        builder = FresnelZonePlateProbeBuilder(self._settings, self._fresnel_zone_plate_chooser)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def labels_for_presets(self) -> Iterator[str]:
        for plugin in self._fresnel_zone_plate_chooser:
            yield plugin.display_name

    def apply_presets(self, display_name: str) -> None:
        self._fresnel_zone_plate_chooser.set_current_plugin(display_name)
        fzp = self._fresnel_zone_plate_chooser.get_current_plugin().strategy
        self.zone_plate_diameter_m.set_value(fzp.zone_plate_diameter_m)
        self.outermost_zone_width_m.set_value(fzp.outermost_zone_width_m)
        self.central_beamstop_diameter_m.set_value(fzp.central_beamstop_diameter_m)

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        wavelength_m = geometry_provider.probe_wavelength_m
        zone_plate = FresnelZonePlate(
            zone_plate_diameter_m=self.zone_plate_diameter_m.get_value(),
            outermost_zone_width_m=self.outermost_zone_width_m.get_value(),
            central_beamstop_diameter_m=self.central_beamstop_diameter_m.get_value(),
        )
        focal_length_m = zone_plate.get_focal_length_m(wavelength_m)
        distance_m = focal_length_m + self.defocus_distance_m.get_value()
        sample_plane_geometry = geometry_provider.get_probe_geometry()
        fzp_half_width = (sample_plane_geometry.width_px + 1) // 2
        fzp_half_height = (sample_plane_geometry.height_px + 1) // 2
        fzp_plane_pixel_size_numerator = wavelength_m * distance_m
        fzp_pixel_geometry = PixelGeometry(
            width_m=fzp_plane_pixel_size_numerator / sample_plane_geometry.width_m,
            height_m=fzp_plane_pixel_size_numerator / sample_plane_geometry.height_m,
        )

        # coordinate on FZP plane
        lx_fzp = -fzp_pixel_geometry.width_m * numpy.arange(-fzp_half_width, fzp_half_width)
        ly_fzp = -fzp_pixel_geometry.height_m * numpy.arange(-fzp_half_height, fzp_half_height)

        YY_FZP, XX_FZP = numpy.meshgrid(ly_fzp, lx_fzp)  # noqa: N806
        RR_FZP = numpy.hypot(XX_FZP, YY_FZP)  # noqa: N806

        # transmission function of FZP
        T = numpy.exp(  # noqa: N806
            -2j * numpy.pi / wavelength_m * (XX_FZP**2 + YY_FZP**2) / 2 / focal_length_m
        )
        C = RR_FZP <= zone_plate.zone_plate_diameter_m / 2  # noqa: N806
        H = RR_FZP >= zone_plate.central_beamstop_diameter_m / 2  # noqa: N806
        fzp_transmission_function = T * C * H

        propagator_parameters = PropagatorParameters(
            wavelength_m=wavelength_m,
            width_px=fzp_transmission_function.shape[-1],
            height_px=fzp_transmission_function.shape[-2],
            pixel_width_m=fzp_pixel_geometry.width_m,
            pixel_height_m=fzp_pixel_geometry.height_m,
            propagation_distance_m=distance_m,
        )
        propagator = FresnelTransformPropagator(propagator_parameters)
        array = propagator.propagate(fzp_transmission_function)

        return ProbeSequence(
            array=self.normalize(array),
            opr_weights=None,
            pixel_geometry=sample_plane_geometry.get_pixel_geometry(),
        )
