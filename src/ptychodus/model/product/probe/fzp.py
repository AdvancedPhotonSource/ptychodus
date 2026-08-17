from __future__ import annotations
from collections.abc import Iterator

import numpy

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    FresnelZonePlate,
    generate_fresnel_zone_plate_probe,
)

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class FresnelZonePlateProbeBuilder(ProbeSequenceBuilder):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
    ) -> None:
        super().__init__(rng, settings, 'fresnel_zone_plate')
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
        builder = FresnelZonePlateProbeBuilder(
            self._rng, self._settings, self._fresnel_zone_plate_chooser
        )

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

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        zone_plate = FresnelZonePlate(
            zone_plate_diameter_m=self.zone_plate_diameter_m.get_value(),
            outermost_zone_width_m=self.outermost_zone_width_m.get_value(),
            central_beamstop_diameter_m=self.central_beamstop_diameter_m.get_value(),
        )
        return self._rescale_to_photon_count(
            generate_fresnel_zone_plate_probe(
                geometry=geometry_provider.get_probe_geometry(),
                zone_plate=zone_plate,
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                defocus_distance_m=self.defocus_distance_m.get_value(),
            ),
            geometry_provider,
        )
