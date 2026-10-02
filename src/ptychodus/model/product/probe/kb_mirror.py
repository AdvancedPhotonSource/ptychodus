from __future__ import annotations
from collections.abc import Iterator

import numpy

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import ProbeSequence, ProbeGeometryProvider
from ptychodus.api.simulate.probe import (
    KirkpatrickBaezMirror,
    KirkpatrickBaezMirrorPair,
    generate_kb_mirror_probe,
)

from .builder import ProbeSequenceBuilder
from .settings import ProbeSettings


class KBMirrorProbeBuilder(ProbeSequenceBuilder):
    """Build a probe from a Kirkpatrick-Baez mirror pair.

    The source distance each mirror dataclass accepts is left out: it feeds only a focal
    length the probe simulation never reads, so a control for it would do nothing.
    """

    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        kb_mirror_chooser: PluginChooser[KirkpatrickBaezMirrorPair],
    ) -> None:
        super().__init__(rng, settings, 'kb_mirror')
        self._settings = settings
        self._kb_mirror_chooser = kb_mirror_chooser

        self.horizontal_acceptance_length_m = settings.kb_horizontal_acceptance_length_m.copy()
        self._add_parameter('horizontal_acceptance_length_m', self.horizontal_acceptance_length_m)

        self.horizontal_grazing_angle_rad = settings.kb_horizontal_grazing_angle_rad.copy()
        self._add_parameter('horizontal_grazing_angle_rad', self.horizontal_grazing_angle_rad)

        self.horizontal_focus_distance_m = settings.kb_horizontal_focus_distance_m.copy()
        self._add_parameter('horizontal_focus_distance_m', self.horizontal_focus_distance_m)

        self.vertical_acceptance_length_m = settings.kb_vertical_acceptance_length_m.copy()
        self._add_parameter('vertical_acceptance_length_m', self.vertical_acceptance_length_m)

        self.vertical_grazing_angle_rad = settings.kb_vertical_grazing_angle_rad.copy()
        self._add_parameter('vertical_grazing_angle_rad', self.vertical_grazing_angle_rad)

        self.vertical_focus_distance_m = settings.kb_vertical_focus_distance_m.copy()
        self._add_parameter('vertical_focus_distance_m', self.vertical_focus_distance_m)

        # separation of the two axis foci along the beam; zero is an aligned pair
        self.astigmatism_m = settings.kb_astigmatism_m.copy()
        self._add_parameter('astigmatism_m', self.astigmatism_m)

        # from sample to the focal plane
        self.defocus_distance_m = settings.defocus_distance_m.copy()
        self._add_parameter('defocus_distance_m', self.defocus_distance_m)

    def copy(self) -> KBMirrorProbeBuilder:
        builder = KBMirrorProbeBuilder(self._rng, self._settings, self._kb_mirror_chooser)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def labels_for_presets(self) -> Iterator[str]:
        for plugin in self._kb_mirror_chooser:
            yield plugin.display_name

    def apply_presets(self, display_name: str) -> None:
        self._kb_mirror_chooser.set_current_plugin(display_name)
        mirrors = self._kb_mirror_chooser.get_current_plugin().strategy
        self.horizontal_acceptance_length_m.set_value(mirrors.horizontal.acceptance_length_m)
        self.horizontal_grazing_angle_rad.set_value(mirrors.horizontal.grazing_angle_rad)
        self.horizontal_focus_distance_m.set_value(mirrors.horizontal.focus_distance_m)
        self.vertical_acceptance_length_m.set_value(mirrors.vertical.acceptance_length_m)
        self.vertical_grazing_angle_rad.set_value(mirrors.vertical.grazing_angle_rad)
        self.vertical_focus_distance_m.set_value(mirrors.vertical.focus_distance_m)

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        mirrors = KirkpatrickBaezMirrorPair(
            horizontal=KirkpatrickBaezMirror(
                acceptance_length_m=self.horizontal_acceptance_length_m.get_value(),
                grazing_angle_rad=self.horizontal_grazing_angle_rad.get_value(),
                focus_distance_m=self.horizontal_focus_distance_m.get_value(),
            ),
            vertical=KirkpatrickBaezMirror(
                acceptance_length_m=self.vertical_acceptance_length_m.get_value(),
                grazing_angle_rad=self.vertical_grazing_angle_rad.get_value(),
                focus_distance_m=self.vertical_focus_distance_m.get_value(),
            ),
        )
        return self._rescale_to_photon_count(
            generate_kb_mirror_probe(
                geometry_provider.get_probe_geometry(),
                mirrors,
                probe_wavelength_m=geometry_provider.probe_wavelength_m,
                defocus_distance_m=self.defocus_distance_m.get_value(),
                astigmatism_m=self.astigmatism_m.get_value(),
            ),
            geometry_provider,
        )
