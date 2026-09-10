from __future__ import annotations

from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.simulate.object import generate_siemens_star_object

from .builder import ObjectBuilder
from .settings import ObjectSettings


class SiemensStarObjectBuilder(ObjectBuilder):
    def __init__(self, settings: ObjectSettings) -> None:
        super().__init__(settings, 'siemens_star')
        self._settings = settings

        self.num_spokes = settings.siemens_star_num_spokes.copy()
        self._add_parameter('num_spokes', self.num_spokes)

        self.outer_radius_fraction = settings.siemens_star_outer_radius_fraction.copy()
        self._add_parameter('outer_radius_fraction', self.outer_radius_fraction)

        self.spoke_amplitude = settings.siemens_star_spoke_amplitude.copy()
        self._add_parameter('spoke_amplitude', self.spoke_amplitude)

        self.background_amplitude = settings.siemens_star_background_amplitude.copy()
        self._add_parameter('background_amplitude', self.background_amplitude)

        self.spoke_phase_tr = settings.siemens_star_spoke_phase_tr.copy()
        self._add_parameter('spoke_phase_turns', self.spoke_phase_tr)

        self.background_phase_tr = settings.siemens_star_background_phase_tr.copy()
        self._add_parameter('background_phase_turns', self.background_phase_tr)

    def copy(self) -> SiemensStarObjectBuilder:
        builder = SiemensStarObjectBuilder(self._settings)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        object_ = generate_siemens_star_object(
            geometry_provider.get_object_geometry(),
            num_spokes=self.num_spokes.get_value(),
            outer_radius_fraction=self.outer_radius_fraction.get_value(),
            spoke_amplitude=self.spoke_amplitude.get_value(),
            background_amplitude=self.background_amplitude.get_value(),
            spoke_phase_tr=self.spoke_phase_tr.get_value(),
            background_phase_tr=self.background_phase_tr.get_value(),
        )
        return self._pad_object(object_)
