from __future__ import annotations
from collections.abc import Sequence
import logging


from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.object_gen import generate_paganin_object

from ...diffraction import DiffractionAPI
from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class PaganinObjectBuilder(ObjectBuilder):
    def __init__(
        self,
        settings: ObjectSettings,
        diffraction_api: DiffractionAPI,
    ) -> None:
        super().__init__(settings, 'paganin')
        self._settings = settings
        self._diffraction_api = diffraction_api

        self.probe_wavelength_m = settings.paganin_probe_wavelength_m.copy()
        self._add_parameter('probe_wavelength_m', self.probe_wavelength_m)
        self.propagation_distance_m = settings.paganin_propagation_distance_m.copy()
        self._add_parameter('propagation_distance_m', self.propagation_distance_m)
        self.delta_over_beta = settings.paganin_delta_over_beta.copy()
        self._add_parameter('delta_over_beta', self.delta_over_beta)

    def copy(self) -> PaganinObjectBuilder:
        builder = PaganinObjectBuilder(self._settings, self._diffraction_api)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        object_ = generate_paganin_object(
            geometry_provider.get_object_geometry(),
            self._diffraction_api.get_assembled_data(),
            geometry_provider.get_probe_positions(),
            probe_wavelength_m=self.probe_wavelength_m.get_value(),
            propagation_distance_m=self.propagation_distance_m.get_value(),
            delta_over_beta=self.delta_over_beta.get_value(),
        )
        return self._create_object(object_, layer_spacing_m)
