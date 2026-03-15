from __future__ import annotations
from collections.abc import Sequence
import logging


from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.object_gen import generate_stxm_object

from ...diffraction import DiffractionAPI
from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class STXMObjectBuilder(ObjectBuilder):
    def __init__(
        self,
        settings: ObjectSettings,
        diffraction_api: DiffractionAPI,
    ) -> None:
        super().__init__(settings, 'stxm')
        self._settings = settings
        self._diffraction_api = diffraction_api

    def copy(self) -> STXMObjectBuilder:
        builder = STXMObjectBuilder(self._settings, self._diffraction_api)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        object_ = generate_stxm_object(
            geometry_provider.get_object_geometry(),
            self._diffraction_api.get_assembled_data(),
            geometry_provider.get_probe_positions(),
        )
        return self._create_object(object_, layer_spacing_m)
