from __future__ import annotations
import logging


from ptychodus.api.object import Object, ObjectGeometryProvider
from ptychodus.api.simulate.object import generate_stxm_object

from ...diffraction import AssembledDiffractionDataset
from .builder import ObjectBuilder
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class STXMObjectBuilder(ObjectBuilder):
    def __init__(
        self,
        settings: ObjectSettings,
        dataset: AssembledDiffractionDataset,
    ) -> None:
        super().__init__(settings, 'stxm')
        self._settings = settings
        self._dataset = dataset

    def copy(self) -> STXMObjectBuilder:
        builder = STXMObjectBuilder(self._settings, self._dataset)

        for key, value in self.parameters().items():
            builder.parameters()[key].set_value(value.get_value())

        return builder

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        object_ = generate_stxm_object(
            geometry_provider.get_object_geometry(),
            self._dataset.get_assembled_data(),
            geometry_provider.get_probe_positions(),
        )
        return self._pad_object(object_)
