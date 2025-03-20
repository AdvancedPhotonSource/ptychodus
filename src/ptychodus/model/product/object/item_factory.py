import logging

import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import FromMemoryObjectBuilder
from .builder_factory import ObjectBuilderFactory
from .item import ObjectRepositoryItem
from .settings import ObjectSettings

logger = logging.getLogger(__name__)


class ObjectRepositoryItemFactory:
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ObjectSettings,
        builderFactory: ObjectBuilderFactory,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._builderFactory = builderFactory

    def create(
        self, geometryProvider: ObjectGeometryProvider, object_: Object | None = None
    ) -> ObjectRepositoryItem:
        builder = (
            self._builderFactory.createDefault()
            if object_ is None
            else FromMemoryObjectBuilder(self._settings, object_)
        )
        return ObjectRepositoryItem(geometryProvider, self._settings, builder)

    def createFromSettings(self, geometryProvider: ObjectGeometryProvider) -> ObjectRepositoryItem:
        try:
            builder = self._builderFactory.createFromSettings()
        except Exception as exc:
            logger.error(''.join(exc.args))
            builder = self._builderFactory.createDefault()

        return ObjectRepositoryItem(geometryProvider, self._settings, builder)
