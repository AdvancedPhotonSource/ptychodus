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
        builder_factory: ObjectBuilderFactory,
    ) -> None:
        self._rng = rng
        self._settings = settings
        self._builder_factory = builder_factory

    def create(
        self, geometry_provider: ObjectGeometryProvider, object_: Object | None = None
    ) -> ObjectRepositoryItem:
        # TODO layers_builder = MultilayerObjectBuilder()

        if object_ is None:
            builder = self._builder_factory.create_default()
        else:
            builder = FromMemoryObjectBuilder(self._settings, object_)
            # TODO layers_builder.set_identity()

        return ObjectRepositoryItem(geometry_provider, self._settings, builder)

    def create_from_settings(
        self, geometry_provider: ObjectGeometryProvider
    ) -> ObjectRepositoryItem:
        try:
            builder = self._builder_factory.create_from_settings()
        except Exception as exc:
            logger.error(''.join(exc.args))
            builder = self._builder_factory.create_default()

        return ObjectRepositoryItem(geometry_provider, self._settings, builder)
