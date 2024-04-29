import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import FromMemoryObjectBuilder
from .builderFactory import ObjectBuilderFactory
from .item import ObjectRepositoryItem
from .settings import ObjectSettings


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ObjectSettings,
                 builderFactory: ObjectBuilderFactory) -> None:
        self._rng = rng
        self._settings = settings
        self._builderFactory = builderFactory

    def create(self,
               geometryProvider: ObjectGeometryProvider,
               object_: Object | None = None) -> ObjectRepositoryItem:
        builder = self._builderFactory.createDefault() if object_ is None \
                else FromMemoryObjectBuilder(object_)
        return ObjectRepositoryItem(geometryProvider, self._settings, builder)
