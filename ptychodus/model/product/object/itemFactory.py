import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import FromMemoryObjectBuilder
from .item import ObjectRepositoryItem
from .random import RandomObjectBuilder


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def create(self,
               geometryProvider: ObjectGeometryProvider,
               object_: Object | None = None) -> ObjectRepositoryItem:
        builder = RandomObjectBuilder(self._rng) if object_ is None \
                else FromMemoryObjectBuilder(object_)
        return ObjectRepositoryItem(geometryProvider, builder)
