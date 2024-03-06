import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from .builder import FromMemoryObjectBuilder
from .item import ObjectRepositoryItem
from .random import RandomObjectBuilder


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def createDefault(self, geometryProvider: ObjectGeometryProvider) -> ObjectRepositoryItem:
        builder = RandomObjectBuilder(self._rng, geometryProvider)
        return ObjectRepositoryItem(builder)

    def create(self, object_: Object) -> ObjectRepositoryItem:
        builder = FromMemoryObjectBuilder(object_)
        return ObjectRepositoryItem(builder)
