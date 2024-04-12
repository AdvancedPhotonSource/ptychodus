import numpy

from ptychodus.api.object import Object, ObjectGeometryProvider

from ...patterns import ProductSettings
from .builder import FromMemoryObjectBuilder
from .item import ObjectRepositoryItem
from .random import RandomObjectBuilder


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ProductSettings) -> None:
        self._rng = rng
        self._settings = settings

    def create(self,
               geometryProvider: ObjectGeometryProvider,
               object_: Object | None = None) -> ObjectRepositoryItem:
        builder = RandomObjectBuilder(self._rng) if object_ is None \
                else FromMemoryObjectBuilder(object_)
        return ObjectRepositoryItem(geometryProvider, self._settings, builder)
