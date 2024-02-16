import numpy

from ...api.object import Object
from ...api.parametric import Parameter
from .builder import FromMemoryObjectBuilder
from .item import ObjectRepositoryItem


class ObjectRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def create(self, name: Parameter[str], object_: Object) -> ObjectRepositoryItem:
        builder = FromMemoryObjectBuilder(object_)
        return ObjectRepositoryItem(name, builder)
