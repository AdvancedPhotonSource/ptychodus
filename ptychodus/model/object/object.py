import numpy

from ...api.object import ObjectArrayType
from ...api.observer import Observable
from .sizer import ObjectSizer


class Object(Observable):

    def __init__(self, sizer: ObjectSizer) -> None:
        super().__init__()
        self._array = numpy.zeros(sizer.getObjectExtent().shape, dtype=complex)

    def getArray(self) -> ObjectArrayType:
        return self._array

    def setArray(self, array: ObjectArrayType) -> None:
        if not numpy.iscomplexobj(array):
            raise TypeError('Object must be a complex-valued ndarray')

        self._array = array
        self.notifyObservers()
