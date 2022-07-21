from abc import abstractmethod
from collections.abc import Callable
import logging

import numpy
import numpy.typing

from ...api.image import RealArrayType, ScalarTransformation
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser, PluginEntry

InexactArrayType = numpy.typing.NDArray[numpy.inexact]
NumericArrayType = numpy.typing.NDArray[numpy.number]

logger = logging.getLogger(__name__)


class VisualizationArray(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._array: InexactArrayType = numpy.empty((0, 0))

    def __call__(self) -> InexactArrayType:
        return self._array

    def setArray(self, array: NumericArrayType) -> None:
        if numpy.issubdtype(array.dtype, numpy.inexact):
            self._array = array
        elif numpy.issubdtype(array.dtype, numpy.integer):
            self._array = array.astype(numpy.float64)
        else:
            logger.error(f'Refusing to assign array with non-numeric dtype \"{array.dtype}\"!')
            self._array = numpy.empty((0, 0))

        self.notifyObservers()


class VisualizationArrayComponent(Observable, Observer):

    def __init__(self, name: str, isCyclic: bool, array: VisualizationArray,
                 transformChooser: PluginChooser[ScalarTransformation]) -> None:
        super().__init__()
        self._name = name
        self._isCyclic = isCyclic
        self._array = array
        self._transformChooser = transformChooser

        array.addObserver(self)
        transformChooser.addObserver(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def isCyclic(self) -> bool:
        return self._isCyclic

    def getScalarTransformationList(self) -> list[str]:
        return self._transformChooser.getDisplayNameList()

    def getScalarTransformation(self) -> str:
        return self._transformChooser.getCurrentDisplayName()

    def setScalarTransformation(self, name: str) -> None:
        self._transformChooser.setFromDisplayName(name)

    def getArray(self) -> InexactArrayType:
        return self._array()

    @abstractmethod
    def getComponent(self) -> RealArrayType:
        pass

    def __call__(self) -> RealArrayType:
        transform = self._transformChooser.getCurrentStrategy()
        return transform(self.getComponent())

    def update(self, observable: Observable) -> None:
        if observable is self._array:
            self.notifyObservers()
        elif observable is self._transformChooser:
            self.notifyObservers()


class AmplitudeArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray,
                 transformChooser: PluginChooser[ScalarTransformation]):
        super().__init__('Amplitude', False, array, transformChooser)

    def getComponent(self) -> RealArrayType:
        return numpy.absolute(self.getArray())


class PhaseArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray,
                 transformChooser: PluginChooser[ScalarTransformation]):
        super().__init__('Phase', True, array, transformChooser)

    def getComponent(self) -> RealArrayType:
        return numpy.angle(self.getArray())


class RealArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray,
                 transformChooser: PluginChooser[ScalarTransformation]):
        super().__init__('Real', False, array, transformChooser)

    def getComponent(self) -> RealArrayType:
        return numpy.real(self.getArray())


class ImaginaryArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray,
                 transformChooser: PluginChooser[ScalarTransformation]):
        super().__init__('Imaginary', False, array, transformChooser)

    def getComponent(self) -> RealArrayType:
        return numpy.imag(self.getArray())
