from abc import ABC, abstractmethod
import logging

from skimage.restoration import unwrap_phase
import numpy
import numpy.typing

from ...api.observer import Observable, Observer

InexactArrayType = numpy.typing.NDArray[numpy.inexact]
NumericArrayType = numpy.typing.NDArray[numpy.number]

logger = logging.getLogger(__name__)


class VisualizationArray(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._array: InexactArrayType = numpy.zeros((1, 1))

    def _reset(self) -> None:
        self._array = numpy.zeros((1, 1))

    def __call__(self) -> InexactArrayType:
        return self._array

    def setArray(self, array: NumericArrayType) -> None:
        if array is None:
            logger.error('Refusing to assign null array!')
            self._reset()
        elif numpy.size(array) < 1:
            logger.error('Refusing to assign empty array!')
            self._reset()
        elif numpy.issubdtype(array.dtype, numpy.inexact):
            self._array = array
        elif numpy.issubdtype(array.dtype, numpy.integer):
            self._array = array.astype(numpy.float64)
        else:
            logger.error(f'Refusing to assign array with non-numeric dtype \"{array.dtype}\"!')
            self._reset()

        self.notifyObservers()


class VisualizationArrayComponent(Observable, Observer, ABC):

    def __init__(self, name: str, array: VisualizationArray) -> None:
        super().__init__()
        self._name = name
        self._array = array
        self._array.addObserver(self)

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def __call__(self) -> InexactArrayType:
        pass

    def update(self, observable: Observable) -> None:
        if observable is self._array:
            self.notifyObservers()


class ComplexArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray) -> None:
        super().__init__('Complex', array)

    def __call__(self) -> InexactArrayType:
        return self._array()


class AmplitudeArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray) -> None:
        super().__init__('Amplitude', array)

    def __call__(self) -> InexactArrayType:
        return numpy.absolute(self._array())


class PhaseArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray) -> None:
        super().__init__('Phase', array)

    def __call__(self) -> InexactArrayType:
        return numpy.angle(self._array())


class UnwrappedPhaseArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray) -> None:
        super().__init__('Phase (Unwrapped)', array)

    def __call__(self) -> InexactArrayType:
        phase = numpy.angle(self._array())
        return unwrap_phase(phase)


class RealArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray) -> None:
        super().__init__('Real', array)

    def __call__(self) -> InexactArrayType:
        return numpy.real(self._array())


class ImaginaryArrayComponent(VisualizationArrayComponent):

    def __init__(self, array: VisualizationArray) -> None:
        super().__init__('Imaginary', array)

    def __call__(self) -> InexactArrayType:
        return numpy.imag(self._array())
