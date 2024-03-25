from abc import ABC, abstractmethod
from typing import override

from skimage.restoration import unwrap_phase
import numpy

from ptychodus.api.visualization import NumberArrayType, RealArrayType


class DataArrayComponent(ABC):

    def __init__(self, name: str, *, isCyclic: bool) -> None:
        self._name = name
        self._isCyclic = isCyclic

    @property
    def name(self) -> str:
        return self._name

    @property
    def isCyclic(self) -> bool:
        return self._isCyclic

    @abstractmethod
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        pass


class RealArrayComponent(DataArrayComponent):

    def __init__(self) -> None:
        super().__init__('Real', isCyclic=False)

    @override
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.real(array).astype(numpy.single)


class ImaginaryArrayComponent(DataArrayComponent):

    def __init__(self) -> None:
        super().__init__('Imaginary', isCyclic=False)

    @override
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.imag(array).astype(numpy.single)


class AmplitudeArrayComponent(DataArrayComponent):

    def __init__(self) -> None:
        super().__init__('Amplitude', isCyclic=False)

    @override
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.absolute(array).astype(numpy.single)


class IntensityArrayComponent(DataArrayComponent):

    def __init__(self) -> None:
        super().__init__('Intensity', isCyclic=False)

    @override
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        amplitude = numpy.absolute(array).astype(numpy.single)
        return numpy.square(amplitude)


class PhaseInRadiansArrayComponent(DataArrayComponent):

    def __init__(self) -> None:
        super().__init__('Phase', isCyclic=True)

    @override
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.angle(array).astype(numpy.single)


class UnwrappedPhaseInRadiansArrayComponent(DataArrayComponent):

    def __init__(self) -> None:
        super().__init__('Phase (Unwrapped)', isCyclic=False)

    @override
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        phaseInRadians = numpy.angle(array).astype(numpy.single)
        return unwrap_phase(phaseInRadians)
