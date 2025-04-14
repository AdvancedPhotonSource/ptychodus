from abc import ABC, abstractmethod

from skimage.restoration import unwrap_phase
import numpy

from ptychodus.api.typing import NumberArrayType, RealArrayType


class DataArrayComponent(ABC):
    def __init__(self, name: str, *, is_cyclic: bool) -> None:
        self._name = name
        self._is_cyclic = is_cyclic

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_cyclic(self) -> bool:
        return self._is_cyclic

    @abstractmethod
    def calculate(self, array: NumberArrayType) -> RealArrayType:
        pass


class RealArrayComponent(DataArrayComponent):
    def __init__(self) -> None:
        super().__init__('real', is_cyclic=False)

    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.real(array).astype(numpy.single)


class ImaginaryArrayComponent(DataArrayComponent):
    def __init__(self) -> None:
        super().__init__('imaginary', is_cyclic=False)

    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.imag(array).astype(numpy.single)


class AmplitudeArrayComponent(DataArrayComponent):
    def __init__(self) -> None:
        super().__init__('amplitude', is_cyclic=False)

    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.absolute(array).astype(numpy.single)


class PhaseInRadiansArrayComponent(DataArrayComponent):
    def __init__(self) -> None:
        super().__init__('phase', is_cyclic=True)

    def calculate(self, array: NumberArrayType) -> RealArrayType:
        return numpy.angle(array).astype(numpy.single)  # type: ignore


class UnwrappedPhaseInRadiansArrayComponent(DataArrayComponent):
    def __init__(self) -> None:
        super().__init__('unwrapped_phase', is_cyclic=False)

    def calculate(self, array: NumberArrayType) -> RealArrayType:
        phase_rad = numpy.angle(array).astype(numpy.single)  # type: ignore
        return unwrap_phase(phase_rad)
