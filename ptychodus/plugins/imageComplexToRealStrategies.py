import numpy

from ptychodus.api.image import ComplexToRealStrategy, RealArrayType, ComplexArrayType
from ptychodus.api.plugins import PluginRegistry


class ComplexMagnitudeStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Magnitude'

    @property
    def isCyclic(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType) -> RealArrayType:
        return numpy.absolute(array)


class ComplexPhaseStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Phase'

    @property
    def isCyclic(self) -> bool:
        return True

    def __call__(self, array: ComplexArrayType) -> RealArrayType:
        return numpy.angle(array)


class ComplexRealComponentStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Real'

    @property
    def isCyclic(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType) -> RealArrayType:
        return numpy.real(array)


class ComplexImaginaryComponentStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Imaginary'

    @property
    def isCyclic(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType) -> RealArrayType:
        return numpy.imag(array)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(ComplexMagnitudeStrategy())
    registry.registerPlugin(ComplexPhaseStrategy())
    registry.registerPlugin(ComplexRealComponentStrategy())
    registry.registerPlugin(ComplexImaginaryComponentStrategy())
