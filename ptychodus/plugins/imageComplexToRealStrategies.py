import numpy

from ptychodus.api.image import ComplexArrayType, ComplexToRealStrategy, \
        RealArrayType, ScalarTransformation
from ptychodus.api.plugins import PluginRegistry

# FIXME encode magnitude/phase in hsv/hsl channels


class ComplexMagnitudeStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Magnitude'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.absolute(array)
        return transformation(values)


class ComplexPhaseStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Phase'

    @property
    def isCyclic(self) -> bool:
        return True

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.angle(array)
        return transformation(values)


class ComplexRealComponentStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Real'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.real(array)
        return transformation(values)


class ComplexImaginaryComponentStrategy(ComplexToRealStrategy):
    @property
    def name(self) -> str:
        return 'Imaginary'

    @property
    def isCyclic(self) -> bool:
        return False

    @property
    def isColorized(self) -> bool:
        return False

    def __call__(self, array: ComplexArrayType,
                 transformation: ScalarTransformation) -> RealArrayType:
        values = numpy.imag(array)
        return transformation(values)


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(ComplexMagnitudeStrategy())
    registry.registerPlugin(ComplexPhaseStrategy())
    registry.registerPlugin(ComplexRealComponentStrategy())
    registry.registerPlugin(ComplexImaginaryComponentStrategy())
