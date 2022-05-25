import numpy

from ptychodus.api.image import ScalarTransformation, RealArrayType
from ptychodus.api.plugins import PluginRegistry


class IdentityScalarTransformation(ScalarTransformation):

    @property
    def name(self) -> str:
        return 'Identity'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        return array


class SquareRootScalarTransformation(ScalarTransformation):

    @property
    def name(self) -> str:
        return 'Square Root'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.sqrt(array, out=nil, where=(array > 0))


class LogScalarTransformation(ScalarTransformation):

    @property
    def name(self) -> str:
        return 'Natural Logarithm'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log(array, out=nil, where=(array > 0))


class Log2ScalarTransformation(ScalarTransformation):

    @property
    def name(self) -> str:
        return 'Logarithm (Base 2)'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log2(array, out=nil, where=(array > 0))


class Log10ScalarTransformation(ScalarTransformation):

    @property
    def name(self) -> str:
        return 'Logarithm (Base 10)'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log10(array, out=nil, where=(array > 0))


def registerPlugins(registry: PluginRegistry) -> None:
    registry.registerPlugin(IdentityScalarTransformation())
    registry.registerPlugin(SquareRootScalarTransformation())
    registry.registerPlugin(LogScalarTransformation())
    registry.registerPlugin(Log2ScalarTransformation())
    registry.registerPlugin(Log10ScalarTransformation())
