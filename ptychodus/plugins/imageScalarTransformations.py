import numpy

from ptychodus.api.image import ScalarTransformation, RealArrayType
from ptychodus.api.plugins import PluginRegistry


class IdentityScalarTransformation(ScalarTransformation):

    def decorateText(self, text: str) -> str:
        return text

    def __call__(self, array: RealArrayType) -> RealArrayType:
        return array


class SquareRootScalarTransformation(ScalarTransformation):

    def decorateText(self, text: str) -> str:
        return f'$\\sqrt{{\mathrm{{{text}}}}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.sqrt(array, out=nil, where=(array > 0))


class Log2ScalarTransformation(ScalarTransformation):

    def decorateText(self, text: str) -> str:
        return f'$\\log_2{{\\left(\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log2(array, out=nil, where=(array > 0))


class LogScalarTransformation(ScalarTransformation):

    def decorateText(self, text: str) -> str:
        return f'$\\ln{{\\left(\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log(array, out=nil, where=(array > 0))


class Log10ScalarTransformation(ScalarTransformation):

    def decorateText(self, text: str) -> str:
        return f'$\\log_{{10}}{{\\left(\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log10(array, out=nil, where=(array > 0))


def registerPlugins(registry: PluginRegistry) -> None:
    registry.scalarTransformations.registerPlugin(
        IdentityScalarTransformation(),
        simpleName='Identity',
    )
    registry.scalarTransformations.registerPlugin(
        SquareRootScalarTransformation(),
        simpleName='sqrt',
        displayName='Square Root',
    )
    registry.scalarTransformations.registerPlugin(
        Log2ScalarTransformation(),
        simpleName='log2',
        displayName='Logarithm (Base 2)',
    )

    registry.scalarTransformations.registerPlugin(
        LogScalarTransformation(),
        simpleName='ln',
        displayName='Natural Logarithm',
    )
    registry.scalarTransformations.registerPlugin(
        Log10ScalarTransformation(),
        simpleName='log10',
        displayName='Logarithm (Base 10)',
    )
