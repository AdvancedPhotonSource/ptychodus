from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.visualization import RealArrayType

__all__ = [
    'ScalarTransformation',
    'ScalarTransformationParameter',
]


class ScalarTransformation(ABC):
    @abstractmethod
    def decorateText(self, text: str) -> str:
        pass

    @abstractmethod
    def __call__(self, array: RealArrayType) -> RealArrayType:
        pass


class IdentityScalarTransformation(ScalarTransformation):
    def decorateText(self, text: str) -> str:
        return text

    def __call__(self, array: RealArrayType) -> RealArrayType:
        return array


class SquareRootScalarTransformation(ScalarTransformation):
    def decorateText(self, text: str) -> str:
        return f'$\\sqrt{{\\mathrm{{{text}}}}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.sqrt(array, out=nil, where=(array > 0))


class Log2ScalarTransformation(ScalarTransformation):
    def decorateText(self, text: str) -> str:
        return f'$\\log_2{{\\left(\\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log2(array, out=nil, where=(array > 0))


class LogScalarTransformation(ScalarTransformation):
    def decorateText(self, text: str) -> str:
        return f'$\\ln{{\\left(\\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log(array, out=nil, where=(array > 0))


class Log10ScalarTransformation(ScalarTransformation):
    def decorateText(self, text: str) -> str:
        return f'$\\log_{{10}}{{\\left(\\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log10(array, out=nil, where=(array > 0))


class ScalarTransformationParameter(Parameter[str], Observer):
    def __init__(self) -> None:
        super().__init__()
        self._chooser = PluginChooser[ScalarTransformation]()
        self._chooser.registerPlugin(
            IdentityScalarTransformation(),
            displayName='Identity',
        )
        self._chooser.registerPlugin(
            SquareRootScalarTransformation(),
            simpleName='sqrt',
            displayName='Square Root',
        )
        self._chooser.registerPlugin(
            Log2ScalarTransformation(),
            simpleName='log2',
            displayName='Logarithm (Base 2)',
        )

        self._chooser.registerPlugin(
            LogScalarTransformation(),
            simpleName='ln',
            displayName='Natural Logarithm',
        )
        self._chooser.registerPlugin(
            Log10ScalarTransformation(),
            simpleName='log10',
            displayName='Logarithm (Base 10)',
        )
        self.setValue('Identity')
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for name in self._chooser.getDisplayNameList():
            yield name

    def getValue(self) -> str:
        return self._chooser.currentPlugin.displayName

    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.setCurrentPluginByName(value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(value)

    def copy(self) -> Parameter[str]:
        parameter = ScalarTransformationParameter()
        parameter.setValue(self.getValue())
        return parameter

    def decorateText(self, text: str) -> str:
        return self._chooser.currentPlugin.strategy.decorateText(text)

    def transform(self, values: RealArrayType) -> RealArrayType:
        return self._chooser.currentPlugin.strategy(values)

    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notifyObservers()
