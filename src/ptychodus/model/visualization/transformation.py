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
        self._chooser.register_plugin(
            IdentityScalarTransformation(),
            display_name='Identity',
        )
        self._chooser.register_plugin(
            SquareRootScalarTransformation(),
            simple_name='sqrt',
            display_name='Square Root',
        )
        self._chooser.register_plugin(
            Log2ScalarTransformation(),
            simple_name='log2',
            display_name='Logarithm (Base 2)',
        )

        self._chooser.register_plugin(
            LogScalarTransformation(),
            simple_name='ln',
            display_name='Natural Logarithm',
        )
        self._chooser.register_plugin(
            Log10ScalarTransformation(),
            simple_name='log10',
            display_name='Logarithm (Base 10)',
        )
        self.setValue('Identity')
        self._chooser.addObserver(self)

    def choices(self) -> Iterator[str]:
        for plugin in self._chooser:
            yield plugin.display_name

    def getValue(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.set_current_plugin(value)

    def getValueAsString(self) -> str:
        return self.getValue()

    def setValueFromString(self, value: str) -> None:
        self.setValue(value)

    def copy(self) -> Parameter[str]:
        parameter = ScalarTransformationParameter()
        parameter.setValue(self.getValue())
        return parameter

    def decorateText(self, text: str) -> str:
        return self._chooser.get_current_plugin().strategy.decorateText(text)

    def transform(self, values: RealArrayType) -> RealArrayType:
        return self._chooser.get_current_plugin().strategy(values)

    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notifyObservers()
