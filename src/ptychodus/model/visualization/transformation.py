from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import Parameter
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.typing import RealArrayType

__all__ = [
    'ScalarTransformation',
    'ScalarTransformationParameter',
]


class ScalarTransformation(ABC):
    @abstractmethod
    def decorate_text(self, text: str) -> str:
        pass

    @abstractmethod
    def __call__(self, array: RealArrayType) -> RealArrayType:
        pass


class IdentityScalarTransformation(ScalarTransformation):
    def decorate_text(self, text: str) -> str:
        return text

    def __call__(self, array: RealArrayType) -> RealArrayType:
        return array


class SquareRootScalarTransformation(ScalarTransformation):
    def decorate_text(self, text: str) -> str:
        return f'$\\sqrt{{\\mathrm{{{text}}}}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.sqrt(array, out=nil, where=(array > 0))


class Log2ScalarTransformation(ScalarTransformation):
    def decorate_text(self, text: str) -> str:
        return f'$\\log_2{{\\left(\\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log2(array, out=nil, where=(array > 0))


class LogScalarTransformation(ScalarTransformation):
    def decorate_text(self, text: str) -> str:
        return f'$\\ln{{\\left(\\mathrm{{{text}}}\\right)}}$'

    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log(array, out=nil, where=(array > 0))


class Log10ScalarTransformation(ScalarTransformation):
    def decorate_text(self, text: str) -> str:
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
        self.set_value('Identity')
        self._chooser.add_observer(self)

    def choices(self) -> Iterator[str]:
        for plugin in self._chooser:
            yield plugin.display_name

    def get_value(self) -> str:
        return self._chooser.get_current_plugin().display_name

    def set_value(self, value: str, *, notify: bool = True) -> None:
        self._chooser.set_current_plugin(value)

    def get_value_as_string(self) -> str:
        return self.get_value()

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value)

    def copy(self) -> Parameter[str]:
        parameter = ScalarTransformationParameter()
        parameter.set_value(self.get_value())
        return parameter

    def decorate_text(self, text: str) -> str:
        return self._chooser.get_current_plugin().strategy.decorate_text(text)

    def transform(self, values: RealArrayType) -> RealArrayType:
        return self._chooser.get_current_plugin().strategy(values)

    def _update(self, observable: Observable) -> None:
        if observable is self._chooser:
            self.notify_observers()
