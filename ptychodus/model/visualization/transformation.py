from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import override

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

    @override
    def decorateText(self, text: str) -> str:
        return text

    @override
    def __call__(self, array: RealArrayType) -> RealArrayType:
        return array


class SquareRootScalarTransformation(ScalarTransformation):

    @override
    def decorateText(self, text: str) -> str:
        return f'$\\sqrt{{\\mathrm{{{text}}}}}$'

    @override
    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.sqrt(array, out=nil, where=(array > 0))


class Log2ScalarTransformation(ScalarTransformation):

    @override
    def decorateText(self, text: str) -> str:
        return f'$\\log_2{{\\left(\\mathrm{{{text}}}\\right)}}$'

    @override
    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log2(array, out=nil, where=(array > 0))


class LogScalarTransformation(ScalarTransformation):

    @override
    def decorateText(self, text: str) -> str:
        return f'$\\ln{{\\left(\\mathrm{{{text}}}\\right)}}$'

    @override
    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log(array, out=nil, where=(array > 0))


class Log10ScalarTransformation(ScalarTransformation):

    @override
    def decorateText(self, text: str) -> str:
        return f'$\\log_{{10}}{{\\left(\\mathrm{{{text}}}\\right)}}$'

    @override
    def __call__(self, array: RealArrayType) -> RealArrayType:
        nil = numpy.zeros_like(array)
        return numpy.log10(array, out=nil, where=(array > 0))


class ScalarTransformationParameter(Parameter[str], Observer):

    def __init__(self) -> None:
        super().__init__('')
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

    @override
    def setValue(self, value: str, *, notify: bool = True) -> None:
        self._chooser.setCurrentPluginByName(value)
        super().setValue(self._chooser.currentPlugin.displayName, notify=notify)

    def getPlugin(self) -> ScalarTransformation:
        return self._chooser.currentPlugin.strategy

    @override
    def update(self, observable: Observable) -> None:
        if observable is self._chooser:
            super().setValue(self._chooser.currentPlugin.displayName)
