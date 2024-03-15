from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, TypeVar
import logging

from .observer import Observable, Observer

__all__ = [
    'ParameterRepository',
]

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Parameter(Generic[T], Observable):

    def __init__(self, value: T) -> None:
        super().__init__()
        self._value = value

    def getValue(self) -> T:
        return self._value

    def setValue(self, value: T, *, notify: bool = True) -> None:
        if self._value != value:
            self._value = value

            if notify:
                self.notifyObservers()


class StringParameter(Parameter[str]):

    def __init__(self, value: str) -> None:
        super().__init__(value)


class PathParameter(Parameter[Path]):

    def __init__(self, value: Path) -> None:
        super().__init__(value)


class BooleanParameter(Parameter[bool]):

    def __init__(self, value: bool) -> None:
        super().__init__(value)


class IntegerParameter(Parameter[int]):

    def __init__(self, value: int, *, minimum: int | None, maximum: int | None) -> None:
        super().__init__(value)
        self._minimum = minimum
        self._maximum = maximum

    def getMinimum(self) -> int | None:
        return self._minimum

    def getMaximum(self) -> int | None:
        return self._maximum

    def getValue(self) -> int:
        value = super().getValue()

        if self._minimum is not None:
            value = max(self._minimum, value)

        if self._maximum is not None:
            value = min(value, self._maximum)

        return value


class RealParameter(Parameter[float]):

    def __init__(self, value: float, *, minimum: float | None, maximum: float | None) -> None:
        super().__init__(value)
        self._minimum = minimum
        self._maximum = maximum

    def getMinimum(self) -> float | None:
        return self._minimum

    def getMaximum(self) -> float | None:
        return self._maximum

    def getValue(self) -> float:
        value = super().getValue()

        if self._minimum is not None:
            value = max(self._minimum, value)

        if self._maximum is not None:
            value = min(value, self._maximum)

        return value


class RealArrayParameter(Parameter[Sequence[float]]):

    def __init__(self, value: Sequence[float]) -> None:
        super().__init__(list(value))

    def __iter__(self) -> Iterator[float]:
        return iter(self._value)

    def __getitem__(self, index: int) -> float:
        return self._value[index]

    def __len__(self) -> int:
        return len(self._value)

    def setValue(self, value: Sequence[float], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notifyObservers()


class ComplexArrayParameter(Parameter[Sequence[complex]]):

    def __init__(self, value: Sequence[complex]) -> None:
        super().__init__(list(value))

    def __iter__(self) -> Iterator[complex]:
        return iter(self._value)

    def __getitem__(self, index: int) -> complex:
        return self._value[index]

    def __len__(self) -> int:
        return len(self._value)

    def setValue(self, value: Sequence[complex], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notifyObservers()


class ParameterRepository(Mapping[str, Any], Observable, Observer):

    def __init__(self, name: str, parent: ParameterRepository | None = None) -> None:
        super().__init__()
        self._parameterDict: dict[str, Parameter[Any]] = dict()
        self._repositoryList: list[ParameterRepository] = list()

        if parent is not None:
            parent._addParameterRepository(self)

    def _addParameterRepository(self,
                                repository: ParameterRepository,
                                *,
                                observe: bool = False) -> None:
        if repository not in self._repositoryList:
            self._repositoryList.append(repository)

        if observe:
            repository.addObserver(self)

    def _removeParameterRepository(self, repository: ParameterRepository) -> None:
        try:
            self._repositoryList.remove(repository)
        except ValueError:
            pass
        else:
            repository.removeObserver(self)

    def _registerParameter(self, name: str, parameter: Parameter[Any]) -> None:
        if self._parameterDict.setdefault(name, parameter) == parameter:
            parameter.addObserver(self)
        else:
            raise ValueError('Name already exists!')

    def _registerStringParameter(self, name: str, value: str) -> StringParameter:
        parameter = StringParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def _registerPathParameter(self, name: str, value: Path) -> Parameter[Path]:
        parameter = PathParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def _registerBooleanParameter(self, name: str, value: bool) -> BooleanParameter:
        parameter = BooleanParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def _registerIntegerParameter(self,
                                  name: str,
                                  value: int,
                                  *,
                                  minimum: int | None = None,
                                  maximum: int | None = None) -> IntegerParameter:
        parameter = IntegerParameter(value, minimum=minimum, maximum=maximum)
        self._registerParameter(name, parameter)
        return parameter

    def _registerRealParameter(self,
                               name: str,
                               value: float,
                               *,
                               minimum: float | None = None,
                               maximum: float | None = None) -> RealParameter:
        parameter = RealParameter(value, minimum=minimum, maximum=maximum)
        self._registerParameter(name, parameter)
        return parameter

    def _registerRealArrayParameter(self, name: str, value: Sequence[float]) -> RealArrayParameter:
        parameter = RealArrayParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def _registerComplexArrayParameter(self, name: str,
                                       value: Sequence[complex]) -> ComplexArrayParameter:
        parameter = ComplexArrayParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameterDict)

    def __getitem__(self, name: str) -> Any:
        return self._parameterDict[name]

    def __len__(self) -> int:
        return len(self._parameterDict)

    def setParameters(self, parameterMap: Mapping[str, Any]) -> None:
        for key, value in parameterMap.items():
            try:
                parameter = self._parameterDict[key]
            except KeyError:
                logger.debug(f'Parameter \"{key}\" not found!')
            else:
                valueOld = parameter.getValue()
                logger.debug(f'Parameter \"{key}\": {valueOld} -> {value}')

                if type(value) is type(valueOld):
                    parameter.setValue(value, notify=False)
                else:
                    raise ValueError(
                        f'Parameter \"{key}\" type mismatch! {type(valueOld)} != {type(value)}')

        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable in self._parameterDict.values():
            self.notifyObservers()
