from __future__ import annotations
from collections.abc import Iterator, Mapping
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

    def setValue(self, value: T, *, notify=True) -> None:
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

    def getValue(self) -> float:
        value = super().getValue()

        if self._minimum is not None:
            value = max(self._minimum, value)

        if self._maximum is not None:
            value = min(value, self._maximum)

        return value


class ParameterRepository(Mapping[str, Any], Observable, Observer):

    def __init__(self, name: str, parent: ParameterRepository | None = None) -> None:
        self._parameterList: dict[str, Parameter[Any]] = dict()
        self._repositoryList: list[ParameterRepository] = list()

        if parent is not None:
            parent._addParameterRepository(self)

    def _addParameterRepository(self, repository: ParameterRepository) -> None:
        if repository not in self._repositoryList:
            self._repositoryList.append(repository)

    def _removeParameterRepository(self, repository: ParameterRepository) -> None:
        try:
            self._repositoryList.remove(repository)
        except ValueError:
            pass

    def _registerParameter(self, name: str, parameter: Parameter[Any]) -> None:
        if self._parameterList.setdefault(name, parameter) == parameter:
            parameter.addObserver(self)
        else:
            raise ValueError('Name already exists!')

    def _registerStringParameter(self, name: str, value: str) -> Parameter[str]:
        parameter = StringParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def _registerPathParameter(self, name: str, value: Path) -> Parameter[Path]:
        parameter = PathParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def _registerBooleanParameter(self, name: str, value: bool) -> Parameter[bool]:
        parameter = BooleanParameter(value)
        self._registerParameter(name, parameter)
        return parameter

    def _registerIntegerParameter(self,
                                  name: str,
                                  value: int,
                                  *,
                                  minimum: int | None = None,
                                  maximum: int | None = None) -> Parameter[int]:
        parameter = IntegerParameter(value, minimum=minimum, maximum=maximum)
        self._registerParameter(name, parameter)
        return parameter

    def _registerRealParameter(self,
                               name: str,
                               value: float,
                               *,
                               minimum: float | None = None,
                               maximum: float | None = None) -> Parameter[float]:
        parameter = RealParameter(value, minimum=minimum, maximum=maximum)
        self._registerParameter(name, parameter)
        return parameter

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameterList)

    def __getitem__(self, name: str) -> Any:
        return self._parameterList[name]

    def __len__(self) -> int:
        return len(self._parameterList)

    def setParameters(self, parameterList: Mapping[str, Any]) -> None:
        for key, value in parameterList.items():
            try:
                parameter = self._parameterList[key]
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
        if observable in self._parameterList.values():
            self.notifyObservers()
