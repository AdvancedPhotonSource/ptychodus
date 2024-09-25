from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableSequence, Sequence
from decimal import Decimal
from pathlib import Path
from typing import Any, Final, Generic, TypeVar
from uuid import UUID
import json
import logging

from .observer import Observable, Observer

__all__ = [
    "BooleanParameter",
    "ComplexArrayParameter",
    "DecimalParameter",
    "IntegerParameter",
    "Parameter",
    "ParameterGroup",
    "PathParameter",
    "RealArrayParameter",
    "RealParameter",
    "StringParameter",
    "UUIDParameter",
]

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Parameter(ABC, Generic[T], Observable):

    @abstractmethod
    def getValue(self) -> T:
        pass

    @abstractmethod
    def setValue(self, value: T, *, notify: bool = True) -> None:
        pass

    @abstractmethod
    def setValueFromString(self, value: str) -> None:
        pass


class ParameterGroup(Observable, Observer):

    def __init__(self) -> None:
        super().__init__()
        self._parameters: dict[str, Parameter[Any]] = dict()
        self._groups: dict[str, ParameterGroup] = dict()

    def parameters(self) -> Mapping[str, Parameter[Any]]:
        return self._parameters

    def _addParameter(self, name: str, parameter: Parameter[Any]) -> None:
        if self._parameters.setdefault(name, parameter) is parameter:
            parameter.addObserver(self)
        else:
            raise ValueError(f'Parameter "{name}" already exists!')

    def _removeParameter(self, name: str) -> None:
        try:
            parameter = self._parameters.pop(name)
        except KeyError:
            pass
        else:
            parameter.removeObserver(self)

    def groups(self) -> Mapping[str, ParameterGroup]:
        return self._groups

    def _addGroup(self, name: str, group: ParameterGroup, *, observe: bool = False) -> None:
        if self._groups.setdefault(name, group) is group:
            if observe:
                group.addObserver(self)
        else:
            raise ValueError(f'Group "{name}" already exists!')

    def _removeGroup(self, name: str) -> None:
        try:
            group = self._groups.pop(name)
        except KeyError:
            pass
        else:
            group.removeObserver(self)

    def createGroup(self, name: str) -> ParameterGroup:
        group = ParameterGroup()
        self._addGroup(name, group)
        return group

    def getGroup(self, name: str) -> ParameterGroup:
        return self._groups[name]

    def update(self, observable: Observable) -> None:
        if observable in self._parameters.values():
            self.notifyObservers()
        elif observable in self._groups.values():
            self.notifyObservers()


class ParameterBase(Parameter[T]):

    def __init__(self, parent: ParameterGroup, name: str, value: T) -> None:
        super().__init__()
        self._value = value
        parent._addParameter(name, self)

    def getValue(self) -> T:
        return self._value

    def setValue(self, value: T, *, notify: bool = True) -> None:
        if self._value != value:
            self._value = value

            if notify:
                self.notifyObservers()

    def __str__(self) -> str:
        return str(self._value)


class StringParameter(ParameterBase[str]):

    def __init__(self, parent: ParameterGroup, name: str, value: str) -> None:
        super().__init__(parent, name, value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(str(value))


class PathParameter(ParameterBase[Path]):

    def __init__(self, parent: ParameterGroup, name: str, value: Path) -> None:
        super().__init__(parent, name, value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(Path(value))

    def changePathPrefix(self, find_path_prefix: Path, replacement_path_prefix: Path) -> Path:
        value = self.getValue()

        try:
            relative_path = value.resolve().relative_to(find_path_prefix)
        except ValueError:
            pass
        else:
            return replacement_path_prefix / relative_path

        return value


class DecimalParameter(ParameterBase[Decimal]):

    def __init__(self, parent: ParameterGroup, name: str, value: Decimal | str) -> None:
        super().__init__(parent, name, Decimal(value) if isinstance(value, str) else value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(Decimal(value))


class UUIDParameter(ParameterBase[UUID]):

    def __init__(self, parent: ParameterGroup, name: str, value: UUID) -> None:
        super().__init__(parent, name, value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(UUID(value))


class BooleanParameter(ParameterBase[bool]):
    TRUE_VALUES: Final = ("1", "true", "t", "yes", "y")

    def __init__(self, parent: ParameterGroup, name: str, value: bool) -> None:
        super().__init__(parent, name, value)

    def setValueFromString(self, value: str) -> None:
        self.setValue(value.lower() in BooleanParameter.TRUE_VALUES)


class IntegerParameter(ParameterBase[int]):

    def __init__(
        self,
        parent: ParameterGroup,
        name: str,
        value: int,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> None:
        super().__init__(parent, name, value)
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

    def setValueFromString(self, value: str) -> None:
        self.setValue(int(value))


class RealParameter(ParameterBase[float]):

    def __init__(
        self,
        parent: ParameterGroup,
        name: str,
        value: float,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        super().__init__(parent, name, value)
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

    def setValueFromString(self, value: str) -> None:
        self.setValue(float(value))


class RealArrayParameter(ParameterBase[MutableSequence[float]]):

    def __init__(self, parent: ParameterGroup, name: str, value: Sequence[float]) -> None:
        super().__init__(parent, name, list(value))

    def __iter__(self) -> Iterator[float]:
        return iter(self._value)

    def __getitem__(self, index: int) -> float:
        return self._value[index]

    def __setitem__(self, index: int, value: float) -> None:
        if self._value[index] != value:
            self._value[index] = value
            self.notifyObservers()

    def __delitem__(self, index: int) -> None:
        del self._value[index]
        self.notifyObservers()

    def insert(self, index: int, value: float) -> None:
        self._value[index] = value
        self.notifyObservers()

    def __len__(self) -> int:
        return len(self._value)

    def setValue(self, value: Sequence[float], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notifyObservers()

    def setValueFromString(self, value: str) -> None:
        self.setValue(json.loads(value))  # FIXME


class ComplexArrayParameter(ParameterBase[MutableSequence[complex]]):

    def __init__(self, parent: ParameterGroup, name: str, value: Sequence[complex]) -> None:
        super().__init__(parent, name, list(value))

    def __iter__(self) -> Iterator[complex]:
        return iter(self._value)

    def __getitem__(self, index: int) -> complex:
        return self._value[index]

    def __setitem__(self, index: int, value: complex) -> None:
        if self._value[index] != value:
            self._value[index] = value
            self.notifyObservers()

    def __delitem__(self, index: int) -> None:
        del self._value[index]
        self.notifyObservers()

    def insert(self, index: int, value: complex) -> None:
        self._value[index] = value
        self.notifyObservers()

    def __len__(self) -> int:
        return len(self._value)

    def setValue(self, value: Sequence[complex], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notifyObservers()

    def setValueFromString(self, value: str) -> None:
        self.setValue(json.loads(value))  # FIXME
