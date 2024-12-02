from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, MutableSequence, Sequence
from pathlib import Path
from typing import Any, Final, Generic, TypeVar
from uuid import UUID
import logging

from .observer import Observable, Observer

__all__ = [
    'Parameter',
    'ParameterGroup',
]

logger = logging.getLogger(__name__)

T = TypeVar('T')


class Parameter(ABC, Generic[T], Observable):
    def __init__(self, parent: Parameter[T] | None = None) -> None:
        super().__init__()
        self._parent = parent

    @abstractmethod
    def getValue(self) -> T:
        pass

    @abstractmethod
    def setValue(self, value: T, *, notify: bool = True) -> None:
        pass

    @abstractmethod
    def getValueAsString(self) -> str:
        pass

    @abstractmethod
    def setValueFromString(self, value: str) -> None:
        pass

    @abstractmethod
    def copy(self) -> Parameter[T]:
        pass

    def syncValueToParent(self) -> None:
        if self._parent is None:
            logger.warning('syncValueToParent: parent is None!')
        else:
            self._parent.setValue(self.getValue())

    def syncValueFromParent(self) -> None:
        if self._parent is None:
            logger.warning('syncValueFromParent: parent is None!')
        else:
            self.setValue(self._parent.getValue())


class ParameterBase(Parameter[T]):
    def __init__(self, value: T, parent: Parameter[T] | None) -> None:
        super().__init__(parent)
        self._value = value

    def getValue(self) -> T:
        return self._value

    def setValue(self, value: T, *, notify: bool = True) -> None:
        if self._value != value:
            self._value = value

            if notify:
                self.notifyObservers()

    def getValueAsString(self) -> str:
        return repr(self._value)


class StringParameter(ParameterBase[str]):
    def __init__(self, value: str, parent: StringParameter | None) -> None:
        super().__init__(value, parent)

    def setValueFromString(self, value: str) -> None:
        self.setValue(str(value))

    def getValueAsString(self) -> str:
        return str(self._value)

    def copy(self) -> StringParameter:
        return StringParameter(self.getValue(), self)


class PathParameter(ParameterBase[Path]):
    def __init__(self, value: Path, parent: PathParameter | None) -> None:
        super().__init__(value, parent)

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

    def getValueAsString(self) -> str:
        return str(self._value)

    def copy(self) -> PathParameter:
        return PathParameter(self.getValue(), self)


class UUIDParameter(ParameterBase[UUID]):
    def __init__(self, value: UUID, parent: UUIDParameter | None) -> None:
        super().__init__(value, parent)

    def setValueFromString(self, value: str) -> None:
        self.setValue(UUID(value))

    def getValueAsString(self) -> str:
        return str(self._value)

    def copy(self) -> UUIDParameter:
        return UUIDParameter(self.getValue(), self)


class BooleanParameter(ParameterBase[bool]):
    TRUE_VALUES: Final = ('1', 'true', 't', 'yes', 'y')

    def __init__(self, value: bool, parent: BooleanParameter | None) -> None:
        super().__init__(value, parent)

    def setValueFromString(self, value: str) -> None:
        self.setValue(value.lower() in BooleanParameter.TRUE_VALUES)

    def copy(self) -> BooleanParameter:
        return BooleanParameter(self.getValue(), self)


class IntegerParameter(ParameterBase[int]):
    def __init__(
        self,
        value: int,
        parent: IntegerParameter | None,
        *,
        minimum: int | None = None,
        maximum: int | None = None,
    ) -> None:
        super().__init__(value, parent)
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

    def copy(self) -> IntegerParameter:
        return IntegerParameter(
            self.getValue(), self, minimum=self.getMinimum(), maximum=self.getMaximum()
        )


class RealParameter(ParameterBase[float]):
    def __init__(
        self,
        value: float,
        parent: RealParameter | None,
        *,
        minimum: float | None = None,
        maximum: float | None = None,
    ) -> None:
        super().__init__(value, parent)
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

    def copy(self) -> RealParameter:
        return RealParameter(
            self.getValue(), self, minimum=self.getMinimum(), maximum=self.getMaximum()
        )


class IntegerSequenceParameter(ParameterBase[MutableSequence[int]]):
    def __init__(self, value: Sequence[int], parent: IntegerSequenceParameter | None) -> None:
        super().__init__(list(value), parent)

    def __iter__(self) -> Iterator[int]:
        return iter(self._value)

    def __getitem__(self, index: int) -> int:
        return self._value[index]

    def __setitem__(self, index: int, value: int) -> None:
        if self._value[index] != value:
            self._value[index] = value
            self.notifyObservers()

    def __delitem__(self, index: int) -> None:
        del self._value[index]
        self.notifyObservers()

    def insert(self, index: int, value: int) -> None:
        self._value.insert(index, value)
        self.notifyObservers()

    def __len__(self) -> int:
        return len(self._value)

    def setValue(self, value: Sequence[int], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notifyObservers()

    def getValueAsString(self) -> str:
        return ','.join(repr(value) for value in self)

    def setValueFromString(self, value: str) -> None:
        newValue: list[int] = list()

        for xstr in value.split(','):
            if xstr:
                newValue.append(int(xstr))

        self.setValue(newValue)

    def copy(self) -> IntegerSequenceParameter:
        return IntegerSequenceParameter(self.getValue(), self)


class RealSequenceParameter(ParameterBase[MutableSequence[float]]):
    def __init__(self, value: Sequence[float], parent: RealSequenceParameter | None) -> None:
        super().__init__(list(value), parent)

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
        self._value.insert(index, value)
        self.notifyObservers()

    def __len__(self) -> int:
        return len(self._value)

    def setValue(self, value: Sequence[float], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notifyObservers()

    def getValueAsString(self) -> str:
        return ','.join(repr(value) for value in self)

    def setValueFromString(self, value: str) -> None:
        tmp: list[float] = list()

        for xstr in value.split(','):
            try:
                x = float(xstr)
            except ValueError:
                x = float('nan')

            tmp.append(x)

        self.setValue(tmp)

    def copy(self) -> RealSequenceParameter:
        return RealSequenceParameter(self.getValue(), self)


class ComplexSequenceParameter(ParameterBase[MutableSequence[complex]]):
    def __init__(self, value: Sequence[complex], parent: ComplexSequenceParameter | None) -> None:
        super().__init__(list(value), parent)

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
        self._value.insert(index, value)
        self.notifyObservers()

    def __len__(self) -> int:
        return len(self._value)

    def setValue(self, value: Sequence[complex], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notifyObservers()

    def getValueAsString(self) -> str:
        return ','.join(repr(value) for value in self)

    def setValueFromString(self, value: str) -> None:
        tmp: list[complex] = list()

        for xstr in value.split(','):
            try:
                x = complex(xstr)
            except ValueError:
                x = float('nan') * 1j

            tmp.append(x)

        self.setValue(tmp)

    def copy(self) -> ComplexSequenceParameter:
        return ComplexSequenceParameter(self.getValue(), self)


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

    def createStringParameter(self, name: str, value: str) -> StringParameter:
        parameter = StringParameter(value, parent=None)
        self._addParameter(name, parameter)
        return parameter

    def createPathParameter(self, name: str, value: Path) -> PathParameter:
        parameter = PathParameter(value, parent=None)
        self._addParameter(name, parameter)
        return parameter

    def createUUIDParameter(self, name: str, value: UUID) -> UUIDParameter:
        parameter = UUIDParameter(value, parent=None)
        self._addParameter(name, parameter)
        return parameter

    def createBooleanParameter(self, name: str, value: bool) -> BooleanParameter:
        parameter = BooleanParameter(value, parent=None)
        self._addParameter(name, parameter)
        return parameter

    def createIntegerParameter(
        self, name: str, value: int, *, minimum: int | None = None, maximum: int | None = None
    ) -> IntegerParameter:
        parameter = IntegerParameter(value, parent=None, minimum=minimum, maximum=maximum)
        self._addParameter(name, parameter)
        return parameter

    def createIntegerSequenceParameter(
        self, name: str, value: Sequence[int]
    ) -> IntegerSequenceParameter:
        parameter = IntegerSequenceParameter(value, parent=None)
        self._addParameter(name, parameter)
        return parameter

    def createRealParameter(
        self, name: str, value: float, *, minimum: float | None = None, maximum: float | None = None
    ) -> RealParameter:
        parameter = RealParameter(value, parent=None, minimum=minimum, maximum=maximum)
        self._addParameter(name, parameter)
        return parameter

    def createRealSequenceParameter(
        self, name: str, value: Sequence[float]
    ) -> RealSequenceParameter:
        parameter = RealSequenceParameter(value, parent=None)
        self._addParameter(name, parameter)
        return parameter

    def createComplexSequenceParameter(
        self, name: str, value: Sequence[complex]
    ) -> ComplexSequenceParameter:
        parameter = ComplexSequenceParameter(value, parent=None)
        self._addParameter(name, parameter)
        return parameter

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
