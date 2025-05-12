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
    def get_value(self) -> T:
        pass

    @abstractmethod
    def set_value(self, value: T, *, notify: bool = True) -> None:
        pass

    @abstractmethod
    def get_value_as_string(self) -> str:
        pass

    @abstractmethod
    def set_value_from_string(self, value: str) -> None:
        pass

    @abstractmethod
    def copy(self) -> Parameter[T]:
        pass

    def sync_value_to_parent(self) -> None:
        if self._parent is None:
            logger.warning('sync_value_to_parent: parent is None!')
        else:
            self._parent.set_value(self.get_value())

    def sync_value_from_parent(self) -> None:
        if self._parent is None:
            logger.warning('sync_value_from_parent: parent is None!')
        else:
            self.set_value(self._parent.get_value())


class ParameterBase(Parameter[T]):
    def __init__(self, value: T, parent: Parameter[T] | None) -> None:
        super().__init__(parent)
        self._value = value

    def get_value(self) -> T:
        return self._value

    def set_value(self, value: T, *, notify: bool = True) -> None:
        if self._value != value:
            self._value = value

            if notify:
                self.notify_observers()

    def get_value_as_string(self) -> str:
        return repr(self._value)


class StringParameter(ParameterBase[str]):
    def __init__(self, value: str, parent: StringParameter | None) -> None:
        super().__init__(value, parent)

    def set_value_from_string(self, value: str) -> None:
        self.set_value(str(value))

    def get_value_as_string(self) -> str:
        return str(self._value)

    def copy(self) -> StringParameter:
        return StringParameter(self.get_value(), self)


class PathParameter(ParameterBase[Path]):
    def __init__(self, value: Path, parent: PathParameter | None) -> None:
        super().__init__(value, parent)

    def set_value_from_string(self, value: str) -> None:
        self.set_value(Path(value))

    def change_path_prefix(self, find_path_prefix: Path, replacement_path_prefix: Path) -> Path:
        value = self.get_value()

        try:
            relative_path = value.resolve().relative_to(find_path_prefix)
        except ValueError:
            pass
        else:
            return replacement_path_prefix / relative_path

        return value

    def get_value_as_string(self) -> str:
        return str(self._value)

    def copy(self) -> PathParameter:
        return PathParameter(self.get_value(), self)


class UUIDParameter(ParameterBase[UUID]):
    def __init__(self, value: UUID, parent: UUIDParameter | None) -> None:
        super().__init__(value, parent)

    def set_value_from_string(self, value: str) -> None:
        self.set_value(UUID(value))

    def get_value_as_string(self) -> str:
        return str(self._value)

    def copy(self) -> UUIDParameter:
        return UUIDParameter(self.get_value(), self)


class BooleanParameter(ParameterBase[bool]):
    TRUE_VALUES: Final = ('1', 'true', 't', 'yes', 'y')

    def __init__(self, value: bool, parent: BooleanParameter | None) -> None:
        super().__init__(value, parent)

    def set_value_from_string(self, value: str) -> None:
        self.set_value(value.lower() in BooleanParameter.TRUE_VALUES)

    def copy(self) -> BooleanParameter:
        return BooleanParameter(self.get_value(), self)


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

    def get_minimum(self) -> int | None:
        return self._minimum

    def get_maximum(self) -> int | None:
        return self._maximum

    def get_value(self) -> int:
        value = super().get_value()

        if self._minimum is not None:
            value = max(self._minimum, value)

        if self._maximum is not None:
            value = min(value, self._maximum)

        return value

    def set_value_from_string(self, value: str) -> None:
        self.set_value(int(value))

    def copy(self) -> IntegerParameter:
        return IntegerParameter(
            self.get_value(), self, minimum=self.get_minimum(), maximum=self.get_maximum()
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

    def get_minimum(self) -> float | None:
        return self._minimum

    def get_maximum(self) -> float | None:
        return self._maximum

    def get_value(self) -> float:
        value = super().get_value()

        if self._minimum is not None:
            value = max(self._minimum, value)

        if self._maximum is not None:
            value = min(value, self._maximum)

        return value

    def set_value(self, value: float, *, notify: bool = True) -> None:
        super().set_value(float(value), notify=notify)

    def set_value_from_string(self, value: str) -> None:
        self.set_value(float(value))

    def copy(self) -> RealParameter:
        return RealParameter(
            self.get_value(), self, minimum=self.get_minimum(), maximum=self.get_maximum()
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
            self.notify_observers()

    def __delitem__(self, index: int) -> None:
        del self._value[index]
        self.notify_observers()

    def insert(self, index: int, value: int) -> None:
        self._value.insert(index, value)
        self.notify_observers()

    def __len__(self) -> int:
        return len(self._value)

    def set_value(self, value: Sequence[int], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notify_observers()

    def get_value_as_string(self) -> str:
        return ','.join(repr(value) for value in self)

    def set_value_from_string(self, value: str) -> None:
        new_value: list[int] = list()

        for xstr in value.split(','):
            if xstr:
                new_value.append(int(xstr))

        self.set_value(new_value)

    def copy(self) -> IntegerSequenceParameter:
        return IntegerSequenceParameter(self.get_value(), self)


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
            self.notify_observers()

    def __delitem__(self, index: int) -> None:
        del self._value[index]
        self.notify_observers()

    def insert(self, index: int, value: float) -> None:
        self._value.insert(index, value)
        self.notify_observers()

    def __len__(self) -> int:
        return len(self._value)

    def set_value(self, value: Sequence[float], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notify_observers()

    def get_value_as_string(self) -> str:
        return ','.join(repr(value) for value in self)

    def set_value_from_string(self, value: str) -> None:
        tmp: list[float] = list()

        for xstr in value.split(','):
            if xstr:
                try:
                    x = float(xstr)
                except ValueError:
                    x = float('nan')

                tmp.append(x)

        self.set_value(tmp)

    def copy(self) -> RealSequenceParameter:
        return RealSequenceParameter(self.get_value(), self)


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
            self.notify_observers()

    def __delitem__(self, index: int) -> None:
        del self._value[index]
        self.notify_observers()

    def insert(self, index: int, value: complex) -> None:
        self._value.insert(index, value)
        self.notify_observers()

    def __len__(self) -> int:
        return len(self._value)

    def set_value(self, value: Sequence[complex], *, notify: bool = True) -> None:
        if self._value != value:
            self._value = list(value)

            if notify:
                self.notify_observers()

    def get_value_as_string(self) -> str:
        return ','.join(repr(value) for value in self)

    def set_value_from_string(self, value: str) -> None:
        tmp: list[complex] = list()

        for xstr in value.split(','):
            if xstr:
                try:
                    x = complex(xstr)
                except ValueError:
                    x = float('nan') * 1j

                tmp.append(x)

        self.set_value(tmp)

    def copy(self) -> ComplexSequenceParameter:
        return ComplexSequenceParameter(self.get_value(), self)


class ParameterGroup(Observable, Observer):
    def __init__(self) -> None:
        super().__init__()
        self._parameters: dict[str, Parameter[Any]] = dict()
        self._groups: dict[str, ParameterGroup] = dict()

    def parameters(self) -> Mapping[str, Parameter[Any]]:
        return self._parameters

    def _add_parameter(self, name: str, parameter: Parameter[Any]) -> None:
        if self._parameters.setdefault(name, parameter) is parameter:
            parameter.add_observer(self)
        else:
            raise ValueError(f'Parameter "{name}" already exists!')

    def create_string_parameter(self, name: str, value: str) -> StringParameter:
        parameter = StringParameter(value, parent=None)
        self._add_parameter(name, parameter)
        return parameter

    def create_path_parameter(self, name: str, value: Path) -> PathParameter:
        parameter = PathParameter(value, parent=None)
        self._add_parameter(name, parameter)
        return parameter

    def create_uuid_parameter(self, name: str, value: UUID) -> UUIDParameter:
        parameter = UUIDParameter(value, parent=None)
        self._add_parameter(name, parameter)
        return parameter

    def create_boolean_parameter(self, name: str, value: bool) -> BooleanParameter:
        parameter = BooleanParameter(value, parent=None)
        self._add_parameter(name, parameter)
        return parameter

    def create_integer_parameter(
        self, name: str, value: int, *, minimum: int | None = None, maximum: int | None = None
    ) -> IntegerParameter:
        parameter = IntegerParameter(value, parent=None, minimum=minimum, maximum=maximum)
        self._add_parameter(name, parameter)
        return parameter

    def create_integer_sequence_parameter(
        self, name: str, value: Sequence[int]
    ) -> IntegerSequenceParameter:
        parameter = IntegerSequenceParameter(value, parent=None)
        self._add_parameter(name, parameter)
        return parameter

    def create_real_parameter(
        self, name: str, value: float, *, minimum: float | None = None, maximum: float | None = None
    ) -> RealParameter:
        parameter = RealParameter(value, parent=None, minimum=minimum, maximum=maximum)
        self._add_parameter(name, parameter)
        return parameter

    def create_real_sequence_parameter(
        self, name: str, value: Sequence[float]
    ) -> RealSequenceParameter:
        parameter = RealSequenceParameter(value, parent=None)
        self._add_parameter(name, parameter)
        return parameter

    def create_complex_sequence_parameter(
        self, name: str, value: Sequence[complex]
    ) -> ComplexSequenceParameter:
        parameter = ComplexSequenceParameter(value, parent=None)
        self._add_parameter(name, parameter)
        return parameter

    def groups(self) -> Mapping[str, ParameterGroup]:
        return self._groups

    def _add_group(self, name: str, group: ParameterGroup, *, observe: bool = False) -> None:
        if self._groups.setdefault(name, group) is group:
            if observe:
                group.add_observer(self)
        else:
            raise ValueError(f'Group "{name}" already exists!')

    def _remove_group(self, name: str) -> None:
        try:
            group = self._groups.pop(name)
        except KeyError:
            pass
        else:
            group.remove_observer(self)

    def create_group(self, name: str) -> ParameterGroup:
        group = ParameterGroup()
        self._add_group(name, group)
        return group

    def get_group(self, name: str) -> ParameterGroup:
        return self._groups[name]

    def _update(self, observable: Observable) -> None:
        if observable in self._parameters.values():
            self.notify_observers()
        elif observable in self._groups.values():
            self.notify_observers()
