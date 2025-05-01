from collections.abc import Sequence
from enum import IntEnum
from typing import overload

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import IntegerParameter


class PtyChiAffineDegreesOfFreedom(IntEnum):
    TRANSLATION = 0
    ROTATION = 1
    SCALING = 2
    SHEARING = 3
    ASYMMETRY = 4


class PtyChiAffineDegreesOfFreedomBitField(Sequence[str], Observable, Observer):
    def __init__(self, parameter: IntegerParameter) -> None:
        super().__init__()
        self._parameter = parameter
        parameter.add_observer(self)

    def is_bit_set(self, bit: int) -> bool:
        value = self._parameter.get_value()
        mask = 1 << bit
        return value & mask != 0

    def set_bit(self, bit: int, is_set: bool) -> None:
        value = self._parameter.get_value()
        mask = 1 << bit

        if is_set:
            value |= mask
        else:
            value &= ~mask

        self._parameter.set_value(value)

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        if isinstance(index, slice):
            return [self[idx] for idx in range(index.start, index.stop, index.step)]

        return PtyChiAffineDegreesOfFreedom(index).name.title()

    def __len__(self) -> int:
        return len(PtyChiAffineDegreesOfFreedom)

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.notify_observers()
