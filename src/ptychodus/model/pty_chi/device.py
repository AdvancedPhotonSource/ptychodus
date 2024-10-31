from collections.abc import Sequence
from typing import overload


class PtyChiDeviceRepository(Sequence[str]):
    def __init__(self, devices: Sequence[str]) -> None:
        self._devices = devices

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[str]: ...

    def __getitem__(self, index: int | slice) -> str | Sequence[str]:
        return self._devices[index]

    def __len__(self) -> int:
        return len(self._devices)
