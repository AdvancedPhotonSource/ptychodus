from collections.abc import Mapping
from typing import Iterator, Optional

from ...api.observer import Observable
from .initializer import ScanInitializer


class ScanRepository(Mapping[str, ScanInitializer], Observable):

    def __init__(self) -> None:
        super().__init__()
        self._initializers: dict[str, ScanInitializer] = dict()

    def __iter__(self) -> Iterator[str]:
        return iter(self._initializers)

    def __getitem__(self, name: str) -> ScanInitializer:
        return self._initializers[name]

    def __len__(self) -> int:
        return len(self._initializers)

    def insertScan(self, initializer: ScanInitializer) -> None:
        name = initializer.nameHint
        index = 0

        while name in self._initializers:
            index += 1
            name = f'{initializer.nameHint}-{index}'

        self._initializers[name] = initializer
        self.notifyObservers()

    def canActivateScan(self, name: str) -> bool:
        initializer = self._initializers.get(name)

        if initializer is not None:
            return initializer.canActivate

        return False

    def canRemoveScan(self, name: str) -> bool:
        return len(self._initializers) > 1

    def removeScan(self, name: str) -> None:
        if self.canRemoveScan(name):
            try:
                initializer = self._initializers.pop(name)
            except KeyError:
                pass
            else:
                initializer.clearObservers()

        self.notifyObservers()
