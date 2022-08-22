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

    def insertScan(self, initializer: ScanInitializer, name: Optional[str] = None) -> None:
        if name is None:
            name = initializer.variant

        initializerName = name
        index = 0

        while initializerName in self._initializers:
            index += 1
            initializerName = f'{name}-{index}'

        self._initializers[initializerName] = initializer
        self.notifyObservers()

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
