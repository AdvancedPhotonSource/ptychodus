from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ScanPoint:
    x: Decimal
    y: Decimal


class ScanFileReader(ABC):
    @abstractproperty
    def simpleName(self) -> str:
        pass

    @abstractproperty
    def fileFilter(self) -> str:
        pass

    @abstractmethod
    def read(self, filePath: Path) -> Iterable[ScanPoint]:
        pass


class ScanPointParseError(Exception):
    pass
