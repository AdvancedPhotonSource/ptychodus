from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from sys import getsizeof

from .object import Object
from .probe import Probe
from .scan import Scan


@dataclass(frozen=True)
class ProductMetadata:
    name: str
    comments: str
    detectorDistanceInMeters: float
    probeEnergyInElectronVolts: float
    probePhotonsPerSecond: float
    exposureTimeInSeconds: float

    @property
    def sizeInBytes(self) -> int:
        sz = getsizeof(self.name)
        sz += getsizeof(self.comments)
        sz += getsizeof(self.detectorDistanceInMeters)
        sz += getsizeof(self.probeEnergyInElectronVolts)
        sz += getsizeof(self.probePhotonsPerSecond)
        sz += getsizeof(self.exposureTimeInSeconds)
        return sz


@dataclass(frozen=True)
class Product:
    metadata: ProductMetadata
    scan: Scan
    probe: Probe
    object_: Object
    costs: Sequence[float]

    @property
    def sizeInBytes(self) -> int:
        sz = self.metadata.sizeInBytes
        sz += self.scan.sizeInBytes
        sz += self.probe.sizeInBytes
        sz += self.object_.sizeInBytes
        return sz


class ProductFileReader(ABC):
    @abstractmethod
    def read(self, filePath: Path) -> Product:
        """reads a product from file"""
        pass


class ProductFileWriter(ABC):
    @abstractmethod
    def write(self, filePath: Path, product: Product) -> None:
        """writes a product to file"""
        pass
