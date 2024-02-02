from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .object import Object
from .probe import Probe
from .scan import Scan
from .visualize import Plot2D


@dataclass(frozen=True)
class ProductMetadata:
    name: str
    comments: str
    probeEnergyInElectronVolts: float
    detectorDistanceInMeters: float


@dataclass(frozen=True)
class Product:
    metadata: ProductMetadata
    scan: Scan
    probe: Probe
    object_: Object
    costs: Plot2D


class ProductFileReader(ABC):

    @abstractmethod
    def read(self, filePath: Path) -> Product:
        '''reads a product from file'''
        pass


class ProductFileWriter(ABC):

    @abstractmethod
    def write(self, filePath: Path, product: Product) -> None:
        '''writes a product to file'''
        pass
