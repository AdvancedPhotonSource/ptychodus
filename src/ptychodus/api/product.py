from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Final
from dataclasses import dataclass
from pathlib import Path
from sys import getsizeof

from .object import Object
from .probe import Probe
from .scan import Scan

# Source: https://physics.nist.gov/cuu/Constants/index.html
ELECTRON_VOLT_J: Final[float] = 1.602176634e-19
LIGHT_SPEED_M_PER_S: Final[float] = 299792458
PLANCK_CONSTANT_J_PER_HZ: Final[float] = 6.62607015e-34


@dataclass(frozen=True)
class ProductMetadata:
    name: str
    comments: str
    detectorDistanceInMeters: float
    probeEnergyInElectronVolts: float
    probePhotonCount: float
    exposureTimeInSeconds: float

    @property
    def probeEnergyInJoules(self) -> float:
        return self.probeEnergyInElectronVolts * ELECTRON_VOLT_J

    @property
    def probeWavelengthInMeters(self) -> float:
        hc_Jm = PLANCK_CONSTANT_J_PER_HZ * LIGHT_SPEED_M_PER_S

        try:
            return hc_Jm / self.probeEnergyInJoules
        except ZeroDivisionError:
            return 0.0

    @property
    def sizeInBytes(self) -> int:
        sz = getsizeof(self.name)
        sz += getsizeof(self.comments)
        sz += getsizeof(self.detectorDistanceInMeters)
        sz += getsizeof(self.probeEnergyInElectronVolts)
        sz += getsizeof(self.probePhotonCount)
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
