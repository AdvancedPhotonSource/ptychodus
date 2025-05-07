from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .product import Product
from .typing import RealArrayType


@dataclass(frozen=True)
class ElementMap:
    name: str
    counts_per_second: RealArrayType


@dataclass(frozen=True)
class FluorescenceDataset:
    element_maps: Sequence[ElementMap]
    counts_per_second_path: str
    channel_names_path: str

    # TODO need to communicate association between element map pixels and scan order.
    #      integer-valued, same shape as counts_per_second
    # scan_indexes: IntegerArray


class FluorescenceEnhancingAlgorithm(ABC):
    @abstractmethod
    def enhance(self, dataset: FluorescenceDataset, product: Product) -> FluorescenceDataset:
        pass


class FluorescenceFileReader(ABC):
    @abstractmethod
    def read(self, file_path: Path) -> FluorescenceDataset:
        """reads a fluorescence dataset from file"""
        pass


class FluorescenceFileWriter(ABC):
    @abstractmethod
    def write(self, file_path: Path, dataset: FluorescenceDataset) -> None:
        """writes a fluorescence dataset to file"""
        pass


class UpscalingStrategy(ABC):
    """Uses ptychography-corrected scan positions to remap element
    concentrations from the regular scan grid to the upscaled grid"""

    @abstractmethod
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        pass


class DeconvolutionStrategy(ABC):
    """Deconvolves the kernel from the accumulated array to obtain the
    resolution-enhanced element map"""

    @abstractmethod
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        pass
