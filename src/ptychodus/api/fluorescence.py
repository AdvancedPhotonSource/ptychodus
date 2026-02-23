"""Fluorescence data structures and enhancement plugin interfaces."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .common import RealArrayType
from .product import Product


@dataclass(frozen=True)
class ElementMap:
    """2D spatial map of fluorescence signal for a single element in counts per second."""

    name: str
    counts_per_second: RealArrayType


@dataclass(frozen=True)
class FluorescenceDataset:
    """Collection of element maps with metadata."""

    element_maps: Sequence[ElementMap]
    counts_per_second_path: str
    channel_names_path: str

    # TODO need to communicate association between element map pixels and scan order.
    #      integer-valued, same shape as counts_per_second
    # scan_indexes: IntegerArray


class FluorescenceEnhancingAlgorithm(ABC):
    """Algorithm that uses a ptychography data product to enhance a fluorescence dataset."""

    @abstractmethod
    def enhance(self, dataset: FluorescenceDataset, product: Product) -> FluorescenceDataset:
        pass


class FluorescenceFileReader(ABC):
    """Plugin interface for reading fluorescence datasets."""

    @abstractmethod
    def read(self, file_path: Path) -> FluorescenceDataset:
        """Read a fluorescence dataset from file."""
        pass


class FluorescenceFileWriter(ABC):
    """Plugin interface for writing fluorescence datasets."""

    @abstractmethod
    def write(self, file_path: Path, dataset: FluorescenceDataset) -> None:
        """Write a fluorescence dataset to file."""
        pass


class UpscalingStrategy(ABC):
    """Remaps element concentrations from the regular scan grid to the upscaled grid using ptychography-corrected probe positions."""

    @abstractmethod
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        pass


class DeconvolutionStrategy(ABC):
    """Deconvolves the probe kernel from the accumulated array to enhance the element map."""

    @abstractmethod
    def __call__(self, emap: ElementMap, product: Product) -> ElementMap:
        pass
