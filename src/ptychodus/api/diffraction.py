from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, Any, TypeAlias

import numpy
import numpy.typing

from .geometry import ImageExtent, PixelGeometry
from .tree import SimpleTreeNode

BadPixels: TypeAlias = numpy.typing.NDArray[numpy.bool_]
DiffractionPatterns: TypeAlias = numpy.typing.NDArray[numpy.integer[Any] | numpy.floating[Any]]
DiffractionIndexes: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]


@dataclass(frozen=True)
class CropCenter:
    position_x_px: int
    position_y_px: int


class DiffractionArray:
    @abstractmethod
    def get_label(self) -> str:
        pass

    @abstractmethod
    def get_indexes(self) -> DiffractionIndexes:
        pass

    @abstractmethod
    def get_patterns(self) -> DiffractionPatterns:
        pass

    def get_num_patterns(self) -> int:
        return self.get_patterns().shape[0]


class SimpleDiffractionArray(DiffractionArray):
    def __init__(
        self,
        label: str,
        indexes: DiffractionIndexes,
        patterns: DiffractionPatterns,
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._patterns = patterns

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self) -> DiffractionPatterns:
        return self._patterns


@dataclass(frozen=True)
class DiffractionMetadata:
    num_patterns_per_array: Sequence[int]
    pattern_dtype: numpy.dtype[numpy.integer[Any] | numpy.floating[Any]]
    detector_distance_m: float | None = None
    detector_extent: ImageExtent | None = None
    detector_pixel_geometry: PixelGeometry | None = None
    detector_bit_depth: int | None = None
    crop_center: CropCenter | None = None
    probe_energy_eV: float | None = None  # noqa: N815
    probe_photon_count: int | None = None
    exposure_time_s: float | None = None
    tomography_angle_deg: float | None = None
    file_path: Path | None = None

    @classmethod
    def create_null(cls, file_path: Path | None = None) -> DiffractionMetadata:
        return cls([], numpy.dtype(numpy.ubyte), file_path=file_path)


class DiffractionDataset(Sequence[DiffractionArray]):
    @abstractmethod
    def get_metadata(self) -> DiffractionMetadata:
        pass

    @abstractmethod
    def get_layout(self) -> SimpleTreeNode:
        pass

    @abstractmethod
    def get_bad_pixels(self) -> BadPixels | None:
        pass


class SimpleDiffractionDataset(DiffractionDataset):
    def __init__(
        self,
        metadata: DiffractionMetadata,
        contents_tree: SimpleTreeNode,
        array_list: Sequence[DiffractionArray],
        bad_pixels: BadPixels | None = None,
    ) -> None:
        super().__init__()
        self._metadata = metadata
        self._contents_tree = contents_tree
        self._array_list = array_list
        self._bad_pixels = bad_pixels

    @classmethod
    def create_null(cls, file_path: Path | None = None) -> SimpleDiffractionDataset:
        metadata = DiffractionMetadata.create_null(file_path)
        contents_tree = SimpleTreeNode.create_root(list())
        array_list: list[DiffractionArray] = list()
        return cls(metadata, contents_tree, array_list)

    def get_metadata(self) -> DiffractionMetadata:
        return self._metadata

    def get_layout(self) -> SimpleTreeNode:
        return self._contents_tree

    def get_bad_pixels(self) -> BadPixels | None:
        return self._bad_pixels

    @overload
    def __getitem__(self, index: int) -> DiffractionArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionArray]: ...

    def __getitem__(self, index: int | slice) -> DiffractionArray | Sequence[DiffractionArray]:
        return self._array_list[index]

    def __len__(self) -> int:
        return len(self._array_list)


class DiffractionFileReader(ABC):
    """interface for plugins that read diffraction files"""

    @abstractmethod
    def read(self, file_path: Path) -> DiffractionDataset:
        """reads a diffraction dataset from file"""
        pass


class DiffractionFileWriter(ABC):
    """interface for plugins that write diffraction files"""

    @abstractmethod
    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        """writes a diffraction dataset to file"""
        pass


class BadPixelsFileReader(ABC):
    """interface for plugins that read bad pixel files"""

    @abstractmethod
    def read(self, file_path: Path) -> BadPixels:
        """reads a bad pixels array from file"""
        pass
