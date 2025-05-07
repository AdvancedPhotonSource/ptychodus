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

PatternDataType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]
PatternIndexesType: TypeAlias = numpy.typing.NDArray[numpy.integer[Any]]


@dataclass(frozen=True)
class CropCenter:
    position_x_px: int
    position_y_px: int


class DiffractionPatternArray:
    @abstractmethod
    def get_label(self) -> str:
        pass

    @abstractmethod
    def get_indexes(self) -> PatternIndexesType:
        pass

    @abstractmethod
    def get_data(self) -> PatternDataType:
        pass

    def get_num_patterns(self) -> int:
        return self.get_data().shape[0]


class SimpleDiffractionPatternArray(DiffractionPatternArray):
    def __init__(
        self,
        label: str,
        indexes: PatternIndexesType,
        data: PatternDataType,
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._data = data

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> PatternIndexesType:
        return self._indexes

    def get_data(self) -> PatternDataType:
        return self._data


@dataclass(frozen=True)
class DiffractionMetadata:
    num_patterns_per_array: int
    num_patterns_total: int
    pattern_dtype: numpy.dtype[numpy.integer[Any]]
    detector_distance_m: float | None = None
    detector_extent: ImageExtent | None = None
    detector_pixel_geometry: PixelGeometry | None = None
    detector_bit_depth: int | None = None
    crop_center: CropCenter | None = None
    probe_photon_count: int | None = None
    probe_energy_eV: float | None = None  # noqa: N815
    tomography_angle_deg: float | None = None
    file_path: Path | None = None

    @classmethod
    def create_null(cls, file_path: Path | None = None) -> DiffractionMetadata:
        return cls(0, 0, numpy.dtype(numpy.ubyte), file_path=file_path)


class DiffractionDataset(Sequence[DiffractionPatternArray]):
    @abstractmethod
    def get_metadata(self) -> DiffractionMetadata:
        pass

    @abstractmethod
    def get_contents_tree(self) -> SimpleTreeNode:
        pass


class SimpleDiffractionDataset(DiffractionDataset):
    def __init__(
        self,
        metadata: DiffractionMetadata,
        contents_tree: SimpleTreeNode,
        array_list: list[DiffractionPatternArray],
    ) -> None:
        super().__init__()
        self._metadata = metadata
        self._contents_tree = contents_tree
        self._array_list = array_list

    @classmethod
    def create_null(cls, file_path: Path | None = None) -> SimpleDiffractionDataset:
        metadata = DiffractionMetadata.create_null(file_path)
        contents_tree = SimpleTreeNode.create_root(list())
        array_list: list[DiffractionPatternArray] = list()
        return cls(metadata, contents_tree, array_list)

    def get_metadata(self) -> DiffractionMetadata:
        return self._metadata

    def get_contents_tree(self) -> SimpleTreeNode:
        return self._contents_tree

    @overload
    def __getitem__(self, index: int) -> DiffractionPatternArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[DiffractionPatternArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DiffractionPatternArray | Sequence[DiffractionPatternArray]:
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
