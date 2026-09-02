"""Diffraction data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from sys import getsizeof
from pathlib import Path
from typing import overload, Any, TypeAlias

import numpy

from .geometry import ImageExtent, PixelGeometry

BadPixels: TypeAlias = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool_]]
DiffractionPatternDType: TypeAlias = numpy.dtype[numpy.integer[Any] | numpy.floating[Any]]
DiffractionPatternCounts: TypeAlias = numpy.ndarray[tuple[int], DiffractionPatternDType]
DiffractionPattern: TypeAlias = numpy.ndarray[tuple[int, int], DiffractionPatternDType]
DiffractionPatterns: TypeAlias = numpy.ndarray[tuple[int, int, int], DiffractionPatternDType]
DiffractionIndexes: TypeAlias = numpy.ndarray[tuple[int], numpy.dtype[numpy.integer[Any]]]
DiffractionPatternPhotonFluxes: TypeAlias = numpy.ndarray[
    tuple[int], numpy.dtype[numpy.floating[Any]]
]


@dataclass(frozen=True)
class CropCenter:
    """Pixel coordinates of the center used when cropping diffraction patterns."""

    x_px: int
    y_px: int


@dataclass(frozen=True)
class CropRegion:
    """Rectangular crop window on a detector: center in pixels + width/height in pixels.

    The window matches `CropStep`'s slice semantics: half-open
    `[x_px - width_px // 2, x_px + width_px // 2)` on x, same on y.
    """

    x_px: int
    y_px: int
    width_px: int
    height_px: int

    @classmethod
    def from_center_extent(cls, center: CropCenter, extent: ImageExtent) -> CropRegion:
        """Build a region from a separate center and extent."""
        return cls(
            x_px=center.x_px,
            y_px=center.y_px,
            width_px=extent.width_px,
            height_px=extent.height_px,
        )

    @classmethod
    def from_largest_pow2(cls, center: CropCenter, detector_extent: ImageExtent) -> CropRegion:
        """Largest power-of-two square region centered at `center` fitting in `detector_extent`.

        Returns a zero-size region when `center` lies on the detector boundary or outside it
        (no positive-radius square exists).
        """
        max_radius = min(
            center.x_px,
            detector_extent.width_px - center.x_px,
            center.y_px,
            detector_extent.height_px - center.y_px,
        )
        if max_radius < 1:
            return cls(x_px=center.x_px, y_px=center.y_px, width_px=0, height_px=0)
        # Largest s = 2^k satisfying s <= 2 * max_radius. bit_length() gives floor(log2)+1.
        size = 1 << ((2 * max_radius).bit_length() - 1)
        return cls(x_px=center.x_px, y_px=center.y_px, width_px=size, height_px=size)

    def clamp_to_detector_extent(self, detector_extent: ImageExtent) -> CropRegion:
        """Return a new region shrunk and shifted so its window fits inside `detector_extent`.

        Width and height are capped at the detector's (floored at 1); the center is then
        clamped into `[radius, detector - radius]` on each axis, so the half-open window
        `[c - radius, c + radius)` lies entirely inside `[0, detector)`.
        """
        width = max(1, min(self.width_px, detector_extent.width_px))
        height = max(1, min(self.height_px, detector_extent.height_px))
        radius_x = width // 2
        radius_y = height // 2
        x = max(radius_x, min(self.x_px, detector_extent.width_px - radius_x))
        y = max(radius_y, min(self.y_px, detector_extent.height_px - radius_y))
        return CropRegion(x_px=x, y_px=y, width_px=width, height_px=height)


class DiffractionArray(ABC):
    """A block of diffraction patterns with associated scan indexes."""

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

    def get_probe_photon_flux_Hz(self) -> DiffractionPatternPhotonFluxes | None:  # noqa: N802
        """Per-pattern incident probe photon flux (photons per second).

        Beamline readers with hardware flux measurements (ion chamber, BPM,
        ring-current-normalized upstream reading) override this. The default
        returns None to signal no measurement is available, and downstream
        assembly falls back to per-pattern detector totals.
        """
        return None


class SimpleDiffractionArray(DiffractionArray):
    """Concrete DiffractionArray backed by in-memory numpy arrays."""

    def __init__(
        self,
        label: str,
        indexes: DiffractionIndexes,
        patterns: DiffractionPatterns,
        probe_photon_flux_Hz: DiffractionPatternPhotonFluxes | None = None,  # noqa: N803
    ) -> None:
        super().__init__()
        self._label = label
        self._indexes = indexes
        self._patterns = patterns
        self._probe_photon_flux_Hz = probe_photon_flux_Hz

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes

    def get_patterns(self) -> DiffractionPatterns:
        return self._patterns

    def get_probe_photon_flux_Hz(self) -> DiffractionPatternPhotonFluxes | None:  # noqa: N802
        return self._probe_photon_flux_Hz

    @property
    def nbytes(self) -> int:
        sz = self._indexes.nbytes + self._patterns.nbytes
        if self._probe_photon_flux_Hz is not None:
            sz += self._probe_photon_flux_Hz.nbytes
        return sz


class Polarization(StrEnum):
    """Beam polarization state (used by XMCD analysis).

    Stored as its string value for stable, human-readable HDF5/INI round-trip.
    """

    LEFT_CIRCULAR = 'left_circular'
    RIGHT_CIRCULAR = 'right_circular'


@dataclass(frozen=True)
class DiffractionMetadata:
    """Metadata describing a diffraction dataset (geometry, energy, file path, etc.)."""

    num_patterns_per_array: Sequence[int]
    pattern_dtype: DiffractionPatternDType
    detector_extent: ImageExtent
    detector_distance_m: float | None = None
    detector_pixel_geometry: PixelGeometry | None = None
    crop_center: CropCenter | None = None
    probe_energy_eV: float | None = None  # noqa: N815
    probe_photon_count: int | None = None
    exposure_time_s: float | None = None
    tomography_angle_deg: float | None = None
    tilt_angle_deg: float | None = None
    polarization: Polarization | None = None
    file_path: Path | None = None

    @property
    def nbytes(self) -> int:
        sz = getsizeof(self.num_patterns_per_array)
        sz += getsizeof(self.pattern_dtype)
        sz += getsizeof(self.detector_extent)
        sz += getsizeof(self.detector_distance_m)
        sz += getsizeof(self.detector_pixel_geometry)
        sz += getsizeof(self.crop_center)
        sz += getsizeof(self.probe_energy_eV)
        sz += getsizeof(self.probe_photon_count)
        sz += getsizeof(self.exposure_time_s)
        sz += getsizeof(self.tomography_angle_deg)
        sz += getsizeof(self.tilt_angle_deg)
        sz += getsizeof(self.polarization)
        sz += getsizeof(self.file_path)
        return sz

    @classmethod
    def create_null(cls, file_path: Path | None = None) -> DiffractionMetadata:
        return cls(
            num_patterns_per_array=[],
            pattern_dtype=numpy.dtype(numpy.ubyte),
            detector_extent=ImageExtent(width_px=0, height_px=0),
            file_path=file_path,
        )


@dataclass
class DiffractionDatasetLayoutNode:
    """Node in the layout tree returned by DiffractionFileReader.

    Each node describes one entry (HDF5 group/dataset/attribute, NPZ array,
    TIFF file, ...) with three display columns: name, dtype, details.
    """

    name: str
    dtype: str
    details: str
    parent: DiffractionDatasetLayoutNode | None = None
    children: list[DiffractionDatasetLayoutNode] = field(default_factory=list)

    @classmethod
    def create_root(cls) -> DiffractionDatasetLayoutNode:
        return cls(name='', dtype='', details='')

    def add_child(self, name: str, dtype: str, details: str) -> DiffractionDatasetLayoutNode:
        child = DiffractionDatasetLayoutNode(name, dtype, details, parent=self)
        self.children.append(child)
        return child

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def row(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.children.index(self)


class DiffractionDataset(Sequence[DiffractionArray], ABC):
    """A sequence of DiffractionArrays with shared metadata and bad-pixel mask.

    Every dataset owns a bad-pixel mask; callers can always rely on
    ``get_bad_pixels()`` returning a real array, defaulting to all-good pixels
    when the source data did not include a mask.
    """

    @abstractmethod
    def get_metadata(self) -> DiffractionMetadata:
        pass

    @abstractmethod
    def get_layout(self) -> DiffractionDatasetLayoutNode:
        pass

    @abstractmethod
    def get_bad_pixels(self) -> BadPixels:
        pass


class SimpleDiffractionDataset(DiffractionDataset):
    """Concrete DiffractionDataset backed by a list of DiffractionArray objects."""

    def __init__(
        self,
        metadata: DiffractionMetadata,
        contents_tree: DiffractionDatasetLayoutNode,
        array_list: Sequence[DiffractionArray],
        bad_pixels: BadPixels | None = None,
    ) -> None:
        super().__init__()
        self._metadata = metadata
        self._contents_tree = contents_tree
        self._array_list = array_list

        extent = metadata.detector_extent

        if bad_pixels is None:
            bad_pixels = numpy.zeros((extent.height_px, extent.width_px), dtype=numpy.bool_)
        elif bad_pixels.shape != extent.get_shape():
            raise ValueError(
                f'Bad pixels shape {bad_pixels.shape} does not match '
                f'detector extent {extent.get_shape()}.'
            )

        self._bad_pixels = bad_pixels

    @classmethod
    def create_null(cls, file_path: Path | None = None) -> SimpleDiffractionDataset:
        metadata = DiffractionMetadata.create_null(file_path)
        contents_tree = DiffractionDatasetLayoutNode.create_root()
        array_list: list[DiffractionArray] = list()
        return cls(metadata, contents_tree, array_list)

    def get_metadata(self) -> DiffractionMetadata:
        return self._metadata

    def get_layout(self) -> DiffractionDatasetLayoutNode:
        return self._contents_tree

    def get_bad_pixels(self) -> BadPixels:
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
    """Plugin interface for reading diffraction files."""

    @abstractmethod
    def read(self, file_path: Path) -> DiffractionDataset:
        """Read a diffraction dataset from file."""
        pass


class DiffractionFileWriter(ABC):
    """Plugin interface for writing diffraction files."""

    @abstractmethod
    def write(self, file_path: Path, dataset: DiffractionDataset) -> None:
        """Write a diffraction dataset to file."""
        pass


class BadPixelsFileReader(ABC):
    """Plugin interface for reading bad-pixel mask files."""

    @abstractmethod
    def read(self, file_path: Path) -> BadPixels:
        """Read a bad-pixel mask from file."""
        pass
