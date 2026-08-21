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
from scipy import ndimage

from .constants import format_bytes
from .geometry import ImageExtent, PixelGeometry
from .preprocess.noise import estimate_noise_floor

BadPixels: TypeAlias = numpy.ndarray[tuple[int, int], numpy.dtype[numpy.bool_]]
DiffractionPatternDType: TypeAlias = numpy.dtype[numpy.integer[Any] | numpy.floating[Any]]
DiffractionPatternCounts: TypeAlias = numpy.ndarray[tuple[int], DiffractionPatternDType]
DiffractionPattern: TypeAlias = numpy.ndarray[tuple[int, int], DiffractionPatternDType]
DiffractionPatterns: TypeAlias = numpy.ndarray[tuple[int, int, int], DiffractionPatternDType]
DiffractionIndexes: TypeAlias = numpy.ndarray[tuple[int], numpy.dtype[numpy.integer[Any]]]


def estimate_probe_photon_count(
    patterns: DiffractionPatterns, bad_pixels: BadPixels | None = None
) -> int:
    """Estimate the per-snapshot probe photon count from diffraction patterns.

    Heuristic: total counts of the brightest pattern over good pixels. The
    brightest pattern bounds the photons reaching the detector when the probe
    is least obstructed by the sample.
    """
    if bad_pixels is None:
        per_pattern = numpy.sum(patterns, axis=(-2, -1))
    else:
        per_pattern = numpy.sum(patterns[:, numpy.logical_not(bad_pixels)], axis=-1)

    return int(per_pattern.max())


def zero_bad_pixels(
    patterns: DiffractionPatterns, bad_pixels: BadPixels | None
) -> DiffractionPatterns:
    """Return a copy of `patterns` with bad pixels zeroed.

    Zero is a neutral choice (no photons measured) but does mildly bias any
    consumer toward predicting low intensity at those locations. Returns the
    input unchanged when `bad_pixels` is None or all-False to avoid a copy.
    """
    if bad_pixels is None or not numpy.any(bad_pixels):
        return patterns

    cleaned = patterns.copy()
    cleaned[:, bad_pixels] = 0
    return cleaned


@dataclass(frozen=True)
class CropCenter:
    """Pixel coordinates of the center used when cropping diffraction patterns."""

    position_x_px: int
    position_y_px: int


def estimate_crop_center(
    pattern: DiffractionPattern,
    bad_pixels: BadPixels | None = None,
    *,
    mad_threshold: float = 4.5,
) -> CropCenter:
    """Estimate the pixel centroid of a diffraction pattern.

    Two passes: pass 1 takes the global intensity-weighted center; pass 2
    re-centroids inside a window around the pass-1 estimate so bright
    asymmetric peaks outside the central region cannot bias the result.

    Falls back to the geometric center if all pixels are masked or rejected.
    """
    height, width = pattern.shape[-2:]
    geometric_center = CropCenter(position_x_px=width // 2, position_y_px=height // 2)

    # Median-filter a float copy with bad pixels zeroed.
    working_pattern = pattern.astype(numpy.float64, copy=True)

    if bad_pixels is not None and numpy.any(bad_pixels):
        good_pixel_mask = numpy.logical_not(bad_pixels)
        working_pattern[bad_pixels] = 0.0
    else:
        good_pixel_mask = numpy.ones(pattern.shape[-2:], dtype=bool)

    filtered_pattern = ndimage.median_filter(working_pattern, size=3)

    # Convert intensities to non-negative weights by subtracting the background
    # and rejecting pixels below background + mad_threshold * MAD. Otsu's
    # threshold is used to identify the background pool when the histogram is
    # bimodal; for unimodal noise-only inputs the helper falls back to
    # median/MAD over all good pixels.
    good_pixel_intensities = filtered_pattern[good_pixel_mask]

    if good_pixel_intensities.size == 0:
        return geometric_center

    noise_floor = estimate_noise_floor(good_pixel_intensities)
    background_intensity = noise_floor.background_value
    significance_threshold = noise_floor.get_significance_threshold(mad_threshold)
    centroid_weights = filtered_pattern - background_intensity
    centroid_weights[~good_pixel_mask] = 0.0
    centroid_weights[filtered_pattern < significance_threshold] = 0.0
    numpy.clip(centroid_weights, 0.0, None, out=centroid_weights)

    # Pixel-coordinate axes centered on the array midpoint, so symmetric weight
    # distributions sum to ~0 rather than relying on catastrophic cancellation.
    midpoint_y = (height - 1) / 2.0
    midpoint_x = (width - 1) / 2.0
    centered_y_axis = numpy.arange(height, dtype=numpy.float64).reshape(-1, 1) - midpoint_y
    centered_x_axis = numpy.arange(width, dtype=numpy.float64).reshape(1, -1) - midpoint_x

    # Pass 1: global intensity-weighted centroid.
    coarse_total_weight = float(centroid_weights.sum())

    if coarse_total_weight <= 0.0:
        return geometric_center

    coarse_center_y = midpoint_y + float(
        (centroid_weights * centered_y_axis).sum() / coarse_total_weight
    )
    coarse_center_x = midpoint_x + float(
        (centroid_weights * centered_x_axis).sum() / coarse_total_weight
    )

    # Pass 2: re-centroid inside a square window around the pass-1 estimate.
    half_window_size = min(height, width) // 4
    pixel_y = numpy.arange(height).reshape(-1, 1)
    pixel_x = numpy.arange(width).reshape(1, -1)
    in_central_window = (numpy.abs(pixel_y - coarse_center_y) <= half_window_size) & (
        numpy.abs(pixel_x - coarse_center_x) <= half_window_size
    )
    windowed_weights = centroid_weights * in_central_window
    refined_total_weight = float(windowed_weights.sum())

    if refined_total_weight > 0.0:
        refined_center_y = midpoint_y + float(
            (windowed_weights * centered_y_axis).sum() / refined_total_weight
        )
        refined_center_x = midpoint_x + float(
            (windowed_weights * centered_x_axis).sum() / refined_total_weight
        )
    else:
        refined_center_y = coarse_center_y
        refined_center_x = coarse_center_x

    return CropCenter(
        position_x_px=int(round(refined_center_x)),
        position_y_px=int(round(refined_center_y)),
    )


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


class SimpleDiffractionArray(DiffractionArray):
    """Concrete DiffractionArray backed by in-memory numpy arrays."""

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

    @property
    def nbytes(self) -> int:
        return self._indexes.nbytes + self._patterns.nbytes


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


class AssembledDiffractionData:
    """In-memory store for a complete set of indexed diffraction patterns and their bad-pixel mask."""

    def __init__(
        self,
        indexes: DiffractionIndexes,
        patterns: DiffractionPatterns,
        pixel_geometry: PixelGeometry,
        bad_pixels: BadPixels,
    ) -> None:
        self._indexes = indexes
        self._patterns = patterns
        self._pixel_geometry = pixel_geometry
        self._bad_pixels = bad_pixels

        if indexes.ndim != 1:
            raise ValueError(
                f'Unexpected number of dimensions for indexes! (actual={indexes.ndim} expected=1)'
            )

        if patterns.ndim != 3:
            raise ValueError(
                f'Unexpected number of dimensions for patterns! (actual={patterns.ndim} expected=3)'
            )

        if bad_pixels.ndim != 2:
            raise ValueError(
                f'Unexpected number of dimensions for bad pixels! (actual={bad_pixels.ndim} expected=2)'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Number of indexes does not match number of patterns!')

        if patterns.shape[1:] != bad_pixels.shape:
            raise ValueError(
                'Patterns shape does not match bad pixels shape! '
                f'(actual={patterns.shape[1:]} expected={bad_pixels.shape})'
            )

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=numpy.intp),
            patterns=numpy.zeros((1, 1, 1), dtype=numpy.intp),
            pixel_geometry=PixelGeometry(0, 0),
            bad_pixels=numpy.zeros((1, 1), dtype=numpy.bool_),
        )

    def get_patterns_shape(self) -> tuple[int, int, int]:
        return self._patterns.shape

    def get_patterns_dtype(self) -> DiffractionPatternDType:
        return self._patterns.dtype

    def get_pattern(self, index: int) -> DiffractionPattern:
        return self._patterns[index]

    def get_pixel_geometry(self) -> PixelGeometry:
        return self._pixel_geometry

    def set_pixel_geometry(self, pixel_geometry: PixelGeometry) -> None:
        # Views produced by assemble() keep their creation-time snapshot; they are
        # only used for per-array display (average pattern, counts) and not by
        # reconstruction, so leaving them stale is acceptable.
        self._pixel_geometry = pixel_geometry

    def get_bad_pixels(self) -> BadPixels:
        return self._bad_pixels

    def assemble(self, data: AssembledDiffractionData, offset: int) -> AssembledDiffractionData:
        assembled_indexes = slice(offset, offset + len(data._indexes))

        self._indexes[assembled_indexes] = data._indexes
        indexes_view = self._indexes[assembled_indexes]
        indexes_view.flags.writeable = False

        self._patterns[assembled_indexes, :, :] = data._patterns
        patterns_view = self._patterns[assembled_indexes, :, :]
        patterns_view.flags.writeable = False

        return AssembledDiffractionData(
            indexes=indexes_view,
            patterns=patterns_view,
            pixel_geometry=self._pixel_geometry,
            bad_pixels=data._bad_pixels,
        )

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes[self._indexes >= 0]

    def get_patterns(self) -> DiffractionPatterns:
        return self._patterns[self._indexes >= 0]

    def get_pattern_counts(self) -> DiffractionPatternCounts:
        good_pixels = numpy.logical_not(self._bad_pixels)
        assembled_patterns = self.get_patterns()
        pattern_counts = numpy.sum(assembled_patterns[:, good_pixels], axis=-1)
        return pattern_counts

    def get_average_pattern(self) -> DiffractionPattern:
        assembled_patterns = self.get_patterns()
        return numpy.mean(assembled_patterns, axis=0)

    @property
    def nbytes(self) -> int:
        """Logical size of the arrays this holds.

        A memory-mapped patterns array (see ``load_diffraction_data``) reports its full
        logical size here even though it is backed by disk rather than RAM.
        """
        return self._indexes.nbytes + self._patterns.nbytes + self._bad_pixels.nbytes

    def __str__(self) -> str:
        number, height, width = self._patterns.shape
        dtype = str(self._patterns.dtype)
        return f'{number} x {height}H x {width}W {dtype} [{format_bytes(self._patterns.nbytes)}]'
