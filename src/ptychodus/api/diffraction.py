"""Diffraction data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload, Any, TypeAlias

import numpy
from scipy import ndimage

from .common import estimate_noise_floor
from .geometry import ImageExtent, PixelGeometry
from .tree import SimpleTreeNode

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

    return per_pattern.max().item()


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


@dataclass(frozen=True)
class DiffractionMetadata:
    """Metadata describing a diffraction dataset (geometry, energy, file path, etc.)."""

    num_patterns_per_array: Sequence[int]
    pattern_dtype: DiffractionPatternDType
    detector_distance_m: float | None = None
    detector_extent: ImageExtent | None = None
    detector_pixel_geometry: PixelGeometry | None = None
    crop_center: CropCenter | None = None
    probe_energy_eV: float | None = None  # noqa: N815
    probe_photon_count: int | None = None
    exposure_time_s: float | None = None
    tomography_angle_deg: float | None = None
    file_path: Path | None = None

    @classmethod
    def create_null(cls, file_path: Path | None = None) -> DiffractionMetadata:
        return cls([], numpy.dtype(numpy.ubyte), file_path=file_path)


class DiffractionDataset(Sequence[DiffractionArray], ABC):
    """A sequence of DiffractionArrays with shared metadata and bad-pixel mask."""

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
    """Concrete DiffractionDataset backed by a list of DiffractionArray objects."""

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
