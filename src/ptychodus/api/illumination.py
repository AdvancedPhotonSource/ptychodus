"""Illumination-map data structures and the algorithm that builds a photon-count canvas
from a ptychography product by summing subpixel-shifted probe intensities."""

from __future__ import annotations
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
import math

import numpy
import scipy.ndimage
import scipy.signal
from scipy.spatial import KDTree

from .typing import RealArrayType
from .fourier import fourier_shift_2d
from .geometry import PixelGeometry
from .object import ObjectCenter
from .probe import PatchBounds
from .probe_positions import ProbePosition
from .product import Product
from .propagate import compute_product_geometry


def _iter_probe_patches(
    product: Product,
) -> Iterator[tuple[ProbePosition, PatchBounds, RealArrayType]]:
    """Yield the object-canvas placement and intensity patch of every scan position.

    Each patch is the incoherent-mode intensity sum of that position's probe,
    Fourier-shifted by the residual subpixel offset so that it lands on the object
    pixel grid. The patch bounds are *not* clipped to the canvas: a scan position
    whose footprint falls outside the object array yields out-of-range slices, and
    assigning through them raises rather than silently truncating.
    """
    object_geometry = product.object_.get_geometry()
    probe_geometry = product.probes.get_geometry()

    for scan_point, probe in product.iter_position_probes():
        object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
        bounds = probe_geometry.resolve_patch_bounds(object_point.x_px, object_point.y_px)

        shifted_modes = fourier_shift_2d(probe.get_array(), dx=bounds.dx, dy=bounds.dy)
        patch = numpy.sum(numpy.abs(shifted_modes) ** 2, axis=0)

        yield scan_point, bounds, patch


@dataclass(frozen=True)
class IlluminationMap:
    """Per-object-pixel photon count plus the metadata needed to derive fluence,
    dose, and intensity quantities."""

    photon_number: RealArrayType
    photon_flux_per_s: float
    photon_energy_J: float  # noqa: N815
    exposure_time_s: float
    mass_attenuation_m2_kg: float
    pixel_geometry: PixelGeometry
    center: ObjectCenter

    @property
    def photon_fluence_1_m2(self) -> RealArrayType:
        return self.photon_number / self.pixel_geometry.get_area_m2()

    @property
    def photon_fluence_rate_per_s_m2(self) -> RealArrayType:
        return self.photon_fluence_1_m2 / self.exposure_time_s

    @property
    def energy_fluence_J_m2(self) -> RealArrayType:  # noqa: N802
        return self.photon_fluence_1_m2 * self.photon_energy_J

    @property
    def energy_fluence_rate_W_m2(self) -> RealArrayType:  # noqa: N802
        return self.photon_fluence_rate_per_s_m2 * self.photon_energy_J

    @property
    def dose_Gy(self) -> RealArrayType:  # noqa: N802
        return self.energy_fluence_J_m2 * self.mass_attenuation_m2_kg

    @property
    def dose_rate_Gy_s(self) -> RealArrayType:  # noqa: N802
        return self.energy_fluence_rate_W_m2 * self.mass_attenuation_m2_kg

    @property
    def intensity_W_m2(self) -> RealArrayType:  # noqa: N802
        return self.energy_fluence_rate_W_m2

    def save_npz(self, file_path: Path) -> None:
        numpy.savez_compressed(
            file_path,
            allow_pickle=False,
            photon_number=self.photon_number,
            photon_fluence_1_m2=self.photon_fluence_1_m2,
            photon_fluence_rate_per_s_m2=self.photon_fluence_rate_per_s_m2,
            energy_fluence_J_m2=self.energy_fluence_J_m2,
            energy_fluence_rate_W_m2=self.energy_fluence_rate_W_m2,
            dose_Gy=self.dose_Gy,
            dose_rate_Gy_s=self.dose_rate_Gy_s,
            pixel_height_m=self.pixel_geometry.height_m,
            pixel_width_m=self.pixel_geometry.width_m,
            center_x_m=self.center.x_m,
            center_y_m=self.center.y_m,
        )


def compute_illumination_map(
    product: Product,
    *,
    probe_photon_counts_by_index: Mapping[int, float] | None = None,
) -> IlluminationMap:
    """Build a per-object-pixel photon-count canvas by summing the subpixel-shifted
    probe intensities at every scan position, packaged with the metadata needed to
    derive fluence, dose, and intensity quantities.

    Pass ``probe_photon_counts_by_index`` (per-scan-index photon counts) to weight
    each scan-point contribution by its true exposure. Weights are normalized to
    the mean of the provided counts so the total photon budget across the canvas
    matches the uniform-weight case; a scan index missing from the mapping gets
    the unit weight. Without a mapping the accumulation is uniform, matching the
    prior behavior.
    """
    object_geometry = product.object_.get_geometry()
    canvas = numpy.zeros((object_geometry.height_px, object_geometry.width_px))

    mean_counts: float | None = None
    if probe_photon_counts_by_index:
        mean_counts = float(numpy.mean(list(probe_photon_counts_by_index.values())))

    for scan_point, bounds, patch in _iter_probe_patches(product):
        if mean_counts is not None and mean_counts > 0.0:
            assert probe_photon_counts_by_index is not None
            counts = probe_photon_counts_by_index.get(scan_point.index, mean_counts)
            patch = patch * (counts / mean_counts)

        canvas[bounds.y_slice, bounds.x_slice] += patch

    exposure_time_s = product.metadata.exposure_time_s
    # The flux has one definition, in compute_product_geometry. The detector geometry
    # it also derives is irrelevant here, so only the distance is supplied.
    product_geometry = compute_product_geometry(
        probe_energy_eV=product.metadata.probe_energy_eV,
        probe_photon_count=product.metadata.probe_photon_count,
        exposure_time_s=exposure_time_s,
        detector_distance_m=product.metadata.detector_distance_m,
    )

    return IlluminationMap(
        photon_number=canvas,
        photon_flux_per_s=product_geometry.probe_photon_flux_per_s,
        photon_energy_J=product.metadata.probe_energy_J,
        exposure_time_s=exposure_time_s,
        mass_attenuation_m2_kg=product.metadata.mass_attenuation_m2_kg,
        pixel_geometry=object_geometry.get_pixel_geometry(),
        center=object_geometry.get_center(),
    )


def _participation_ratio(values: RealArrayType) -> float:
    """Effective support size, in pixels, of a non-negative array.

    The participation ratio ``(sum f)^2 / sum f^2`` is the threshold-free
    counterpart of "count the pixels above some cutoff": for a top-hat it returns
    the support area exactly, for a Gaussian ``exp(-r^2 / 2 sigma^2)`` it returns
    ``4 pi sigma^2``, and it is invariant under a global rescale of ``f``. Zero for
    an all-zero array.
    """
    total = numpy.sum(values)
    total_squared = numpy.sum(numpy.square(values))

    if total_squared <= 0.0:
        return 0.0

    return float(total * total / total_squared)


def _compute_pairwise_overlap(
    product: Product,
    footprint_power: RealArrayType,
    *,
    neighbor_radius_m: float,
) -> RealArrayType:
    """Largest normalized footprint-overlap integral between each position and its neighbors.

    ``footprint_power`` holds ``integral(P_j^2)`` for every position, so the
    normalization is exact per position; the shared numerator ``integral(P_j P_k)``
    is read off the representative probe's autocorrelation at the pair displacement.

    ``neighbor_radius_m`` bounds which positions count as neighbors. Passing the
    effective probe diameter is exact for a top-hat -- two equal-area discs overlap if
    and only if their centers are closer than one diameter -- and truncates only the
    far tail of a smoothly decaying probe, where the overlap integral is negligible.
    """
    num_positions = len(product.probe_positions)
    overlap = numpy.zeros(num_positions)

    if num_positions < 2 or not neighbor_radius_m > 0.0:
        return overlap

    coordinates_m = numpy.array(
        [[point.x_m, point.y_m] for point in product.probe_positions], dtype=float
    )
    pairs = KDTree(coordinates_m).query_pairs(r=neighbor_radius_m, output_type='ndarray')

    if pairs.size == 0:
        return overlap

    probe = product.probes.get_probe_no_opr()
    pixel_geometry = probe.get_pixel_geometry()
    footprint = probe.get_intensity()
    # Linear (zero-padded) autocorrelation, so lag (0, 0) lands at (height_px - 1, width_px - 1).
    correlation = scipy.signal.correlate(footprint, footprint, mode='full', method='fft')

    displacement_m = coordinates_m[pairs[:, 1]] - coordinates_m[pairs[:, 0]]
    column = displacement_m[:, 0] / pixel_geometry.width_m + footprint.shape[-1] - 1
    row = displacement_m[:, 1] / pixel_geometry.height_m + footprint.shape[-2] - 1

    # 'grid-constant' pads with cval and still interpolates, so a displacement partly off
    # the autocorrelation support decays to zero instead of snapping to it ('constant').
    numerator = scipy.ndimage.map_coordinates(
        correlation, numpy.stack([row, column]), order=1, mode='grid-constant', cval=0.0
    )
    denominator = numpy.sqrt(footprint_power[pairs[:, 0]] * footprint_power[pairs[:, 1]])
    values = numpy.divide(
        numerator, denominator, out=numpy.zeros(numerator.shape), where=denominator > 0.0
    )

    numpy.maximum.at(overlap, pairs[:, 0], values)
    numpy.maximum.at(overlap, pairs[:, 1], values)
    return overlap


@dataclass(frozen=True)
class ProbeOverlapMetrics:
    """Threshold-free probe-overlap metrics for a scan, generalizing the linear overlap
    ratio to irregular position layouts and structured, multi-modal probes."""

    redundancy: RealArrayType
    """Per-object-pixel effective probe count ``S^2 / Q``, where ``S`` sums the probe
    footprints and ``Q`` sums their squares. Exactly ``k`` where ``k`` probes contribute
    equally; a soft count under unequal contributions; NaN where nothing is illuminated."""

    pairwise_overlap_by_position: RealArrayType
    """Shape ``(num_positions,)``. For each position, the largest normalized overlap
    integral ``integral(P_j P_k) / sqrt(integral(P_j^2) integral(P_k^2))`` against its
    spatial neighbors. Equals the circle-circle areal overlap ratio for top-hat probes.
    An entry near zero marks a position no neighbor constrains."""

    effective_probe_area_m2: float
    """Mean over positions of the participation-ratio area of a single probe footprint."""

    effective_covered_area_m2: float
    """Participation-ratio area of the accumulated illumination canvas."""

    num_positions: int
    """Number of scan positions the metrics were computed over."""

    pixel_geometry: PixelGeometry
    """Object pixel geometry, matching :attr:`redundancy`."""

    center: ObjectCenter
    """Object center, matching :attr:`redundancy`."""

    @property
    def effective_probe_diameter_m(self) -> float:
        """Diameter of the disc whose area is :attr:`effective_probe_area_m2`."""
        return 2.0 * math.sqrt(self.effective_probe_area_m2 / math.pi)

    @property
    def effective_step_size_m(self) -> float:
        """Side of the square whose area is the covered area per scan position."""
        return math.sqrt(self.effective_covered_area_m2 / self.num_positions)

    @property
    def areal_redundancy(self) -> float:
        """Total probe footprint area divided by the area actually covered: the number of
        probes illuminating a typical point. Only meaningful for many scan positions."""
        if not self.effective_covered_area_m2 > 0.0:
            return float('nan')

        return self.num_positions * self.effective_probe_area_m2 / self.effective_covered_area_m2

    @property
    def equivalent_linear_overlap(self) -> float:
        """The familiar overlap ratio ``1 - step / diameter``, evaluated on the effective
        step and diameter, and so equal to ``1 - sqrt(pi / (4 * areal_redundancy))``.

        Reproduces the textbook value for a square raster of circular probes and extends
        it to arbitrary layouts. Values at or below zero mean the footprints do not tile
        the illuminated region; they are reported as computed rather than clamped.
        """
        diameter_m = self.effective_probe_diameter_m

        if not diameter_m > 0.0:
            return float('nan')

        return 1.0 - self.effective_step_size_m / diameter_m

    @property
    def mean_pairwise_overlap(self) -> float:
        if self.num_positions < 2:
            return float('nan')

        return float(numpy.mean(self.pairwise_overlap_by_position))

    @property
    def median_pairwise_overlap(self) -> float:
        if self.num_positions < 2:
            return float('nan')

        return float(numpy.median(self.pairwise_overlap_by_position))

    @property
    def minimum_pairwise_overlap(self) -> float:
        if self.num_positions < 2:
            return float('nan')

        return float(numpy.min(self.pairwise_overlap_by_position))

    @property
    def maximum_pairwise_overlap(self) -> float:
        if self.num_positions < 2:
            return float('nan')

        return float(numpy.max(self.pairwise_overlap_by_position))

    def save_npz(self, file_path: Path) -> None:
        numpy.savez_compressed(
            file_path,
            allow_pickle=False,
            redundancy=self.redundancy,
            pairwise_overlap_by_position=self.pairwise_overlap_by_position,
            effective_probe_area_m2=self.effective_probe_area_m2,
            effective_covered_area_m2=self.effective_covered_area_m2,
            effective_probe_diameter_m=self.effective_probe_diameter_m,
            effective_step_size_m=self.effective_step_size_m,
            areal_redundancy=self.areal_redundancy,
            equivalent_linear_overlap=self.equivalent_linear_overlap,
            num_positions=self.num_positions,
            pixel_height_m=self.pixel_geometry.height_m,
            pixel_width_m=self.pixel_geometry.width_m,
            center_x_m=self.center.x_m,
            center_y_m=self.center.y_m,
        )


def compute_probe_overlap(
    product: Product, *, roundoff_floor: float = 1.0e-12
) -> ProbeOverlapMetrics:
    """Measure how densely a scan's probe footprints tile the illuminated region.

    The classic linear overlap ratio ``1 - step / diameter`` presumes a regular raster
    of identical circular probes. This function measures the same thing directly from
    the footprints and positions the product actually holds, so it stays meaningful for
    spiral and jittered scans and for structured, multi-modal probes. Every quantity is
    threshold-free: areas are participation ratios rather than counts of pixels above a
    cutoff, so no tunable changes the answer, and all of them are invariant under a
    global rescale of the probe intensity.

    Two complementary views come back. :attr:`ProbeOverlapMetrics.areal_redundancy` and
    :attr:`ProbeOverlapMetrics.equivalent_linear_overlap` summarize the scan as a whole
    -- is it dense enough -- while
    :attr:`ProbeOverlapMetrics.pairwise_overlap_by_position` is per position and exposes
    weakly constrained points that a global average hides.

    The pairwise numerator is sampled from the autocorrelation of the representative
    probe returned by :meth:`ProbeSequence.get_probe_no_opr`, which keeps the cost at
    ``O(N log N)`` instead of ``O(N^2)``; orthogonal-probe-relaxation variation between
    positions therefore affects the exact per-position normalization but not the shared
    numerator. Treat the pairwise numbers as a scan-geometry diagnostic rather than a
    photometric one.

    ``roundoff_floor`` is the fraction of the peak illumination below which a pixel is
    treated as unilluminated and its redundancy reported as NaN. Subpixel Fourier shifts
    leave round-off of order 1e-16 relative to the peak on pixels no footprint reaches,
    and without a floor that noise fills the empty canvas with meaningless redundancy
    values. The default sits twelve orders of magnitude below the peak -- far under any
    physical signal, far over double-precision round-off -- so it suppresses the noise
    without acting as a coverage threshold. Raise it only to trim a genuinely faint halo
    from the map; it affects nothing but :attr:`ProbeOverlapMetrics.redundancy`.

    Raises ``ValueError`` if the product has no probe positions, or if the probe
    sequence carries no pixel geometry.
    """
    num_positions = len(product.probe_positions)

    if num_positions < 1:
        raise ValueError('Cannot compute probe overlap without probe positions!')

    object_geometry = product.object_.get_geometry()
    canvas = numpy.zeros((object_geometry.height_px, object_geometry.width_px))
    canvas_squared = numpy.zeros_like(canvas)
    footprint_area_px = numpy.zeros(num_positions)
    footprint_power = numpy.zeros(num_positions)

    for index, (_, bounds, patch) in enumerate(_iter_probe_patches(product)):
        patch_squared = numpy.square(patch)
        canvas[bounds.y_slice, bounds.x_slice] += patch
        canvas_squared[bounds.y_slice, bounds.x_slice] += patch_squared
        footprint_area_px[index] = _participation_ratio(patch)
        footprint_power[index] = numpy.sum(patch_squared)

    pixel_geometry = object_geometry.get_pixel_geometry()
    pixel_area_m2 = pixel_geometry.get_area_m2()
    effective_probe_area_m2 = float(numpy.mean(footprint_area_px)) * pixel_area_m2

    illuminated = canvas > numpy.max(canvas) * roundoff_floor
    redundancy = numpy.divide(
        numpy.square(canvas),
        canvas_squared,
        out=numpy.full(canvas.shape, numpy.nan),
        where=illuminated,
    )

    return ProbeOverlapMetrics(
        redundancy=redundancy,
        pairwise_overlap_by_position=_compute_pairwise_overlap(
            product,
            footprint_power,
            neighbor_radius_m=2.0 * math.sqrt(effective_probe_area_m2 / math.pi),
        ),
        effective_probe_area_m2=effective_probe_area_m2,
        effective_covered_area_m2=_participation_ratio(canvas) * pixel_area_m2,
        num_positions=num_positions,
        pixel_geometry=pixel_geometry,
        center=object_geometry.get_center(),
    )
