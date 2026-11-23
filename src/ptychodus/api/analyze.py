"""Metrics and comparison reports for a product's probe, object and probe positions.

Each report bundles everything one view of one product, or of a reference/test pair,
says about its quality. A report's ``scalars`` field holds only Python numbers, booleans
and nested dataclasses of them, in SI units, so ``dataclasses.asdict(report.scalars)``
serializes as JSON; the arrays and domain objects the scalars were derived from sit
beside it.
"""

from __future__ import annotations
from collections.abc import Sequence
from dataclasses import dataclass, replace
import math

import numpy
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .affine import AffineTransform
from .illumination import ProbeOverlapMetrics, compute_probe_overlap
from .interpolate import match_at_indexes
from .metrics import (
    FourierRingCorrelation,
    ObjectComparison,
    PixelwiseMetrics,
    PowerSpectralDensity,
    ProbeComparison,
    ScoringRegion,
    SpectralSNRConvention,
    compute_fourier_ring_correlation,
    compute_object_comparison,
    compute_phase_pixelwise_metrics,
    compute_pixelwise_metrics,
    compute_power_spectral_density,
    compute_probe_comparison,
    compute_product_scoring_region,
    estimate_reconstruction_ambiguities,
)
from .geometry import PixelGeometry
from .object import Object, resample_object_to_pixel_size
from .preprocess.noise import subtract_noise_floor
from .probe import (
    Probe,
    ProbeEnsemble,
    ProbeEntropyMetrics,
    ProbeGeometry,
    ProbeSizeMetrics,
    compute_phase_deviation_rad,
    compute_probe_mode_spectrum,
    estimate_probe_entropy,
    estimate_probe_size,
    orthogonalize_probe_modes,
    resample_probe_ensemble,
)
from .probe_positions import ProbePositionSequence
from .product import Product, ProductMetadata
from .reconstruct import ReconstructionAmbiguities
from .scattering.frame import compute_exit_frame_transform
from .scattering.geometry import find_scattering_issues
from .typing import BooleanArrayType, ComplexArrayType, IntegerArrayType, RealArrayType

__all__ = [
    'ObjectComparisonReport',
    'ObjectComparisonScalars',
    'ObjectLayerScalars',
    'ObjectMetricsReport',
    'ObjectMetricsScalars',
    'ProbeComparisonReport',
    'ProbeComparisonScalars',
    'ProbeMetricsReport',
    'ProbeMetricsScalars',
    'ProbeOverlapScalars',
    'ProbePositionsComparisonReport',
    'ProbePositionsComparisonScalars',
    'ProbePositionsMetricsReport',
    'ProbePositionsMetricsScalars',
    'ProbeProfile',
    'ProbeProfileScalars',
    'compute_object_comparison_report',
    'compute_object_metrics_report',
    'compute_probe_comparison_report',
    'compute_probe_metrics_report',
    'compute_probe_positions_comparison_report',
    'compute_probe_positions_metrics_report',
]


def _nan_statistics(values: RealArrayType) -> tuple[float, float, float, float]:
    """Mean, median, minimum and maximum of *values*, all NaN when it is empty."""
    if values.size == 0:
        nan = float('nan')
        return nan, nan, nan, nan

    return (
        float(numpy.mean(values)),
        float(numpy.median(values)),
        float(numpy.min(values)),
        float(numpy.max(values)),
    )


# --------------------------------------------------------------------------- probe


@dataclass(frozen=True)
class ProbeProfileScalars:
    """What one probe is like, independent of how its modes were split."""

    num_modes: int
    size: ProbeSizeMetrics | None
    """Size of the incoherent intensity; ``None`` when it cannot be estimated."""
    entropy: ProbeEntropyMetrics
    """Real-space and spectral entropy of the incoherent intensity."""
    purity: float
    von_neumann_entropy: float
    """In nats."""
    effective_mode_count: float


@dataclass(frozen=True)
class ProbeProfile:
    """A probe, its orthogonalized modes, and the figures derived from them."""

    scalars: ProbeProfileScalars
    probe: Probe
    """:meth:`ProbeEnsemble.get_mean_probe` of the product."""
    orthogonal_probe: Probe
    """``probe``'s modes orthogonalized and sorted by descending power."""
    relative_power: RealArrayType
    """Fraction of the power in each orthogonal mode."""


def _compute_probe_profile(probe: Probe) -> ProbeProfile:
    spectrum = compute_probe_mode_spectrum(probe)

    try:
        size: ProbeSizeMetrics | None = estimate_probe_size(
            probe.get_intensity(), probe.get_pixel_geometry()
        )
    except ValueError:
        size = None

    return ProbeProfile(
        scalars=ProbeProfileScalars(
            num_modes=int(probe.num_incoherent_modes),
            size=size,
            entropy=estimate_probe_entropy(probe),
            purity=spectrum.purity,
            von_neumann_entropy=spectrum.von_neumann_entropy,
            effective_mode_count=spectrum.effective_mode_count,
        ),
        probe=probe,
        orthogonal_probe=orthogonalize_probe_modes(probe),
        relative_power=spectrum.relative_power,
    )


@dataclass(frozen=True)
class ProbeOverlapScalars:
    """The scan-wide figures of :class:`~ptychodus.api.illumination.ProbeOverlapMetrics`."""

    areal_redundancy: float
    equivalent_linear_overlap: float
    effective_probe_diameter_m: float
    effective_step_size_m: float
    mean_pairwise_overlap: float
    median_pairwise_overlap: float
    minimum_pairwise_overlap: float
    maximum_pairwise_overlap: float
    num_positions: int

    @classmethod
    def from_metrics(cls, metrics: ProbeOverlapMetrics) -> ProbeOverlapScalars:
        return cls(
            areal_redundancy=float(metrics.areal_redundancy),
            equivalent_linear_overlap=float(metrics.equivalent_linear_overlap),
            effective_probe_diameter_m=float(metrics.effective_probe_diameter_m),
            effective_step_size_m=float(metrics.effective_step_size_m),
            mean_pairwise_overlap=float(metrics.mean_pairwise_overlap),
            median_pairwise_overlap=float(metrics.median_pairwise_overlap),
            minimum_pairwise_overlap=float(metrics.minimum_pairwise_overlap),
            maximum_pairwise_overlap=float(metrics.maximum_pairwise_overlap),
            num_positions=int(metrics.num_positions),
        )


@dataclass(frozen=True)
class ProbeMetricsScalars:
    profile: ProbeProfileScalars
    overlap: ProbeOverlapScalars | None
    """``None`` when the product has no probe positions."""
    has_opr_weights: bool
    """Whether ``profile`` describes the field at the mean OPR weights."""


@dataclass(frozen=True)
class ProbeMetricsReport:
    scalars: ProbeMetricsScalars
    profile: ProbeProfile
    overlap: ProbeOverlapMetrics | None


def compute_probe_metrics_report(product: Product) -> ProbeMetricsReport:
    """Describe ``product``'s probe: its size, entropy, mode spectrum and scan overlap.

    The probe is the field at the mean OPR weights, :meth:`ProbeEnsemble.get_mean_probe`.

    Raises:
        ValueError: If the probe carries no power.
    """
    profile = _compute_probe_profile(product.probes.get_mean_probe())
    overlap = compute_probe_overlap(product) if len(product.probe_positions) > 0 else None

    return ProbeMetricsReport(
        scalars=ProbeMetricsScalars(
            profile=profile.scalars,
            overlap=None if overlap is None else ProbeOverlapScalars.from_metrics(overlap),
            has_opr_weights=product.probes.get_opr_weights_or_none() is not None,
        ),
        profile=profile,
        overlap=overlap,
    )


def _center_crop_or_pad(array: ComplexArrayType, height_px: int, width_px: int) -> ComplexArrayType:
    """Cut or zero-pad the trailing two axes of *array* to ``(height_px, width_px)`` about its beam axis.

    The beam axis is pixel ``N // 2`` of each axis, so it lands on pixel ``N // 2`` of
    the result whatever the parity of either size.
    """
    source_height_px, source_width_px = array.shape[-2:]
    kept_height_px = min(source_height_px, height_px)
    kept_width_px = min(source_width_px, width_px)
    source_row = source_height_px // 2 - kept_height_px // 2
    source_column = source_width_px // 2 - kept_width_px // 2
    target_row = height_px // 2 - kept_height_px // 2
    target_column = width_px // 2 - kept_width_px // 2
    result = numpy.zeros((*array.shape[:-2], height_px, width_px), dtype=array.dtype)
    result[
        ...,
        target_row : target_row + kept_height_px,
        target_column : target_column + kept_width_px,
    ] = array[
        ...,
        source_row : source_row + kept_height_px,
        source_column : source_column + kept_width_px,
    ]
    return result


def _resample_onto_reference(
    reference: Product, test: Product, *, resample_test: bool
) -> tuple[Product, PixelGeometry | None, ProbeGeometry | None]:
    """Re-express ``test``'s object and probes on ``reference``'s grids where they differ.

    Returns the test product and the object pixel geometry and probe geometry it was
    resampled from, each ``None`` when that part already matched.
    """
    object_ = test.object_
    object_resampled_from: PixelGeometry | None = None
    reference_pixel_geometry = reference.object_.get_pixel_geometry()
    test_pixel_geometry = test.object_.get_pixel_geometry()

    if test_pixel_geometry != reference_pixel_geometry:
        if not resample_test:
            raise ValueError(
                f'Object pixel geometry mismatch: reference {reference_pixel_geometry}'
                f' vs test {test_pixel_geometry}!'
            )

        object_ = resample_object_to_pixel_size(test.object_, reference_pixel_geometry)

        # Pitches within the resampler's tolerance come back untouched; they name the
        # same sampling, so the reference's geometry is stamped on as it stands.
        if object_.get_pixel_geometry() != reference_pixel_geometry:
            object_ = Object(
                array=object_.get_array(),
                pixel_geometry=reference_pixel_geometry,
                center=object_.get_center(),
                layer_spacing_m=list(object_.layer_spacing_m),
            )

        object_resampled_from = test_pixel_geometry

    probes = test.probes
    probe_resampled_from: ProbeGeometry | None = None
    reference_probe_geometry = reference.probes.get_geometry()
    test_probe_geometry = test.probes.get_geometry()

    if test_probe_geometry != reference_probe_geometry:
        if not resample_test:
            raise ValueError(
                f'Probe geometry mismatch: reference {reference_probe_geometry}'
                f' vs test {test_probe_geometry}!'
            )

        # Resampling matches the pitch; a frame of another size at the same pitch is
        # then cropped or padded about its center, with zeros where no light fell.
        probes = resample_probe_ensemble(test.probes, reference_probe_geometry)
        probes = ProbeEnsemble(
            _center_crop_or_pad(
                probes.get_array(),
                reference_probe_geometry.height_px,
                reference_probe_geometry.width_px,
            ),
            probes.get_opr_weights_or_none(),
            reference_probe_geometry.get_pixel_geometry(),
        )
        probe_resampled_from = test_probe_geometry

    return (
        replace(test, object_=object_, probes=probes),
        object_resampled_from,
        probe_resampled_from,
    )


def _pair_scoring_region(
    reference: Product, test: Product, *, erosion_divisor: int
) -> ScoringRegion:
    """The reference's scoring region on the grid both objects are center-cropped to."""
    common_shape = (
        min(reference.object_.height_px, test.object_.height_px),
        min(reference.object_.width_px, test.object_.width_px),
    )
    return compute_product_scoring_region(reference, common_shape, erosion_divisor=erosion_divisor)


@dataclass(frozen=True)
class ProbeComparisonScalars:
    normalized_fidelity: float
    normalized_rms_error: float
    probe_shift_x_px: float
    probe_shift_y_px: float
    object_shift_x_px: float
    object_shift_y_px: float
    reference: ProbeProfileScalars
    test: ProbeProfileScalars
    amplitude: PixelwiseMetrics
    """Agreement of the square roots of the incoherent intensities."""
    phase: PixelwiseMetrics
    """Agreement of the phase-aligned top modes over the reference's noise-floor support."""
    test_object_resampled_from: PixelGeometry | None
    test_probe_resampled_from: ProbeGeometry | None


@dataclass(frozen=True)
class ProbeComparisonReport:
    scalars: ProbeComparisonScalars
    comparison: ProbeComparison
    reference: ProbeProfile
    test: ProbeProfile
    reference_top_mode: ComplexArrayType
    """The reference probe's most powerful orthogonal mode."""
    test_top_mode: ComplexArrayType
    """The registered test probe's most powerful orthogonal mode, phase-aligned to the reference's."""
    phase_support: BooleanArrayType
    """Pixels the phase metrics are scored over."""


def compute_probe_comparison_report(
    reference: Product,
    test: Product,
    *,
    erosion_divisor: int = 8,
    mad_threshold: float = 4.5,
    resample_test: bool = True,
) -> ProbeComparisonReport:
    """Compare ``test``'s probe with ``reference``'s.

    The mixed-state fidelity and RMS error come from :func:`compute_probe_comparison`,
    with the ambiguity estimate weighted by the reference's scoring region. Pixelwise
    amplitude metrics compare the square roots of the incoherent intensities, which do
    not depend on how either probe splits its modes. Pixelwise phase metrics compare
    the most powerful orthogonal mode of each, after giving the test mode the global
    phase ``angle(sum(reference * conj(test)))`` that minimizes their difference, over
    the pixels :func:`~ptychodus.api.preprocess.noise.subtract_noise_floor` keeps in
    the reference mode's intensity.

    Args:
        reference: The reconstruction treated as ground truth.
        test: The reconstruction being evaluated.
        erosion_divisor: See :func:`~ptychodus.api.metrics.compute_product_scoring_region`.
        mad_threshold: Significance threshold of the phase support, in units of the
            median absolute deviation above the background; see
            :func:`~ptychodus.api.probe.estimate_probe_size`.
        resample_test: Resample the test object and probes onto the reference's grids
            where they differ; otherwise a mismatch raises.

    Raises:
        ValueError: On a geometry mismatch with ``resample_test`` off, when the
            reference yields no scoring region, or when a probe carries no power.
    """
    test, object_resampled_from, probe_resampled_from = _resample_onto_reference(
        reference, test, resample_test=resample_test
    )
    region = _pair_scoring_region(reference, test, erosion_divisor=erosion_divisor)
    comparison = compute_probe_comparison(reference, test, weights=region.get_array_weights())

    reference_top_mode = orthogonalize_probe_modes(comparison.reference_probe).get_incoherent_mode(
        0
    )
    test_top_mode = orthogonalize_probe_modes(comparison.test_probe).get_incoherent_mode(0)
    test_top_mode = test_top_mode * numpy.exp(
        1j * numpy.angle(numpy.sum(reference_top_mode * numpy.conj(test_top_mode)))
    )

    cleaned, _ = subtract_noise_floor(
        numpy.square(numpy.absolute(reference_top_mode)),
        mad_threshold=mad_threshold,
        fall_back_to_border=True,
    )
    phase_support = cleaned > 0.0

    reference_profile = _compute_probe_profile(reference.probes.get_mean_probe())
    test_profile = _compute_probe_profile(test.probes.get_mean_probe())

    return ProbeComparisonReport(
        scalars=ProbeComparisonScalars(
            normalized_fidelity=comparison.normalized_fidelity,
            normalized_rms_error=comparison.normalized_rms_error,
            probe_shift_x_px=comparison.probe_shift_x_px,
            probe_shift_y_px=comparison.probe_shift_y_px,
            object_shift_x_px=comparison.object_shift_x_px,
            object_shift_y_px=comparison.object_shift_y_px,
            reference=reference_profile.scalars,
            test=test_profile.scalars,
            amplitude=compute_pixelwise_metrics(
                numpy.sqrt(comparison.reference_probe.get_intensity()),
                numpy.sqrt(comparison.test_probe.get_intensity()),
            ),
            phase=compute_phase_pixelwise_metrics(
                reference_top_mode, test_top_mode, mask=phase_support
            ),
            test_object_resampled_from=object_resampled_from,
            test_probe_resampled_from=probe_resampled_from,
        ),
        comparison=comparison,
        reference=reference_profile,
        test=test_profile,
        reference_top_mode=reference_top_mode,
        test_top_mode=test_top_mode,
        phase_support=phase_support,
    )


# -------------------------------------------------------------------------- object


@dataclass(frozen=True)
class ObjectLayerScalars:
    """Amplitude and phase statistics of one object layer, or of the projection, over the region."""

    amplitude_mean: float
    amplitude_std: float
    amplitude_minimum: float
    amplitude_maximum: float
    phase_mean_rad: float
    """Intensity-weighted circular mean."""
    phase_deviation_rad: float
    """Intensity-weighted circular standard deviation."""
    phase_peak_to_valley_rad: float
    """Span of the phase about its circular mean."""


def _compute_layer_scalars(values: ComplexArrayType) -> ObjectLayerScalars:
    amplitude = numpy.absolute(values)
    resultant = numpy.sum(amplitude * values)
    phase_mean_rad = float(numpy.angle(resultant)) if resultant != 0 else 0.0
    centered_phase = numpy.angle(values * numpy.exp(-1j * phase_mean_rad))

    return ObjectLayerScalars(
        amplitude_mean=float(numpy.mean(amplitude)),
        amplitude_std=float(numpy.std(amplitude)),
        amplitude_minimum=float(numpy.min(amplitude)),
        amplitude_maximum=float(numpy.max(amplitude)),
        phase_mean_rad=phase_mean_rad,
        phase_deviation_rad=float(compute_phase_deviation_rad(values)),
        phase_peak_to_valley_rad=float(numpy.ptp(centered_phase)),
    )


def _compute_layer_spectrum(
    values: ComplexArrayType, region: ScoringRegion, pixel_geometry: PixelGeometry
) -> PowerSpectralDensity:
    return compute_power_spectral_density(
        values,
        pixel_width_m=pixel_geometry.width_m,
        pixel_height_m=pixel_geometry.height_m,
        window=region.weights,
    )


@dataclass(frozen=True)
class ObjectMetricsScalars:
    ambiguities: ReconstructionAmbiguities
    """Fitted to the projection and applied to layer 0; the scale is fixed at one."""
    region_height_px: int
    region_width_px: int
    region_area_fraction: float
    projected: ObjectLayerScalars
    layers: tuple[ObjectLayerScalars, ...]
    layer_depths_m: tuple[float, ...]
    """Depth of each layer below layer 0."""


@dataclass(frozen=True)
class ObjectMetricsReport:
    scalars: ObjectMetricsScalars
    scoring_region: ScoringRegion
    standardized_object: Object
    """The object with the ambiguities removed."""
    projected_spectrum: PowerSpectralDensity
    layer_spectra: Sequence[PowerSpectralDensity]


def compute_object_metrics_report(
    product: Product, *, erosion_divisor: int = 8
) -> ObjectMetricsReport:
    """Describe ``product``'s object over the region its scan illuminated.

    The phase offset and ramp are estimated from the object itself, weighted by the
    region, and removed before the statistics and spectra are taken, so they describe
    the sample rather than an arbitrary tilt.

    Args:
        product: The product whose object is described.
        erosion_divisor: See :func:`~ptychodus.api.metrics.compute_product_scoring_region`.

    Raises:
        ValueError: If the product's illumination yields no scoring region.
    """
    obj = product.object_
    region = compute_product_scoring_region(
        product, (obj.height_px, obj.width_px), erosion_divisor=erosion_divisor
    )
    ambiguities = estimate_reconstruction_ambiguities(product, weights=region.get_array_weights())
    standardized = ambiguities.standardize_product(product).object_
    pixel_geometry = standardized.get_pixel_geometry()

    projected = region.crop(standardized.get_layers_flattened())
    layers = [region.crop(standardized.get_layer(number)) for number in range(obj.num_layers)]
    depths = numpy.concatenate(([0.0], numpy.cumsum(numpy.asarray(obj.layer_spacing_m, float))))

    return ObjectMetricsReport(
        scalars=ObjectMetricsScalars(
            ambiguities=ambiguities,
            region_height_px=int(region.height_px),
            region_width_px=int(region.width_px),
            region_area_fraction=float(
                region.height_px * region.width_px / (obj.height_px * obj.width_px)
            ),
            projected=_compute_layer_scalars(projected),
            layers=tuple(_compute_layer_scalars(layer) for layer in layers),
            layer_depths_m=tuple(float(depth) for depth in depths),
        ),
        scoring_region=region,
        standardized_object=standardized,
        projected_spectrum=_compute_layer_spectrum(projected, region, pixel_geometry),
        layer_spectra=[_compute_layer_spectrum(layer, region, pixel_geometry) for layer in layers],
    )


@dataclass(frozen=True)
class ObjectComparisonScalars:
    resolution_m_at_snr: float
    """Finest resolution where the spectral signal-to-noise ratio stays above the threshold."""
    snr_threshold: float
    resolution_m_at_half_bit: float
    area_under_curve: float
    average_signal_to_noise_ratio: float
    shift_x_px: float
    shift_y_px: float
    ambiguities: ReconstructionAmbiguities
    """Removed from the test, fitted to the projections."""
    amplitude: PixelwiseMetrics
    phase: PixelwiseMetrics
    test_object_resampled_from: PixelGeometry | None
    test_probe_resampled_from: ProbeGeometry | None


@dataclass(frozen=True)
class ObjectComparisonReport:
    scalars: ObjectComparisonScalars
    comparison: ObjectComparison
    """The projected objects, center-cropped, with the test registered and standardized."""
    scoring_region: ScoringRegion
    fourier_ring_correlation: FourierRingCorrelation
    reference_spectrum: PowerSpectralDensity
    test_spectrum: PowerSpectralDensity


def compute_object_comparison_report(
    reference: Product,
    test: Product,
    *,
    convention: SpectralSNRConvention = SpectralSNRConvention.HALF_SPLIT,
    snr_threshold: float = 2.0,
    erosion_divisor: int = 8,
    resample_test: bool = True,
) -> ObjectComparisonReport:
    """Compare ``test``'s projected object with ``reference``'s.

    The test is registered onto the reference and its ambiguities removed by
    :func:`~ptychodus.api.metrics.compute_object_comparison`, weighted by the
    reference's scoring region. Every metric is then taken over that region: the
    Fourier ring correlation, both power spectra and the pixelwise agreement of
    amplitude and phase.

    Args:
        reference: The reconstruction treated as ground truth.
        test: The reconstruction being evaluated.
        convention: Whether the two are halves of one scan or repeat scans.
        snr_threshold: Spectral signal-to-noise ratio defining ``resolution_m_at_snr``;
            2 is equivalent to FRC = 1/2 for a half split and to FRC = 2/3 for repeats.
        erosion_divisor: See :func:`~ptychodus.api.metrics.compute_product_scoring_region`.
        resample_test: Resample the test object and probes onto the reference's grids
            where they differ; otherwise a mismatch raises.

    Raises:
        ValueError: On a geometry mismatch with ``resample_test`` off, or when the
            reference yields no scoring region.
    """
    test, object_resampled_from, probe_resampled_from = _resample_onto_reference(
        reference, test, resample_test=resample_test
    )
    region = _pair_scoring_region(reference, test, erosion_divisor=erosion_divisor)
    comparison = compute_object_comparison(reference, test, weights=region.get_array_weights())
    pixel_geometry = comparison.pixel_geometry
    reference_values = region.crop(comparison.reference_complex)
    test_values = region.crop(comparison.test_complex)

    frc = compute_fourier_ring_correlation(
        reference_values,
        test_values,
        pixel_width_m=pixel_geometry.width_m,
        pixel_height_m=pixel_geometry.height_m,
        window=region.weights,
        convention=convention,
    )

    return ObjectComparisonReport(
        scalars=ObjectComparisonScalars(
            resolution_m_at_snr=float(
                frc.get_resolution_m_at_signal_to_noise_threshold(snr_threshold)
            ),
            snr_threshold=float(snr_threshold),
            resolution_m_at_half_bit=float(frc.get_resolution_m_at_bit_threshold(0.5)),
            area_under_curve=float(frc.get_area_under_curve()),
            average_signal_to_noise_ratio=float(frc.get_average_signal_to_noise_ratio()),
            shift_x_px=comparison.shift_x_px,
            shift_y_px=comparison.shift_y_px,
            ambiguities=comparison.ambiguities,
            amplitude=compute_pixelwise_metrics(
                numpy.absolute(reference_values), numpy.absolute(test_values)
            ),
            phase=compute_phase_pixelwise_metrics(reference_values, test_values),
            test_object_resampled_from=object_resampled_from,
            test_probe_resampled_from=probe_resampled_from,
        ),
        comparison=comparison,
        scoring_region=region,
        fourier_ring_correlation=frc,
        reference_spectrum=_compute_layer_spectrum(reference_values, region, pixel_geometry),
        test_spectrum=_compute_layer_spectrum(test_values, region, pixel_geometry),
    )


# ---------------------------------------------------------------------- positions


def _resolve_exit_transform(metadata: ProductMetadata) -> AffineTransform:
    """The product's exit-frame map, the identity where the frame is undefined, without warning."""
    geometry = metadata.scattering_geometry

    if geometry.is_forward or find_scattering_issues(geometry, far_field=metadata.far_field):
        return AffineTransform.create_identity()

    return compute_exit_frame_transform(geometry)


def _packed_xy_m(positions: ProbePositionSequence) -> RealArrayType:
    return numpy.column_stack((positions.get_coordinates_x_m(), positions.get_coordinates_y_m()))


def _index_gaps(unique_indexes: IntegerArrayType) -> IntegerArrayType:
    """``(G, 2)`` first and last index of each run missing between the sorted *unique_indexes*."""
    if unique_indexes.size < 2:
        return numpy.empty((0, 2), dtype=numpy.intp)

    jumps = numpy.flatnonzero(numpy.diff(unique_indexes) > 1)
    return numpy.column_stack((unique_indexes[jumps] + 1, unique_indexes[jumps + 1] - 1))


@dataclass(frozen=True)
class ProbePositionsMetricsScalars:
    num_positions: int
    num_distinct_indexes: int
    index_minimum: int
    index_maximum: int
    num_duplicated_indexes: int
    """Distinct indexes recorded more than once; their positions are averaged."""
    num_missing_indexes: int
    """Indexes absent between the minimum and maximum."""
    num_index_gaps: int
    width_m: float
    height_m: float
    center_x_m: float
    center_y_m: float
    convex_hull_area_m2: float
    """Zero for fewer than three distinct positions or a one-dimensional scan."""
    density_per_m2: float
    nearest_neighbor_mean_m: float
    nearest_neighbor_median_m: float
    nearest_neighbor_minimum_m: float
    nearest_neighbor_maximum_m: float
    is_exit_frame: bool
    i0_is_calibrated: bool
    """Whether I0 is in incident photons rather than the instrument's counts."""
    i0_mean: float
    i0_std: float
    i0_minimum: float
    i0_maximum: float


@dataclass(frozen=True)
class ProbePositionsMetricsReport:
    scalars: ProbePositionsMetricsScalars
    positions: ProbePositionSequence
    anchor_indexes: IntegerArrayType
    """Distinct scan indexes, ascending."""
    anchor_xy_m: RealArrayType
    """``(N, 2)`` packed ``(x, y)`` position at each distinct index, duplicates averaged."""
    nearest_neighbor_distance_m: RealArrayType
    """Distance from each anchor to its nearest neighbour."""
    duplicated_indexes: IntegerArrayType
    index_gaps: IntegerArrayType
    """``(G, 2)`` first and last index of each missing run."""
    i0: RealArrayType | None
    """Per-position I0, in photons when calibrated; ``None`` when none was recorded."""


def compute_probe_positions_metrics_report(product: Product) -> ProbePositionsMetricsReport:
    """Describe ``product``'s scan: its indexes, extent, step sizes and I0.

    Positions sharing a scan index are averaged into one anchor before the step sizes,
    extent and hull are measured, as the pattern/position matcher would pair them. I0
    is converted to photons when the metadata carries ``i0_photons_per_count``.

    Raises:
        ValueError: If the product has no probe positions.
    """
    positions = product.probe_positions

    if len(positions) == 0:
        raise ValueError('Cannot describe a scan without probe positions!')

    indexes = positions.get_indexes()
    anchor_indexes, counts = numpy.unique(indexes, return_counts=True)
    _, anchor_xy_m = match_at_indexes(indexes, _packed_xy_m(positions), anchor_indexes)
    num_anchors = anchor_indexes.size

    if num_anchors > 1:
        distances, _ = cKDTree(anchor_xy_m).query(anchor_xy_m, k=2)
        nearest_neighbor_distance_m = distances[:, 1]
    else:
        nearest_neighbor_distance_m = numpy.empty(0)

    try:
        hull_area_m2 = float(ConvexHull(anchor_xy_m).volume) if num_anchors >= 3 else 0.0
    except QhullError:
        hull_area_m2 = 0.0

    extent_x_m = (float(numpy.min(anchor_xy_m[:, 0])), float(numpy.max(anchor_xy_m[:, 0])))
    extent_y_m = (float(numpy.min(anchor_xy_m[:, 1])), float(numpy.max(anchor_xy_m[:, 1])))
    index_gaps = _index_gaps(anchor_indexes)
    nn_mean, nn_median, nn_minimum, nn_maximum = _nan_statistics(nearest_neighbor_distance_m)

    i0 = positions.get_i0()
    i0_photons_per_count = product.metadata.i0_photons_per_count
    i0_is_calibrated = i0 is not None and i0_photons_per_count > 0.0

    if i0 is not None and i0_is_calibrated:
        i0 = i0 * i0_photons_per_count

    if i0 is None or i0.size == 0:
        nan = float('nan')
        i0_mean, i0_std, i0_minimum, i0_maximum = nan, nan, nan, nan
    else:
        i0_mean = float(numpy.mean(i0))
        i0_std = float(numpy.std(i0))
        i0_minimum = float(numpy.min(i0))
        i0_maximum = float(numpy.max(i0))

    return ProbePositionsMetricsReport(
        scalars=ProbePositionsMetricsScalars(
            num_positions=len(positions),
            num_distinct_indexes=int(num_anchors),
            index_minimum=int(anchor_indexes[0]),
            index_maximum=int(anchor_indexes[-1]),
            num_duplicated_indexes=int(numpy.count_nonzero(counts > 1)),
            num_missing_indexes=int(anchor_indexes[-1] - anchor_indexes[0] + 1 - num_anchors),
            num_index_gaps=int(index_gaps.shape[0]),
            width_m=extent_x_m[1] - extent_x_m[0],
            height_m=extent_y_m[1] - extent_y_m[0],
            center_x_m=(extent_x_m[0] + extent_x_m[1]) / 2.0,
            center_y_m=(extent_y_m[0] + extent_y_m[1]) / 2.0,
            convex_hull_area_m2=hull_area_m2,
            density_per_m2=num_anchors / hull_area_m2 if hull_area_m2 > 0.0 else float('nan'),
            nearest_neighbor_mean_m=nn_mean,
            nearest_neighbor_median_m=nn_median,
            nearest_neighbor_minimum_m=nn_minimum,
            nearest_neighbor_maximum_m=nn_maximum,
            is_exit_frame=_resolve_exit_transform(product.metadata)
            != AffineTransform.create_identity(),
            i0_is_calibrated=i0_is_calibrated,
            i0_mean=i0_mean,
            i0_std=i0_std,
            i0_minimum=i0_minimum,
            i0_maximum=i0_maximum,
        ),
        positions=positions,
        anchor_indexes=anchor_indexes,
        anchor_xy_m=anchor_xy_m,
        nearest_neighbor_distance_m=nearest_neighbor_distance_m,
        duplicated_indexes=anchor_indexes[counts > 1],
        index_gaps=index_gaps,
        i0=i0,
    )


@dataclass(frozen=True)
class ProbePositionsComparisonScalars:
    num_matched: int
    num_reference_only: int
    """Distinct reference indexes the test lacks."""
    num_test_only: int
    """Distinct test indexes the reference lacks."""
    mean_displacement_x_m: float
    mean_displacement_y_m: float
    rms_magnitude_m: float
    median_magnitude_m: float
    maximum_magnitude_m: float
    is_exit_frame: bool


@dataclass(frozen=True)
class ProbePositionsComparisonReport:
    scalars: ProbePositionsComparisonScalars
    indexes: IntegerArrayType
    """Scan indexes both products have, ascending."""
    reference_xy_m: RealArrayType
    """``(N, 2)`` packed ``(x, y)`` reference positions."""
    displacement_xy_m: RealArrayType
    """``(N, 2)`` packed ``(x, y)`` test minus reference."""
    magnitude_m: RealArrayType
    direction_rad: RealArrayType
    """Direction of each displacement, counterclockwise from +x."""


def compute_probe_positions_comparison_report(
    reference: Product, test: Product
) -> ProbePositionsComparisonReport:
    """Measure how ``test``'s probe positions are displaced from ``reference``'s.

    Positions pair at the scan indexes both products have, with duplicates averaged on
    each side and nothing interpolated; see
    :meth:`~ptychodus.api.probe_positions.ProbePositionSequence.match_indexes`.

    Raises:
        ValueError: If the two products' positions are in different frames, or they
            share no scan index.
    """
    reference_transform = _resolve_exit_transform(reference.metadata)

    if _resolve_exit_transform(test.metadata) != reference_transform:
        raise ValueError(
            'The reference and test positions are in different frames; their scattering'
            ' geometries map the probe positions differently, so displacements would mix frames.'
        )

    indexes, reference_xy_m, test_xy_m = reference.probe_positions.match_indexes(
        test.probe_positions
    )

    if indexes.size == 0:
        raise ValueError('The reference and test positions share no scan index!')

    displacement_xy_m = test_xy_m - reference_xy_m
    magnitude_m = numpy.hypot(displacement_xy_m[:, 0], displacement_xy_m[:, 1])
    mean_displacement = numpy.mean(displacement_xy_m, axis=0)
    reference_indexes = numpy.unique(reference.probe_positions.get_indexes())
    test_indexes = numpy.unique(test.probe_positions.get_indexes())

    return ProbePositionsComparisonReport(
        scalars=ProbePositionsComparisonScalars(
            num_matched=int(indexes.size),
            num_reference_only=int(reference_indexes.size - indexes.size),
            num_test_only=int(test_indexes.size - indexes.size),
            mean_displacement_x_m=float(mean_displacement[0]),
            mean_displacement_y_m=float(mean_displacement[1]),
            rms_magnitude_m=float(math.sqrt(numpy.mean(numpy.square(magnitude_m)))),
            median_magnitude_m=float(numpy.median(magnitude_m)),
            maximum_magnitude_m=float(numpy.max(magnitude_m)),
            is_exit_frame=reference_transform != AffineTransform.create_identity(),
        ),
        indexes=indexes,
        reference_xy_m=reference_xy_m,
        displacement_xy_m=displacement_xy_m,
        magnitude_m=magnitude_m,
        direction_rad=numpy.arctan2(displacement_xy_m[:, 1], displacement_xy_m[:, 0]),
    )
