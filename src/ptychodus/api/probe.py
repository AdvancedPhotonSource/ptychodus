"""Probe (illumination function) data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import overload
import logging
import math

import numpy
import scipy.ndimage
from scipy.fft import fft2

from .constants import format_length
from .fourier import fourier_shift_2d
from .geometry import ImageExtent, PixelGeometry
from .interpolate import resample_along_axis
from .preprocess.noise import estimate_noise_floor
from .propagate import PropagatedWavefield, compute_far_field_pixel_geometry, intensity
from .typing import ComplexArrayType, RealArrayType

logger = logging.getLogger(__name__)


def compute_shannon_entropy(distribution: RealArrayType, *, normalize: bool = True) -> float:
    """Shannon entropy (in bits) of a non-negative array treated as a distribution.

    Each element of ``distribution`` is normalized to a probability
    ``p_i = x_i / sum(x)`` and the Shannon entropy ``H = -sum(p_i log2 p_i)`` is
    computed over the non-zero probabilities. When ``normalize`` is ``True`` the
    result is divided by ``log2(N)`` (``N`` = number of elements), yielding a
    size-independent value in ``[0, 1]`` where ``1.0`` is a perfectly uniform
    distribution and values approaching ``0`` indicate concentration in a single
    element.

    Args:
        distribution: Array of non-negative values (e.g. an intensity or power
            spectrum). Negative values are clipped to zero.
        normalize: If ``True``, divide by ``log2(N)`` so the result lies in
            ``[0, 1]``. If ``False``, return the raw entropy in bits.

    Returns:
        The (optionally normalized) Shannon entropy. Returns ``0.0`` when the
        distribution has no positive mass.
    """
    values = numpy.clip(numpy.asarray(distribution, dtype=numpy.float64).ravel(), 0.0, None)
    total = values.sum()

    if total <= 0.0:
        return 0.0

    p = values / total
    nonzero = p[p > 0.0]
    entropy = -numpy.sum(nonzero * numpy.log2(nonzero))

    if normalize and p.size > 1:
        entropy /= numpy.log2(p.size)

    return float(entropy)


def compute_rms_contrast(image: RealArrayType) -> float:
    """Root-mean-square contrast ``std(I) / mean(I)`` of a non-negative image.

    Dividing by the mean makes the value independent of the image's overall scaling,
    so it is comparable between propagation planes and between probes. It attains a
    maximum where a converging beam is most concentrated, which makes it a focus
    indicator costing a single pass over the image rather than the median filter and
    sort that :func:`estimate_probe_size` requires.

    Returns:
        The RMS contrast, or ``0.0`` when the mean is not positive.
    """
    values = numpy.asarray(image, dtype=numpy.float64)
    mean = values.mean()

    if mean <= 0.0:
        return 0.0

    return float(values.std() / mean)


def compute_amplitude_deviation(wavefield: ComplexArrayType) -> float:
    """Normalized amplitude dispersion ``std(|psi|) / mean(|psi|)`` of a complex wavefield.

    The amplitude counterpart of :func:`compute_phase_deviation_rad`, normalized by the
    mean for the same scale-invariance reason as :func:`compute_rms_contrast`.

    Returns:
        The normalized amplitude deviation, or ``0.0`` when the mean amplitude is not
        positive.
    """
    amplitude = numpy.absolute(numpy.asarray(wavefield))
    mean = amplitude.mean()

    if mean <= 0.0:
        return 0.0

    return float(amplitude.std() / mean)


def compute_phase_deviation_rad(wavefield: ComplexArrayType) -> float:
    """Intensity-weighted circular standard deviation of a wavefield's phase, in radians.

    With weights ``w = |psi|^2``, the normalized resultant of the phase distribution is
    ``R = |sum(w exp(i phi))| / sum(w)``. That simplifies to
    ``|sum(psi |psi|)| / sum(|psi|^2)``, so it is evaluated without ever taking an
    angle, and the dispersion reported is ``sqrt(-2 ln R)`` -- the circular analogue of
    a standard deviation, exactly recovering ``sigma`` for a wrapped-normal phase.

    Being built from a complex weighted sum, the result does not depend on where the
    branch cut of the phase happens to fall: a wavefield whose phase straddles
    ``+/-pi`` is measured as tightly clustered, where a plain standard deviation of
    :func:`numpy.angle` would report it as maximally dispersed. The same reasoning
    underlies the phase handling in :mod:`ptychodus.api.metrics`.

    Weighting by intensity means dim pixels, whose phase is the least trustworthy part
    of a reconstruction, contribute in proportion to how much light they carry.

    A flat wavefront gives ``0.0``. The value grows without bound as the phase spreads;
    for a phase that is uniformly distributed over the circle the resultant is limited
    only by the sample count, so the reported dispersion grows like ``sqrt(ln N)``.

    Returns:
        The circular standard deviation in radians; ``0.0`` when the wavefield carries
        no power, and ``inf`` in the degenerate case of an exactly vanishing resultant.
    """
    values = numpy.asarray(wavefield)
    amplitude = numpy.absolute(values)
    total_power = numpy.square(amplitude).sum()

    if total_power <= 0.0:
        return 0.0

    # sum(w exp(i phi)) == sum(|psi|^2 * psi / |psi|) == sum(psi |psi|), which also
    # leaves zero-amplitude pixels contributing nothing instead of dividing by zero.
    resultant = float(numpy.absolute(numpy.sum(values * amplitude)) / total_power)

    if resultant <= 0.0:
        return float('inf')

    # Rounding can push a perfectly coherent sum a hair above one, where the log would
    # return a small negative and the square root a NaN.
    return float(numpy.sqrt(-2.0 * numpy.log(min(resultant, 1.0))))


@dataclass(frozen=True)
class ProbeEntropyMetrics:
    """Normalized Shannon-entropy metrics for a probe, in bits and in ``[0, 1]``."""

    real_space_intensity_entropy: float
    """Normalized entropy of the real-space intensity distribution."""

    spectral_entropy: float
    """Normalized entropy of the power-spectrum (frequency-domain) distribution."""


@dataclass(frozen=True)
class ProbeSizeMetrics:
    """Probe size metrics: principal-axis tilt, FWHM and RMS extents, and encircled-energy diameter."""

    major_axis_tilt_rad: float
    minor_axis_tilt_rad: float

    fwhm_major_axis_length_m: float
    fwhm_minor_axis_length_m: float

    rms_major_axis_length_m: float
    rms_minor_axis_length_m: float

    encircled_energy_diameter_m: float


def _projected_fwhm(
    coordinate: RealArrayType,
    intensity: RealArrayType,
    num_bins: int,
) -> float:
    """FWHM of a 2D intensity distribution projected onto an arbitrary axis."""
    coord_flat = coordinate.ravel()
    intensity_flat = intensity.ravel()
    cmin = coord_flat.min()
    cmax = coord_flat.max()

    if cmax <= cmin:
        return 0.0

    hist, edges = numpy.histogram(
        coord_flat, bins=num_bins, range=(cmin, cmax), weights=intensity_flat
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    peak = hist.max()

    if peak <= 0.0:
        return 0.0

    half_max = 0.5 * peak
    above = hist >= half_max
    # Use the outermost crossings so isolated noise spikes inside the profile
    # don't fragment the half-max interval.
    left_idx = numpy.argmax(above)
    right_idx = len(above) - 1 - numpy.argmax(above[::-1])

    if left_idx == 0:
        left_x = centers[0]
    else:
        y0 = hist[left_idx - 1]
        y1 = hist[left_idx]
        x0 = centers[left_idx - 1]
        x1 = centers[left_idx]
        left_x = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y1 > y0 else x0

    if right_idx >= len(centers) - 1:
        right_x = centers[-1]
    else:
        y0 = hist[right_idx]
        y1 = hist[right_idx + 1]
        x0 = centers[right_idx]
        x1 = centers[right_idx + 1]
        right_x = x0 + (half_max - y0) * (x1 - x0) / (y1 - y0) if y0 > y1 else x1

    return float(right_x - left_x)


def estimate_probe_size(
    probe_intensity: RealArrayType,
    pixel_geometry: PixelGeometry,
    *,
    energy_fraction: float = 0.8,
    mad_threshold: float = 4.5,
) -> ProbeSizeMetrics:
    """Estimate transverse probe-size metrics from a 2D intensity distribution.

    The pipeline is:

    1. **Pre-filter and noise floor.** The input is passed through a 3x3
       median filter to suppress hot pixels and other isolated outliers
       (matching :func:`ptychodus.api.preprocess.diffraction.estimate_beam_center`).
       Background and noise scale are then estimated via
       :func:`ptychodus.api.preprocess.noise.estimate_noise_floor`, which uses Otsu's
       method on the filtered image to identify the background class when
       the histogram is bimodal and falls back to median / median-absolute-
       deviation over the outermost ring of pixels when it is not. The
       filtered image is then shifted down by ``background + mad_threshold
       * MAD`` and clipped to non-negative values. Larger ``mad_threshold``
       is more aggressive at suppressing noise tails but increasingly
       truncates real signal in the wings.
    2. **Principal axes.** The centroid and intensity-weighted 2x2 covariance
       are computed on the cleaned image (in physical metres). Its eigenvectors
       define the major/minor axes; the tilt of each is reported in radians,
       folded into ``[-pi/2, pi/2)`` since the axis direction is sign-ambiguous.
    3. **RMS widths.** Twice the square root of each covariance eigenvalue —
       i.e. the full ``2 sigma`` width of the intensity distribution along
       each principal axis. The factor of two makes these comparable to the
       FWHM and encircled-energy *diameters* rather than radii.
    4. **FWHM widths.** The cleaned intensity is projected onto each principal
       axis (weighted 1D histogram) and the full width at half maximum is read
       off by linearly interpolating the outermost half-max crossings, which
       makes the result insensitive to isolated bins above half-max inside the
       profile.
    5. **Encircled-energy diameter.** Pixels are sorted by radial distance from
       the centroid; the cumulative cleaned power is taken; and the diameter
       reported is twice the radius at which the cumulative power reaches
       ``energy_fraction`` of the total (with linear interpolation between
       adjacent sorted pixels).

    The major and minor axis tilts in :class:`ProbeSizeMetrics` are *shared*
    between the FWHM and RMS measurements: both are reported along the
    eigenvectors of the second-moment covariance. This assumes the intensity
    distribution is approximately elliptically symmetric (the typical case for
    Gaussian-like probes), so the half-max contour and the variance ellipse
    line up. For distributions where they don't — bimodal lobes, vortex /
    donut probes, or strongly non-elliptical apertures — the reported FWHM
    values are still the projections onto the variance principal axes, which
    may not coincide with the directions of largest / smallest half-max
    extent.

    Args:
        probe_intensity: 2D array of intensity values (any non-negative units).
        pixel_geometry: Physical pixel size used to convert pixel indices into
            metres.
        energy_fraction: Fraction of cleaned total power that defines the
            encircled-energy diameter; must be in ``(0, 1]``.
        mad_threshold: Soft-threshold level, in units of border-MAD, applied
            above the estimated background. ``0.0`` disables thresholding
            (background is still subtracted).

    Raises:
        ValueError: If ``probe_intensity`` is not 2D, ``energy_fraction`` is
            outside ``(0, 1]``, or ``mad_threshold`` is negative.

    Returns:
        A :class:`ProbeSizeMetrics` populated with the shared axis tilts and
        the FWHM, RMS, and encircled-energy widths. If thresholding wipes out
        all signal (cleaned total power is zero), every field is returned as
        ``0.0``.
    """

    if probe_intensity.ndim != 2:
        raise ValueError(f'probe_intensity must be 2-dimensional, got {probe_intensity.ndim}D')

    if not (0.0 < energy_fraction <= 1.0):
        raise ValueError(f'energy_fraction must be in (0, 1], got {energy_fraction}')

    if mad_threshold < 0.0:
        raise ValueError(f'mad_threshold must be non-negative, got {mad_threshold}')

    height_px, width_px = probe_intensity.shape

    filtered = scipy.ndimage.median_filter(probe_intensity.astype(numpy.float64), size=3)

    border = numpy.concatenate(
        [
            filtered[0, :].ravel(),
            filtered[-1, :].ravel(),
            filtered[1:-1, 0].ravel(),
            filtered[1:-1, -1].ravel(),
        ]
    )
    robust_statistics = estimate_noise_floor(filtered, fallback_values=border)
    threshold = robust_statistics.get_significance_threshold(mad_threshold)

    cleaned = numpy.clip(filtered - threshold, 0.0, None)
    total_power = cleaned.sum()

    if total_power <= 0.0:
        return ProbeSizeMetrics(
            major_axis_tilt_rad=0.0,
            minor_axis_tilt_rad=0.0,
            fwhm_major_axis_length_m=0.0,
            fwhm_minor_axis_length_m=0.0,
            rms_major_axis_length_m=0.0,
            rms_minor_axis_length_m=0.0,
            encircled_energy_diameter_m=0.0,
        )

    y_idx, x_idx = numpy.mgrid[:height_px, :width_px]  # noqa: N806
    x_m = (x_idx - (width_px - 1) / 2.0) * pixel_geometry.width_m
    y_m = (y_idx - (height_px - 1) / 2.0) * pixel_geometry.height_m

    centroid_x = (cleaned * x_m).sum() / total_power
    centroid_y = (cleaned * y_m).sum() / total_power

    dx = x_m - centroid_x
    dy = y_m - centroid_y

    mxx = (cleaned * dx * dx).sum() / total_power
    myy = (cleaned * dy * dy).sum() / total_power
    mxy = (cleaned * dx * dy).sum() / total_power
    covariance = numpy.array([[mxx, mxy], [mxy, myy]])

    eigenvalues, eigenvectors = numpy.linalg.eigh(covariance)
    rms_minor_m = 2.0 * numpy.sqrt(max(eigenvalues[0], 0.0))
    rms_major_m = 2.0 * numpy.sqrt(max(eigenvalues[1], 0.0))
    minor_axis = eigenvectors[:, 0]
    major_axis = eigenvectors[:, 1]

    def _axis_tilt(axis: RealArrayType) -> float:
        # axis direction is sign-ambiguous; fold the angle into [-pi/2, pi/2)
        angle = numpy.arctan2(axis[1], axis[0])
        return float((angle + numpy.pi / 2.0) % numpy.pi - numpy.pi / 2.0)

    major_tilt = _axis_tilt(major_axis)
    minor_tilt = _axis_tilt(minor_axis)

    num_bins = max(height_px, width_px)
    projection_major = dx * major_axis[0] + dy * major_axis[1]
    projection_minor = dx * minor_axis[0] + dy * minor_axis[1]
    fwhm_major = _projected_fwhm(projection_major, cleaned, num_bins)
    fwhm_minor = _projected_fwhm(projection_minor, cleaned, num_bins)

    radial = numpy.hypot(dx, dy).ravel()
    intensity_flat = cleaned.ravel()
    order = numpy.argsort(radial)
    sorted_radii = radial[order]
    sorted_power = intensity_flat[order]
    cumulative = numpy.cumsum(sorted_power)
    target = energy_fraction * cumulative[-1]
    idx = numpy.searchsorted(cumulative, target)

    if idx <= 0:
        encircled_radius = sorted_radii[0]
    elif idx >= len(sorted_radii):
        encircled_radius = sorted_radii[-1]
    else:
        c0 = cumulative[idx - 1]
        c1 = cumulative[idx]
        r0 = sorted_radii[idx - 1]
        r1 = sorted_radii[idx]
        encircled_radius = r0 + (target - c0) * (r1 - r0) / (c1 - c0) if c1 > c0 else r1

    return ProbeSizeMetrics(
        major_axis_tilt_rad=major_tilt,
        minor_axis_tilt_rad=minor_tilt,
        fwhm_major_axis_length_m=fwhm_major,
        fwhm_minor_axis_length_m=fwhm_minor,
        rms_major_axis_length_m=float(rms_major_m),
        rms_minor_axis_length_m=float(rms_minor_m),
        encircled_energy_diameter_m=float(2.0 * encircled_radius),
    )


class FocusPolarity(Enum):
    """Whether a focus metric is best at a minimum or a maximum of its z curve."""

    MINIMUM = auto()
    MAXIMUM = auto()


class ProbeFocusMetric(Enum):
    """Focus metrics sampled along a propagation axis.

    Each member pairs a short display name with the direction in which the metric
    improves and the SI unit its samples are stored in (``'m'``, ``'rad'``, or the
    empty string when dimensionless). The name deliberately carries no unit -- callers
    append one derived from :attr:`si_unit`, converting to whichever display unit suits
    them, so that a script and a GUI describe the same quantity the same way.
    """

    FWHM_MAJOR = ('FWHM Major Axis', FocusPolarity.MINIMUM, 'm')
    FWHM_MINOR = ('FWHM Minor Axis', FocusPolarity.MINIMUM, 'm')
    RMS_MAJOR = ('RMS Major Axis', FocusPolarity.MINIMUM, 'm')
    RMS_MINOR = ('RMS Minor Axis', FocusPolarity.MINIMUM, 'm')
    ENCIRCLED_ENERGY_DIAMETER = ('Encircled Energy Diameter', FocusPolarity.MINIMUM, 'm')
    PEAK_INTENSITY = ('Peak Intensity', FocusPolarity.MAXIMUM, '')
    RMS_CONTRAST = ('RMS Contrast', FocusPolarity.MAXIMUM, '')
    INTENSITY_ENTROPY = ('Intensity Entropy', FocusPolarity.MINIMUM, '')
    AMPLITUDE_DEVIATION = ('Amplitude Deviation', FocusPolarity.MAXIMUM, '')
    PHASE_DEVIATION = ('Phase Deviation', FocusPolarity.MINIMUM, 'rad')

    def __init__(self, label: str, polarity: FocusPolarity, si_unit: str) -> None:
        self.label = label
        self.polarity = polarity
        self.si_unit = si_unit


@dataclass(frozen=True)
class ProbeFocusSeries:
    """One focus metric sampled at every plane of a propagated probe."""

    metric: ProbeFocusMetric
    """Which metric this curve measures, and how to interpret it."""

    value: RealArrayType
    """Samples in the SI unit named by ``metric.si_unit``, one per propagation step."""


@dataclass(frozen=True)
class FocalPlane:
    """Best-focus plane estimated from a single metric curve."""

    coordinate_m: float
    """Propagation coordinate of the estimated focus, in meters."""

    value: float
    """The metric's value there, interpolated when :attr:`is_refined` is true."""

    step: int
    """Index of the sampled plane nearest the focus."""

    is_refined: bool
    """Whether sub-step interpolation succeeded. When false the estimate is the
    extremal sample itself, so its precision is the sample spacing."""


@dataclass(frozen=True)
class ProbeFocusCurves:
    """Focus metrics for a propagated probe, sampled along the propagation axis."""

    coordinate_m: RealArrayType
    """Propagation coordinates in meters, shape ``(num_steps,)``."""

    series: Sequence[ProbeFocusSeries]
    """One entry per sampled metric, each the same length as :attr:`coordinate_m`."""

    def get_series(self, metric: ProbeFocusMetric) -> ProbeFocusSeries:
        """Return the sampled curve for *metric*.

        Raises:
            KeyError: If *metric* was not sampled.
        """
        for series in self.series:
            if series.metric is metric:
                return series

        raise KeyError(f'No focus series for {metric}!')

    def get_focal_plane(self, metric: ProbeFocusMetric) -> FocalPlane:
        """Best-focus plane according to *metric*, via :func:`estimate_focal_plane`.

        Raises:
            KeyError: If *metric* was not sampled.
        """
        series = self.get_series(metric)
        return estimate_focal_plane(self.coordinate_m, series.value, metric.polarity)


def estimate_focal_plane(
    coordinate_m: RealArrayType, value: RealArrayType, polarity: FocusPolarity
) -> FocalPlane:
    """Locate the best-focus plane of a sampled metric curve by parabolic refinement.

    The extremal sample is found according to *polarity*, a parabola is fitted through
    it and its two neighbors, and the vertex is reported. This resolves the focus to a
    fraction of the sample spacing, and is exact wherever the curve is locally
    quadratic -- which these metrics are near their extremum.

    The refinement is abandoned, and the extremal sample returned unchanged with
    :attr:`FocalPlane.is_refined` false, whenever it cannot be trusted:

    - there are fewer than three samples, or the extremum falls on either end of the
      curve, so that it is not bracketed;
    - the parabola is degenerate, meaning a flat or near-flat neighborhood in which the
      vertex could land anywhere;
    - the vertex leaves the bracketing interval.

    Args:
        coordinate_m: Propagation coordinates in meters, evenly spaced and increasing.
        value: Metric samples, the same shape as *coordinate_m*.
        polarity: Whether the metric is best at a minimum or a maximum.

    Raises:
        ValueError: If the two arrays differ in shape, or are empty.
    """
    coordinates = numpy.asarray(coordinate_m, dtype=numpy.float64)
    values = numpy.asarray(value, dtype=numpy.float64)

    if coordinates.shape != values.shape:
        raise ValueError(
            f'Coordinate and value arrays must have same shape; '
            f'got {coordinates.shape} vs {values.shape}!'
        )

    if values.size == 0:
        raise ValueError('Cannot locate a focal plane in an empty curve!')

    if polarity is FocusPolarity.MAXIMUM:
        index = int(numpy.argmax(values))
    else:
        index = int(numpy.argmin(values))

    grid_plane = FocalPlane(
        coordinate_m=float(coordinates[index]),
        value=float(values[index]),
        step=index,
        is_refined=False,
    )

    if values.size < 3 or index == 0 or index == values.size - 1:
        return grid_plane

    y_left = values[index - 1]
    y_here = values[index]
    y_right = values[index + 1]
    denominator = y_left - 2.0 * y_here + y_right

    # Scale the flatness tolerance to the curve so that the test means the same thing
    # whether the samples are nanometer-scale lengths or order-one dimensionless ratios.
    tolerance = 1e-12 * max(float(numpy.absolute(values).max()), 1.0)

    if numpy.absolute(denominator) <= tolerance:
        return grid_plane

    offset = 0.5 * (y_left - y_right) / denominator

    if numpy.absolute(offset) > 1.0:
        return grid_plane

    # Half the span of the bracketing pair, so a locally uneven grid still gives the
    # right step size without assuming the whole axis is uniform.
    spacing_m = 0.5 * (coordinates[index + 1] - coordinates[index - 1])
    focus_m = coordinates[index] + offset * spacing_m
    # Vertex of the same parabola: y(x*) = y_here - (y_left - y_right) * x* / 4.
    vertex = y_here - 0.25 * (y_left - y_right) * offset

    return FocalPlane(
        coordinate_m=float(focus_m),
        value=float(vertex),
        step=int(numpy.argmin(numpy.absolute(coordinates - focus_m))),
        is_refined=True,
    )


def compute_probe_focus_curves(
    propagated_probe: PropagatedWavefield, *, mode: int = 0
) -> ProbeFocusCurves:
    """Sample every :class:`ProbeFocusMetric` at each plane of a propagated probe.

    The result is the input to :func:`estimate_focal_plane`, and is what turns a stack
    of propagated wavefields into a statement about where the probe comes to focus.

    Two families are measured on different data. The size and intensity metrics use the
    mode-summed intensity, since that is the physically observable quantity. The
    amplitude and phase deviations are measured on a single incoherent mode selected by
    *mode*, because phase is only defined per mode -- summing mutually incoherent modes
    is meaningful in intensity alone.

    Values are stored raw and in SI units, with no normalization applied, so that a
    caller comparing curves on one axis controls that choice itself.

    Cost is dominated by :func:`estimate_probe_size`, which median-filters, runs Otsu,
    and sorts every pixel once per plane; expect this to scale linearly in the step
    count and to run for seconds over a long sweep. The intensity stack is materialized
    once here rather than per step, since :attr:`PropagatedWavefield.intensity` rebuilds it
    on every access and would otherwise make the sweep quadratic.

    Args:
        propagated_probe: The propagated wavefield stack to measure.
        mode: Zero-based incoherent mode used for the wavefront metrics.

    Raises:
        ValueError: If *mode* is not a valid incoherent mode index.
    """
    num_modes = propagated_probe.num_incoherent_modes

    if not 0 <= mode < num_modes:
        raise ValueError(f'Mode index must be in [0, {num_modes}); got {mode}!')

    # Hoisted deliberately -- see the note on cost above.
    stack = propagated_probe.intensity
    pixel_geometry = propagated_probe.pixel_geometry
    num_steps = propagated_probe.num_steps

    coordinate_m = numpy.linspace(
        propagated_probe.begin_coordinate_m,
        propagated_probe.end_coordinate_m,
        num_steps,
    )
    samples: dict[ProbeFocusMetric, list[float]] = {metric: [] for metric in ProbeFocusMetric}

    for step in range(num_steps):
        plane = stack[step]
        size = estimate_probe_size(plane, pixel_geometry)
        wavefield = propagated_probe.get_xy_wavefield(step, mode)

        step_values = {
            ProbeFocusMetric.FWHM_MAJOR: size.fwhm_major_axis_length_m,
            ProbeFocusMetric.FWHM_MINOR: size.fwhm_minor_axis_length_m,
            ProbeFocusMetric.RMS_MAJOR: size.rms_major_axis_length_m,
            ProbeFocusMetric.RMS_MINOR: size.rms_minor_axis_length_m,
            ProbeFocusMetric.ENCIRCLED_ENERGY_DIAMETER: size.encircled_energy_diameter_m,
            ProbeFocusMetric.PEAK_INTENSITY: float(plane.max()),
            ProbeFocusMetric.RMS_CONTRAST: compute_rms_contrast(plane),
            ProbeFocusMetric.INTENSITY_ENTROPY: compute_shannon_entropy(plane),
            ProbeFocusMetric.AMPLITUDE_DEVIATION: compute_amplitude_deviation(wavefield),
            ProbeFocusMetric.PHASE_DEVIATION: compute_phase_deviation_rad(wavefield),
        }

        for metric, metric_value in step_values.items():
            samples[metric].append(metric_value)

    return ProbeFocusCurves(
        coordinate_m=coordinate_m,
        series=tuple(
            ProbeFocusSeries(metric=metric, value=numpy.array(values))
            for metric, values in samples.items()
        ),
    )


@dataclass(frozen=True)
class ProbeTransverseCoordinates:
    """2D Cartesian coordinate arrays for the transverse plane of the probe, in meters."""

    x_m: RealArrayType
    y_m: RealArrayType

    @property
    def position_r_m(self) -> RealArrayType:
        return numpy.hypot(self.y_m, self.x_m)

    @property
    def angle_rad(self) -> RealArrayType:
        return numpy.arctan2(self.y_m, self.x_m)


@dataclass(frozen=True)
class PatchBounds:
    """Indexing bounds and sub-pixel offset for a probe-sized patch anchored at a float object-pixel center.

    ``x_slice`` and ``y_slice`` are numpy slices that select a
    ``height_px x width_px`` region from an object canvas at the integer
    lower-left corner. ``dx`` and ``dy`` are the residual sub-pixel offsets
    that a Fourier shift must apply to the probe.
    """

    x_slice: slice
    y_slice: slice
    dx: float
    dy: float


@dataclass(frozen=True)
class ProbeGeometry:
    """Pixel dimensions and physical size of the probe array."""

    width_px: int
    height_px: int
    pixel_width_m: float
    pixel_height_m: float

    @classmethod
    def from_far_field(
        cls,
        detector_pixel_geometry: PixelGeometry,
        image_extent: ImageExtent,
        *,
        wavelength_m: float,
        distance_m: float,
    ) -> ProbeGeometry:
        """Sample-plane probe geometry from the Fraunhofer relation ``dx_sample = lambda * |z| / (N * dx_detector)``."""
        pixel_geometry = compute_far_field_pixel_geometry(
            detector_pixel_geometry,
            image_extent,
            wavelength_m=wavelength_m,
            propagation_distance_m=distance_m,
        )
        return cls(
            width_px=image_extent.width_px,
            height_px=image_extent.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )

    @property
    def width_m(self) -> float:
        return self.width_px * self.pixel_width_m

    @property
    def height_m(self) -> float:
        return self.height_px * self.pixel_height_m

    def get_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(
            width_m=self.pixel_width_m,
            height_m=self.pixel_height_m,
        )

    def get_transverse_coordinates(self) -> ProbeTransverseCoordinates:
        Y, X = numpy.mgrid[: self.height_px, : self.width_px]  # noqa: N806
        x_px = X - (self.width_px - 1) / 2
        y_px = Y - (self.height_px - 1) / 2
        return ProbeTransverseCoordinates(
            x_m=x_px * self.pixel_width_m, y_m=y_px * self.pixel_height_m
        )

    def resolve_patch_bounds(self, cx: float, cy: float) -> PatchBounds:
        """Locate a ``height_px x width_px`` patch anchored at object-pixel center ``(cx, cy)``.

        Under the (N-1)/2 centered-pixel convention shared with
        ``get_transverse_coordinates``, returns the numpy slices for the
        integer lower-left corner and the residual sub-pixel offset that a
        Fourier shift must apply.

        The integer split uses Python ``int()`` (truncate toward zero), which
        equals ``math.floor`` when ``cx - (width_px - 1) / 2`` and
        ``cy - (height_px - 1) / 2`` are non-negative — the standard case for
        object-canvas coordinates. Behavior differs for negative arguments.
        """
        rx_px = (self.width_px - 1) / 2
        ry_px = (self.height_px - 1) / 2
        x_lower = int(cx - rx_px)
        y_lower = int(cy - ry_px)
        return PatchBounds(
            x_slice=slice(x_lower, x_lower + self.width_px),
            y_slice=slice(y_lower, y_lower + self.height_px),
            dx=cx - (x_lower + rx_px),
            dy=cy - (y_lower + ry_px),
        )

    def __str__(self) -> str:
        pixel_geometry = self.get_pixel_geometry()
        width_label = format_length(self.pixel_width_m)

        if pixel_geometry.is_square:
            pitch = f'{width_label}/px'
        else:
            pitch = f'{width_label} x {format_length(self.pixel_height_m)}/px'

        return f'{self.width_px} x {self.height_px} px @ {pitch}'


class ProbeGeometryProvider(ABC):
    """Abstract source of detector and probe geometry."""

    @property
    @abstractmethod
    def detector_distance_m(self) -> float:
        pass

    @property
    @abstractmethod
    def probe_photon_count(self) -> float:
        pass

    @property
    @abstractmethod
    def probe_wavelength_m(self) -> float:
        pass

    @property
    @abstractmethod
    def probe_power_W(self) -> float:  # noqa: N802
        pass

    @property
    @abstractmethod
    def num_scan_points(self) -> int:
        pass

    @abstractmethod
    def get_detector_pixel_geometry(self) -> PixelGeometry:
        pass

    @abstractmethod
    def get_probe_geometry(self) -> ProbeGeometry:
        pass


class Probe:
    """Probe (illumination function) stored as a (modes, height, width) complex array."""

    def __init__(
        self,
        array: ComplexArrayType,
        pixel_geometry: PixelGeometry,
    ) -> None:
        if numpy.iscomplexobj(array):
            match array.ndim:
                case 2:
                    self._array = array[numpy.newaxis, :, :]
                case 3:
                    self._array = array
                case _:
                    raise ValueError('Probe must be a 2- or 3-dimensional ndarray.')

        self._pixel_geometry = pixel_geometry

        power = numpy.sum(intensity(self._array), axis=(-2, -1))
        powersum = numpy.sum(power)

        if powersum > 0.0:
            power /= powersum

        self._mode_relative_power = power.tolist()

    @property
    def nbytes(self) -> int:
        return self._array.nbytes

    def copy(self) -> Probe:
        return Probe(
            array=self._array.copy(),
            pixel_geometry=self._pixel_geometry.copy(),
        )

    def get_array(self) -> ComplexArrayType:
        return self._array

    def get_pixel_geometry(self) -> PixelGeometry:
        return self._pixel_geometry

    @property
    def dtype(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def width_px(self) -> int:
        return self._array.shape[-1]

    @property
    def height_px(self) -> int:
        return self._array.shape[-2]

    @property
    def num_incoherent_modes(self) -> int:
        return self._array.shape[-3]

    def get_incoherent_mode(self, number: int) -> ComplexArrayType:
        return self._array[number, :, :]

    def get_incoherent_modes_flattened(self) -> ComplexArrayType:
        return self._array.transpose((1, 0, 2)).reshape(self.height_px, -1)

    def get_incoherent_mode_relative_power(self, number: int) -> float:
        return self._mode_relative_power[number]

    def get_coherence(self) -> float:
        return numpy.sqrt(numpy.sum(numpy.square(self._mode_relative_power)))

    def get_intensity(self) -> RealArrayType:
        return numpy.sum(intensity(self._array), axis=-3)

    def get_power_spectrum(self) -> RealArrayType:
        """Incoherent-sum power spectrum ``|FFT(psi)|^2`` over the mode axis.

        No fftshift is applied: Shannon entropy is permutation-invariant, so the
        frequency ordering is irrelevant for entropy calculations.
        """
        return numpy.sum(intensity(fft2(self._array, axes=(-2, -1))), axis=-3)


def estimate_probe_entropy(probe: Probe) -> ProbeEntropyMetrics:
    """Compute normalized real-space and spectral Shannon entropy for a probe.

    Both quantities use the incoherent sum over modes: the real-space entropy is
    computed from :meth:`Probe.get_intensity` and the spectral entropy from
    :meth:`Probe.get_power_spectrum`. Each is a normalized value in ``[0, 1]``
    (see :func:`compute_shannon_entropy`).
    """
    return ProbeEntropyMetrics(
        real_space_intensity_entropy=compute_shannon_entropy(probe.get_intensity()),
        spectral_entropy=compute_shannon_entropy(probe.get_power_spectrum()),
    )


def shift_probe(probe: Probe, *, shift_y_px: float, shift_x_px: float) -> Probe:
    """Translate every incoherent mode of ``probe`` by ``(shift_y_px, shift_x_px)``.

    The translation is applied as a Fourier phase ramp, matching :func:`shift_object`, so
    the complex phase survives and the shift is exact for a bandlimited wavefield at
    subpixel offsets. It is circular: content pushed past one edge reappears at the
    opposite one, which is inert for a probe whose support sits well inside the frame.
    A probe carries no world-coordinate center, so no metadata is adjusted.
    """
    if shift_y_px == 0.0 and shift_x_px == 0.0:
        return probe

    return Probe(
        array=fourier_shift_2d(probe.get_array(), dx=shift_x_px, dy=shift_y_px),
        pixel_geometry=probe.get_pixel_geometry().copy(),
    )


class ProbeSequence(Sequence[Probe]):
    """Position-dependent probe ensemble stored as a (coherent, incoherent, height, width) array.

    Supports optional OPR (orthogonal probe relaxation) weights for per-position probe variation.
    """

    def __init__(
        self,
        array: ComplexArrayType | None,
        opr_weights: RealArrayType | None,
        pixel_geometry: PixelGeometry | None,
    ) -> None:
        if array is None:
            self._array: ComplexArrayType = numpy.zeros((1, 1, 0, 0), dtype=complex)
        elif numpy.iscomplexobj(array):
            match array.ndim:
                case 2:
                    self._array = array[numpy.newaxis, numpy.newaxis, ...]
                case 3:
                    self._array = array[numpy.newaxis, ...]
                case 4:
                    self._array = array
                case _:
                    raise ValueError('Probe must be 2-, 3-, or 4-dimensional ndarray.')
        else:
            raise TypeError('Probe must be a complex-valued ndarray')

        if opr_weights is None:
            self._opr_weights = None
        elif numpy.issubdtype(opr_weights.dtype, numpy.floating):
            if opr_weights.ndim == 2:
                num_weights_actual = opr_weights.shape[1]
                num_weights_expected = self._array.shape[0]

                if num_weights_actual == num_weights_expected:
                    self._opr_weights = opr_weights
                else:
                    raise ValueError(
                        (
                            'inconsistent number of opr weights!'
                            f' actual={num_weights_actual}'
                            f' expected={num_weights_expected}'
                        )
                    )
            else:
                raise ValueError('opr_weights must be 2-dimensional ndarray')
        else:
            raise TypeError('opr_weights must be a floating-point ndarray')

        self._pixel_geometry = pixel_geometry

    @classmethod
    def from_probe(cls, probe: Probe) -> ProbeSequence:
        """Wrap a single :class:`Probe` as a length-1 sequence with no OPR basis."""
        return cls(
            array=probe.get_array(),
            opr_weights=None,
            pixel_geometry=probe.get_pixel_geometry(),
        )

    def copy(self) -> ProbeSequence:
        return ProbeSequence(
            self._array.copy(),
            None if self._opr_weights is None else self._opr_weights.copy(),
            None if self._pixel_geometry is None else self._pixel_geometry.copy(),
        )

    def get_array(self) -> ComplexArrayType:
        return self._array

    def get_opr_weights(self) -> RealArrayType:
        if self._opr_weights is None:
            raise ValueError('Missing opr_weights!')

        return self._opr_weights

    def get_opr_weights_or_none(self) -> RealArrayType | None:
        return self._opr_weights

    def get_pixel_geometry(self) -> PixelGeometry:
        if self._pixel_geometry is None:
            raise ValueError('Missing probe pixel geometry!')

        return self._pixel_geometry

    @property
    def dtype(self) -> numpy.dtype:
        return self._array.dtype

    @property
    def nbytes(self) -> int:
        sz = self._array.nbytes

        if self._opr_weights is not None:
            sz += self._opr_weights.nbytes

        return sz

    @property
    def num_coherent_modes(self) -> int:
        return self._array.shape[0]

    @property
    def num_incoherent_modes(self) -> int:
        return self._array.shape[1]

    @property
    def height_px(self) -> int:
        return self._array.shape[2]

    @property
    def width_px(self) -> int:
        return self._array.shape[3]

    @overload
    def __getitem__(self, index: int) -> Probe: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[Probe]: ...

    def __getitem__(self, index: int | slice) -> Probe | Sequence[Probe]:
        if isinstance(index, slice):
            # slice.indices normalizes implicit bounds, negative indexes, and
            # out-of-range values against the sequence length.
            return [self[idx] for idx in range(*index.indices(len(self)))]

        array = self._array[0, :, :, :].copy()

        if self._opr_weights is not None:
            array[0, :, :] = numpy.tensordot(
                self._opr_weights[index, :], self._array[:, 0, :, :], axes=1
            )

        return Probe(array, self.get_pixel_geometry())

    def get_probe_no_opr(self) -> Probe:
        array = self._array[0, :, :, :].copy()
        return Probe(array, self.get_pixel_geometry())

    def get_geometry(self) -> ProbeGeometry:
        pixel_geometry = self.get_pixel_geometry()

        return ProbeGeometry(
            width_px=self.width_px,
            height_px=self.height_px,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
        )

    def __len__(self) -> int:
        return 1 if self._opr_weights is None else self._opr_weights.shape[0]

    def __repr__(self) -> str:
        return f'{self._array.dtype}{self._array.shape}'


def _centered_grid_indexes(
    source_pixel_m: float, source_px: int, target_pixel_m: float, target_px: int
) -> RealArrayType:
    """Fractional source indexes sampled by each target pixel of a co-centered grid.

    Centered-pixel convention: the center of a grid sits at index ``(N - 1) / 2``, so the
    two grids share a center and differ only in pitch and extent.
    """
    offsets_px = numpy.arange(target_px) - (target_px - 1) / 2
    return offsets_px * (target_pixel_m / source_pixel_m) + (source_px - 1) / 2


def resample_probe_sequence(
    probes: ProbeSequence, target_geometry: ProbeGeometry, *, rel_tol: float = 1.0e-9
) -> ProbeSequence:
    """Re-express ``probes`` on ``target_geometry``, preserving the integrated power.

    A probe saved by a previous run samples the illumination at that run's pixel size,
    which the photon energy and the detector distance fix. Reusing it against a different
    geometry without resampling starts the reconstruction from an illumination of the
    wrong physical size. This compares the two pitches and, when they disagree,
    interpolates every coherent and incoherent mode onto the target grid.

    A probe carries no world center, so both grids are centered on their own arrays and
    the resampling separates into one monotone-cubic pass per axis. Content beyond the
    source is zero: outside the frame there is no illumination, which is the answer, not
    a filler. That is what distinguishes this from an object, where the same fill would
    assert an opaque border.

    Amplitudes scale by ``sqrt(target pixel area / source pixel area)`` so that
    ``sum(abs(P)**2)`` -- the photon count the probe carries -- comes through unchanged
    while the field it represents stays the same physical illumination.

    Returns ``probes`` itself when there is nothing to do: no pixel geometry to compare
    against, an axis too short to interpolate along, or pitches agreeing within
    ``rel_tol``. Returning the input itself keeps this a true no-op for callers that
    rebuild on every settings notification, which the interpolator by itself is not.

    Args:
        probes: Probe ensemble to re-express, typically an initial guess read from a file.
        target_geometry: Grid implied by the run's detector and illumination geometry.
        rel_tol: Largest relative pitch difference that still names the same sampling.
    """
    try:
        pixel_geometry = probes.get_pixel_geometry()
    except ValueError:
        logger.debug('Probe records no pixel size; taking it to be at the run sampling.')
        return probes

    width_agrees = math.isclose(
        pixel_geometry.width_m, target_geometry.pixel_width_m, rel_tol=rel_tol
    )
    height_agrees = math.isclose(
        pixel_geometry.height_m, target_geometry.pixel_height_m, rel_tol=rel_tol
    )

    if width_agrees and height_agrees:
        return probes

    if probes.width_px < 2 or probes.height_px < 2:
        logger.warning(
            f'Probe is {probes.width_px} x {probes.height_px} px and cannot be resampled to '
            'the run sampling; leaving it at its own pixel size.'
        )
        return probes

    logger.info(
        'Probe pixel size is %s x %s and the run samples at %s x %s; resampling from '
        '%d x %d to %d x %d px.',
        format_length(pixel_geometry.width_m),
        format_length(pixel_geometry.height_m),
        format_length(target_geometry.pixel_width_m),
        format_length(target_geometry.pixel_height_m),
        probes.width_px,
        probes.height_px,
        target_geometry.width_px,
        target_geometry.height_px,
    )

    columns = _centered_grid_indexes(
        pixel_geometry.width_m,
        probes.width_px,
        target_geometry.pixel_width_m,
        target_geometry.width_px,
    )
    rows = _centered_grid_indexes(
        pixel_geometry.height_m,
        probes.height_px,
        target_geometry.pixel_height_m,
        target_geometry.height_px,
    )

    resampled = probes.get_array()

    for indexes, axis in ((columns, -1), (rows, -2)):
        real = resample_along_axis(resampled.real, indexes, axis=axis, fill_value=0.0)
        imaginary = resample_along_axis(resampled.imag, indexes, axis=axis, fill_value=0.0)
        resampled = real + 1j * imaginary

    source_area_m2 = pixel_geometry.width_m * pixel_geometry.height_m
    target_area_m2 = target_geometry.pixel_width_m * target_geometry.pixel_height_m

    return ProbeSequence(
        array=resampled * numpy.sqrt(target_area_m2 / source_area_m2),
        opr_weights=probes.get_opr_weights_or_none(),
        pixel_geometry=target_geometry.get_pixel_geometry(),
    )


class OPRWeightPolicy(Enum):
    """How a loaded probe's OPR weights are reconciled with the current probe positions."""

    KEEP = auto()
    """Use the weights unchanged, and reject a probe sized for a different scan."""

    AVERAGE = auto()
    """Give every probe position the mean of the loaded weight rows.

    The coherent-mode basis survives and every position starts from the ensemble average
    of the scan that solved it, so the reconstruction refines per-position variation from
    a neutral start rather than from another scan's.
    """

    REINITIALIZE = auto()
    """Start the weights over: unit weight on the primary mode, noise on the rest.

    This is the state a fresh OPR run begins from. The loaded coherent modes are kept as
    a basis while the weights carry nothing from the scan that solved them.
    """

    COLLAPSE = auto()
    """Combine the coherent modes under the mean weight row into a single coherent mode.

    The result is the ensemble-average illumination the reconstruction solved for,
    expressed without an OPR basis.
    """

    DISCARD = auto()
    """Keep coherent mode 0 and drop the remaining modes along with the weights."""


def _pixel_geometry_or_none(probes: ProbeSequence) -> PixelGeometry | None:
    """Pixel geometry of ``probes``, or ``None`` when it records none."""
    try:
        return probes.get_pixel_geometry()
    except ValueError:
        return None


def conform_opr_weights(
    rng: numpy.random.Generator,
    probes: ProbeSequence,
    num_positions: int,
    policy: OPRWeightPolicy,
    *,
    small_value: float = 1.0e-6,
) -> ProbeSequence:
    """Re-express ``probes`` so its OPR weights describe ``num_positions`` probe positions.

    OPR weights carry one row per probe position, so a probe solved on one scan cannot
    initialize a reconstruction of another without a decision about what those rows mean
    there. ``policy`` names that decision; see :class:`OPRWeightPolicy`.

    The transform is unconditional: a probe whose weights already have ``num_positions``
    rows is still averaged, reinitialized or collapsed when asked. A caller that only
    wants to resolve a disagreement compares the row count itself and skips the call.

    A probe with no OPR weights and a single coherent mode names no basis to act on and
    is returned unchanged under every policy. With no weights but several coherent modes,
    :attr:`OPRWeightPolicy.COLLAPSE` has no mean row to weight by and reduces to
    :attr:`OPRWeightPolicy.DISCARD`, while :attr:`OPRWeightPolicy.REINITIALIZE` builds the
    weights the modes lack.

    :attr:`OPRWeightPolicy.COLLAPSE` applies the mean row across every incoherent mode,
    where :meth:`ProbeSequence.__getitem__` only ever rewrites incoherent mode 0 from the
    coherent basis.

    Args:
        rng: Random generator for :attr:`OPRWeightPolicy.REINITIALIZE`; unused otherwise.
        probes: Probe ensemble to re-express, typically an initial guess read from a file.
        num_positions: Number of probe positions the result must describe.
        policy: How the loaded weights carry over.
        small_value: Scale of the Gaussian noise :attr:`OPRWeightPolicy.REINITIALIZE` puts
            on the non-primary modes. Larger values let the reconstruction move off the
            primary mode sooner, at the cost of starting further from the probe the modes
            were solved for.

    Raises:
        ValueError: If ``num_positions`` is not positive, or if ``policy`` is
            :attr:`OPRWeightPolicy.KEEP` and the weights describe a different number of
            probe positions.
    """
    if num_positions < 1:
        raise ValueError(f'Number of probe positions must be positive; got {num_positions}!')

    array = probes.get_array()
    weights = probes.get_opr_weights_or_none()
    num_cmodes = probes.num_coherent_modes
    pixel_geometry = _pixel_geometry_or_none(probes)

    if weights is None and num_cmodes < 2:
        # Nothing here names an OPR basis, so there is nothing to reconcile.
        return probes

    match policy:
        case OPRWeightPolicy.KEEP:
            if weights is not None and weights.shape[0] != num_positions:
                raise ValueError(
                    'OPR weights describe a different scan!'
                    f' weight rows={weights.shape[0]}'
                    f' probe positions={num_positions}'
                )

            return probes
        case OPRWeightPolicy.AVERAGE if weights is not None:
            mean_row = weights.mean(axis=0)
            return ProbeSequence(
                array=array,
                opr_weights=numpy.broadcast_to(mean_row, (num_positions, num_cmodes)).copy(),
                pixel_geometry=pixel_geometry,
            )
        case OPRWeightPolicy.AVERAGE:
            # Several coherent modes, but no weight rows to average over.
            return probes
        case OPRWeightPolicy.REINITIALIZE:
            reinitialized = small_value * rng.normal(size=(num_positions, num_cmodes))
            reinitialized[:, 0] = 1.0
            return ProbeSequence(
                array=array,
                opr_weights=reinitialized,
                pixel_geometry=pixel_geometry,
            )
        case OPRWeightPolicy.COLLAPSE if weights is not None:
            mean_row = weights.mean(axis=0)
            return ProbeSequence(
                array=numpy.tensordot(mean_row, array, axes=1).astype(array.dtype),
                opr_weights=None,
                pixel_geometry=pixel_geometry,
            )
        case _:
            # DISCARD, and COLLAPSE with no mean row to weight by. Copying rather than
            # slicing keeps the discarded modes from staying resident behind a view of
            # the original buffer.
            return ProbeSequence(
                array=array[:1].copy(),
                opr_weights=None,
                pixel_geometry=pixel_geometry,
            )


class ProbeFileReader(ABC):
    """Plugin interface for reading probe sequences."""

    @abstractmethod
    def read(self, file_path: Path) -> ProbeSequence:
        """Read a probe sequence from file."""
        pass


class ProbeFileWriter(ABC):
    """Plugin interface for writing probe sequences."""

    @abstractmethod
    def write(self, file_path: Path, probes: ProbeSequence) -> None:
        """Write a probe sequence to file."""
        pass
