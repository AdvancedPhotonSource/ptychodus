"""Probe (illumination function) data structures and file I/O plugin interfaces."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy
import scipy.ndimage
from scipy.fft import fft2

from .constants import format_length
from .geometry import ImageExtent, PixelGeometry
from .preprocess.noise import estimate_noise_floor
from .propagate import intensity
from .typing import ComplexArrayType, RealArrayType


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
       (matching :func:`ptychodus.api.preprocess.diffraction.estimate_crop_center`).
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
        width_px = image_extent.width_px
        height_px = image_extent.height_px
        numerator_m2 = wavelength_m * abs(distance_m)
        return cls(
            width_px=width_px,
            height_px=height_px,
            pixel_width_m=numerator_m2 / (detector_pixel_geometry.width_m * width_px),
            pixel_height_m=numerator_m2 / (detector_pixel_geometry.height_m * height_px),
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
