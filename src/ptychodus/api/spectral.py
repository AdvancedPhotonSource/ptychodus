"""The spectral covariance of ptychographic data, applied matrix-free.

The weighted covariance ``Z = sum_j sum_k S_jk^H diag(T(y_j)) S_jk`` of L. Valzania, J.
Dong and S. Gigan, Opt. Lett. 46, 1357 (2021), where ``S_jk`` places incoherent probe mode
``k`` at position ``j``, multiplies it into the object and propagates it to the detector.
:class:`SpectralCovariance` applies it as a :class:`scipy.sparse.linalg.LinearOperator`,
one forward pass, a weight and one adjoint pass per pattern, and measures how well an
object predicts the measured amplitudes; :class:`ShiftedProbeStack` supplies the placed
probe modes, and :class:`FourierSandwich` or :class:`PropagatorSandwich` the detector
propagator and its adjoint.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
import math

import numpy
from scipy.fft import fft2, fftshift, ifft2, ifftshift
from scipy.sparse.linalg import LinearOperator

from .assemble import DetectorCorrections
from .diffraction import DiffractionPatterns
from .object import ObjectGeometry
from .patch import accumulate_patches_on, compute_patch_pixel_indexes, shift_probe_onto_patches
from .probe import PatchBounds
from .propagate import FresnelTransformPropagator, Propagator
from .typing import BooleanArrayType, ComplexArrayType, IntegerArrayType, RealArrayType

__all__ = [
    'FourierSandwich',
    'PropagatorSandwich',
    'ShiftedProbeStack',
    'SpectralCovariance',
    'SpectralPreprocessing',
    'measure_corrected_intensities',
]


class SpectralPreprocessing(StrEnum):
    """How the spectral covariance weights each normalized intensity ``y``."""

    OPTIMAL = 'optimal'
    """``max(-beta, 1 - 1 / y)``: the preprocessing W. Luo, W. Alghamdi and Y. M. Lu,
    IEEE Trans. Signal Process. 67, 2347 (2019), prove uniformly optimal for noiseless
    intensities, floored at ``-beta`` as L. Valzania, J. Dong and S. Gigan, Opt. Lett. 46,
    1357 (2021), floor it."""
    THRESHOLD = 'threshold'
    """One for the brightest fraction of good pixels and zero for the rest; S. Marchesini,
    Y.-C. Tu and H.-T. Wu, Appl. Comput. Harmon. Anal. 41, 815 (2016)."""
    IDENTITY = 'identity'
    """The intensity itself; E. J. Candès, X. Li and M. Soltanolkotabi, IEEE Trans. Inf.
    Theory 61, 1985 (2015)."""


def _compute_spectral_weights(
    normalized: RealArrayType,
    *,
    preprocessing: SpectralPreprocessing,
    lower_bound: float,
    threshold: float,
) -> RealArrayType:
    """Map intensities normalized to unit mean through the preprocessing ``T(y)``.

    *lower_bound* is the floor of :attr:`SpectralPreprocessing.OPTIMAL` and *threshold*
    the normalized cut of :attr:`SpectralPreprocessing.THRESHOLD`; each preprocessing
    ignores the other's. Always returns a new array.
    """
    match preprocessing:
        case SpectralPreprocessing.OPTIMAL:
            # 1 - 1/y reaches -beta at y = 1/(1 + beta), so clamping y there first applies
            # the floor without dividing by the zeros photon counting records.
            floor = 1.0 / (1.0 + lower_bound)
            return 1.0 - 1.0 / numpy.maximum(normalized, floor)
        case SpectralPreprocessing.THRESHOLD:
            return (normalized > threshold).astype(numpy.float64)
        case SpectralPreprocessing.IDENTITY:
            return normalized.copy()


@dataclass(frozen=True)
class _IntensityScale:
    """What one pass over the corrected intensities measures."""

    mean: float
    """The mean over the good pixels of every pattern."""
    threshold: float
    """The normalized threshold, or zero without a threshold fraction."""
    mean_pattern: RealArrayType
    """The mean pattern, bad pixels included."""


def _measure_intensity_scale(
    intensity_chunks: Iterable[RealArrayType],
    good_pixels: BooleanArrayType,
    *,
    threshold_fraction: float | None,
) -> _IntensityScale:
    """Measure the mean over the good pixels of every chunk, the mean pattern, and a threshold.

    With *threshold_fraction*, the threshold is the intensity, divided by the mean, that
    the brightest fraction of good pixels exceeds, found over a float32 copy of every good
    pixel; without one it is zero and nothing is copied.

    Raises:
        ValueError: when the good pixels hold no signal.
    """
    total = 0.0
    num_good = 0
    num_patterns = 0
    pattern_sum = numpy.zeros(good_pixels.shape)
    good_values: list[RealArrayType] = []

    for intensities in intensity_chunks:
        values = intensities[:, good_pixels]
        total += float(numpy.sum(values))
        num_good += values.size
        num_patterns += intensities.shape[0]
        pattern_sum += numpy.sum(intensities, axis=0)

        if threshold_fraction is not None:
            good_values.append(values.astype(numpy.float32).ravel())

    mean = total / num_good if num_good > 0 else 0.0

    if not mean > 0.0:
        raise ValueError('The diffraction patterns hold no signal on their good pixels!')

    threshold = (
        float(numpy.quantile(numpy.concatenate(good_values), 1.0 - threshold_fraction)) / mean
        if threshold_fraction is not None
        else 0.0
    )
    return _IntensityScale(mean, threshold, pattern_sum / num_patterns)


def measure_corrected_intensities(
    chunk: slice,
    *,
    patterns: DiffractionPatterns,
    pattern_rows: IntegerArrayType,
    corrections: DetectorCorrections,
    i0_scale: RealArrayType,
) -> RealArrayType:
    """The paired patterns of *chunk*, corrected for the detector maps and scaled by I0."""
    corrected = corrections.apply(patterns[pattern_rows[chunk]])
    return corrected * i0_scale[chunk, numpy.newaxis, numpy.newaxis]


class PropagatorSandwich:
    """``D^H diag(w) D`` through the detector propagator and its adjoint, any regime."""

    def __init__(self, forward: Propagator, backward: Propagator) -> None:
        self._forward = forward
        self._backward = backward

    def prepare_probes(self, shifted: ComplexArrayType) -> ComplexArrayType:
        return shifted

    def prepare_weights(self, weights: RealArrayType) -> RealArrayType:
        return weights

    def apply(self, exit_waves: ComplexArrayType, weights: RealArrayType) -> ComplexArrayType:
        return self._backward.propagate(weights * self._forward.propagate(exit_waves))

    def predict_intensities(self, exit_waves: ComplexArrayType) -> RealArrayType:
        """``sum_k |D w_k|^2`` of prepared exit waves ``(n, modes, height, width)``."""
        return numpy.sum(numpy.square(numpy.abs(self._forward.propagate(exit_waves))), axis=1)


class FourierSandwich:
    """``D^H diag(w) D`` for a Fresnel-transform ``D``, as one FFT pair per wave.

    The output-plane phase cancels against the real weights and the shifts commute with
    the convolution left between them, so only the input-plane phase ``B`` remains: it
    is folded into the probes once, as the weights are into FFT order.
    """

    def __init__(self, propagator: FresnelTransformPropagator) -> None:
        self._input_plane_phase = propagator.get_input_plane_phase()

    def prepare_probes(self, shifted: ComplexArrayType) -> ComplexArrayType:
        return shifted * self._input_plane_phase

    def prepare_weights(self, weights: RealArrayType) -> RealArrayType:
        return ifftshift(weights, axes=(-2, -1))

    def apply(self, exit_waves: ComplexArrayType, weights: RealArrayType) -> ComplexArrayType:
        return ifft2(weights * fft2(exit_waves))

    def predict_intensities(self, exit_waves: ComplexArrayType) -> RealArrayType:
        """``sum_k |D w_k|^2`` of prepared exit waves ``(n, modes, height, width)``, up to
        a constant factor: the output-plane phase and the transform's centering are unit
        modulus, so only the input-plane phase the probes carry and the order remain."""
        intensities = numpy.sum(numpy.square(numpy.abs(fft2(exit_waves))), axis=1)
        return fftshift(intensities, axes=(-2, -1))


class ShiftedProbeStack:
    """The probe shifted onto every placement, computed once and kept or per request.

    Chunks bound the working memory of the shift either way. *prepare* maps every
    shifted stack before it is kept or returned, in single precision, which costs eight
    bytes a pixel and mode for every placement kept and halves the cost of every wave
    computed from it.
    """

    def __init__(
        self,
        probe_modes: ComplexArrayType,
        bounds: Sequence[PatchBounds],
        chunks: Sequence[slice],
        *,
        prepare: Callable[[ComplexArrayType], ComplexArrayType],
        cached: bool,
    ) -> None:
        self._probe_modes = probe_modes
        self._bounds = bounds
        self._prepare = prepare
        self._cached = (
            numpy.concatenate([self._shift(chunk) for chunk in chunks]) if cached else None
        )

    def _shift(self, chunk: slice) -> ComplexArrayType:
        shifted = shift_probe_onto_patches(self._probe_modes, self._bounds[chunk])
        return self._prepare(shifted).astype(numpy.complex64)

    def get(self, chunk: slice) -> ComplexArrayType:
        """The shifted modes of the placements in *chunk*, ``(n, modes, height, width)``."""
        if self._cached is None:
            return self._shift(chunk)
        else:
            return self._cached[chunk]


class SpectralCovariance(LinearOperator):
    """The weighted covariance ``sum_j sum_k S_jk^H diag(T(y_j)) S_jk``, matrix-free.

    ``S_jk`` places mode ``k`` of the shifted probe at placement ``j``, multiplies it into
    the object and propagates it to the detector. The intensities are normalized to unit
    mean over the good pixels when the operator is built; their weights, zero at every bad
    pixel, are kept as float32 when `cache_weights` asks and recomputed per product
    otherwise. ``Z`` is Hermitian, so its adjoint is itself. `num_products` counts the
    products taken, and `mean_pattern` is the mean of the intensities.
    """

    def __init__(
        self,
        *,
        intensities: Callable[[slice], RealArrayType],
        chunks: Sequence[slice],
        bounds: Sequence[PatchBounds],
        geometry: ObjectGeometry,
        probes: ShiftedProbeStack,
        sandwich: PropagatorSandwich | FourierSandwich,
        good_pixels: BooleanArrayType,
        preprocessing: SpectralPreprocessing,
        lower_bound: float,
        threshold_fraction: float,
        cache_weights: bool,
    ) -> None:
        num_pixels = geometry.height_px * geometry.width_px
        super().__init__(numpy.complex128, (num_pixels, num_pixels))
        self._intensities = intensities
        self._chunks = chunks
        self._bounds = bounds
        self._geometry = geometry
        self._probes = probes
        self._sandwich = sandwich
        self._bad_pixels = numpy.logical_not(good_pixels)
        self._preprocessing = preprocessing
        self._lower_bound = lower_bound
        scale = _measure_intensity_scale(
            (intensities(chunk) for chunk in chunks),
            good_pixels,
            threshold_fraction=(
                threshold_fraction if preprocessing == SpectralPreprocessing.THRESHOLD else None
            ),
        )
        self._mean = scale.mean
        self._threshold = scale.threshold
        self.mean_pattern = scale.mean_pattern
        self._cached_weights = (
            [self._compute_weights(chunk).astype(numpy.float32) for chunk in chunks]
            if cache_weights
            else None
        )
        self.num_products = 0

    def _compute_weights(self, chunk: slice) -> RealArrayType:
        weights = _compute_spectral_weights(
            self._intensities(chunk) / self._mean,
            preprocessing=self._preprocessing,
            lower_bound=self._lower_bound,
            threshold=self._threshold,
        )
        weights[:, self._bad_pixels] = 0.0
        return self._sandwich.prepare_weights(weights)

    def _weights(self, chunk_index: int) -> RealArrayType:
        if self._cached_weights is None:
            return self._compute_weights(self._chunks[chunk_index])
        else:
            return self._cached_weights[chunk_index]

    def _matvec(self, x: ComplexArrayType) -> ComplexArrayType:
        self.num_products += 1
        canvas = numpy.ravel(x)
        result = numpy.zeros(
            (self._geometry.height_px, self._geometry.width_px), dtype=numpy.complex128
        )

        for chunk_index, chunk in enumerate(self._chunks):
            pixel_indexes = compute_patch_pixel_indexes(self._bounds[chunk], self._geometry)
            shifted = self._probes.get(chunk)
            weights = self._weights(chunk_index)[:, numpy.newaxis]
            values = canvas[pixel_indexes].astype(numpy.complex64)
            exit_waves = shifted * values[:, numpy.newaxis]
            back = self._sandwich.apply(exit_waves, weights)
            update = numpy.sum(numpy.conj(shifted) * back, axis=1)
            result += accumulate_patches_on(update, pixel_indexes, self._geometry)

        return result.ravel()

    def _rmatvec(self, x: ComplexArrayType) -> ComplexArrayType:
        return self._matvec(x)

    def measure_amplitude_residual(self, estimate: ComplexArrayType) -> float:
        """``sum_j ||sqrt(y_j) - s |S_j O||||^2 / sum_j y_j`` over the good pixels.

        ``|S_j O|`` is the amplitude the canvas *estimate* predicts at placement ``j``,
        summed over the modes in quadrature, and ``s`` the scale that fits it best, which
        also absorbs any constant factor of the propagator. The intensities are the
        corrected ones the operator was built from, before normalization. Infinite when
        either the prediction or the data holds no signal.
        """
        canvas = numpy.ravel(estimate)
        cross = model_power = data_power = 0.0

        for chunk in self._chunks:
            pixel_indexes = compute_patch_pixel_indexes(self._bounds[chunk], self._geometry)
            values = canvas[pixel_indexes].astype(numpy.complex64)
            exit_waves = self._probes.get(chunk) * values[:, numpy.newaxis]
            predicted = numpy.sqrt(
                self._sandwich.predict_intensities(exit_waves)[:, ~self._bad_pixels],
                dtype=numpy.float64,
            )
            measured = numpy.sqrt(
                numpy.maximum(self._intensities(chunk)[:, ~self._bad_pixels], 0.0),
                dtype=numpy.float64,
            )
            cross += float(numpy.sum(predicted * measured))
            model_power += float(numpy.sum(numpy.square(predicted)))
            data_power += float(numpy.sum(numpy.square(measured)))

        if not (model_power > 0.0 and data_power > 0.0):
            return math.inf

        # With s = cross / model_power fitted, the misfit is data_power - cross^2 / model_power.
        return max(1.0 - cross * cross / (model_power * data_power), 0.0)
