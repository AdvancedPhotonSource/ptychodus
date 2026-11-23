"""Estimates of the probe's defocus from far-field diffraction data.

Two estimators, the second refining the first. :func:`estimate_defocus_from_parallax` fits
the defocus to the shifts between bright-field images that
:func:`~ptychodus.api.parallax.measure_parallax_shifts` measures, in one binned pass over
the patterns. :func:`estimate_defocus_by_autofocus` asks every pattern which defocus makes
the average-pattern probe explain the measured amplitudes best.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
import logging
import math

import numpy
from scipy.fft import set_workers

from .accumulate import DiffractionMoments, compute_diffraction_moments, estimate_incident_beam
from .assemble import AssembledDiffractionData, DetectorCorrections
from .geometry import ImageExtent
from .object import ObjectGeometry
from .parallax import measure_parallax_shifts
from .patch import (
    accumulate_probe_intensity_on,
    compute_patch_pixel_indexes,
    resolve_probe_patch_bounds_on,
    shift_probe_onto_patches,
)
from .preprocess.diffraction import BinningStep
from .probe import Probe, ProbeEnsemble, ProbeGeometry, estimate_probe_extent
from .probe_positions import ProbePosition, ProbePositionSequence
from .propagate import FresnelTransformPropagator, PropagatorParameters
from .simulate.object import compute_stxm_dpc_signals, generate_stxm_dpc_object
from .simulate.probe import generate_average_pattern_probe
from .spectral import (
    FourierSandwich,
    ShiftedProbeStack,
    SpectralCovariance,
    SpectralPreprocessing,
    measure_corrected_intensities,
)
from .typing import IntegerArrayType, RealArrayType

__all__ = [
    'AutofocusEstimate',
    'ParallaxEstimate',
    'estimate_defocus_by_autofocus',
    'estimate_defocus_from_assembled_data',
    'estimate_defocus_from_parallax',
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ParallaxEstimate:
    """A defocus fitted to the parallax of `num_bins` bright-field images.

    `standard_error_m` is the defocus's standard error from the scatter of the shifts
    about the fitted line; it is zero when the shifts lie on it exactly.
    `depth_of_focus_m` is the probe's ``lambda / NA**2``, from the second moments of the
    mean pattern as :attr:`ProbeExtent.depth_of_focus_m` measures it. Within about that
    distance of focus the parallax is not a shift, so a fitted defocus no larger than
    it is not evidence of one.
    """

    defocus_distance_m: float
    standard_error_m: float
    depth_of_focus_m: float
    num_bins: int

    @property
    def significance(self) -> float:
        """The defocus in units of its standard error; zero for a defocus of zero."""
        if self.defocus_distance_m == 0.0:
            return 0.0

        if self.standard_error_m > 0.0:
            return abs(self.defocus_distance_m) / self.standard_error_m

        return math.inf

    @property
    def fresnel_number(self) -> float:
        """The defocus in depths of focus, as :attr:`ProbeExtent.fresnel_number`."""
        return abs(self.defocus_distance_m) / self.depth_of_focus_m

    def is_resolved(
        self, *, minimum_significance: float = 3.0, minimum_defocus_dof: float = 1.0
    ) -> bool:
        """Whether the defocus is both far enough from focus and statistically real.

        *minimum_defocus_dof* is the number of depths of focus the defocus must reach,
        and *minimum_significance* the number of its standard errors.
        """
        return (
            self.fresnel_number >= minimum_defocus_dof and self.significance >= minimum_significance
        )


def estimate_defocus_from_assembled_data(
    data: AssembledDiffractionData,
    probe_positions: ProbePositionSequence,
    object_geometry: ObjectGeometry,
    probe_geometry: ProbeGeometry,
    *,
    photon_wavelength_m: float,
    i0_photons_per_count: float = 0.0,
    bin_size_px: int = 4,
    max_bins: int = 64,
    chunk_bins: int = 8,
) -> ParallaxEstimate | None:
    """Estimate the defocus from assembled far-field patterns, in one binned pass.

    The patterns are binned *bin_size_px* detector pixels a side as their moments are
    accumulated, with the data's own bad pixels, background and flat field; the detector sums are
    corrected for beam decay by the I0 the positions carry, converted at
    *i0_photons_per_count* (see :func:`estimate_incident_beam`); and the corrected moments
    go to :func:`estimate_defocus_from_parallax`, which keeps at most *max_bins* of the
    bright-field bins and maps them *chunk_bins* at a time. Larger bins cost less memory
    but leave fewer inside the bright field.

    Returns None when there are no patterns or the parallax gives no estimate.

    Raises:
        ValueError: where :func:`estimate_defocus_from_parallax` refuses the inputs.
    """
    moments = compute_diffraction_moments(
        data.get_indexes(),
        data.get_patterns(),
        bad_pixels=data.get_bad_pixels(),
        background=data.get_background(),
        flat_field=data.get_flat_field(),
        binning=BinningStep(bin_size_x=bin_size_px, bin_size_y=bin_size_px),
    )

    if moments is None:
        return None

    corrected = estimate_incident_beam(
        moments,
        probe_positions.get_i0_by_index(),
        i0_photons_per_count=i0_photons_per_count,
    ).moments
    return estimate_defocus_from_parallax(
        corrected,
        probe_positions,
        object_geometry,
        probe_geometry,
        photon_wavelength_m=photon_wavelength_m,
        max_bins=max_bins,
        chunk_bins=chunk_bins,
    )


def estimate_defocus_from_parallax(
    moments: DiffractionMoments,
    probe_positions: ProbePositionSequence,
    object_geometry: ObjectGeometry,
    probe_geometry: ProbeGeometry,
    *,
    photon_wavelength_m: float,
    bright_field_fraction: float = 0.5,
    upsample_factor: int = 20,
    minimum_bins: int = 4,
    max_bins: int = 64,
    chunk_bins: int = 8,
) -> ParallaxEstimate | None:
    """Estimate the probe's defocus from the shifts between its bright-field images.

    The shifts are measured by :func:`~ptychodus.api.parallax.measure_parallax_shifts`,
    which states what *moments* must hold and what *bright_field_fraction*,
    *upsample_factor*, *minimum_bins*, *max_bins* and *chunk_bins* do. Then:

    1. The shifts are fitted by least squares to ``s_b = a - lambda * z * q_b``. The
       offset ``a``, shared by every bin, absorbs both a common shift and wherever the
       beam centre lies, so ``q_b`` is measured from the DC pixel ``N // 2``.
    2. The depth of focus is :func:`~ptychodus.api.probe.estimate_probe_extent`'s, from
       the mean pattern.

    A featureless sample gives shifts that are noise about zero, and a probe within the
    depth of focus gives a defocus no larger than it; judge the result by
    :meth:`ParallaxEstimate.is_resolved`.

    Returns None when fewer than *minimum_bins* bright-field bins remain or fewer than
    three positions pair with a pattern.

    Raises:
        ValueError: if *moments* holds no binned patterns, its patterns do not match
            *probe_geometry*, or *max_bins* is below *minimum_bins*.
    """
    measured = measure_parallax_shifts(
        moments,
        probe_positions,
        object_geometry,
        probe_geometry,
        bright_field_fraction=bright_field_fraction,
        upsample_factor=upsample_factor,
        minimum_bins=minimum_bins,
        max_bins=max_bins,
        chunk_bins=chunk_bins,
    )

    if measured is None:
        return None

    q_per_m, shifts_m = measured
    num_bins = q_per_m.shape[0]
    q_x_per_m, q_y_per_m = q_per_m[:, 0], q_per_m[:, 1]
    shift_x_m, shift_y_m = shifts_m[:, 0], shifts_m[:, 1]

    # Unknowns (a_x, a_y, c), one equation per bin and axis, with c = -lambda * z.
    design = numpy.zeros((2 * num_bins, 3))
    design[:num_bins, 0] = 1.0
    design[:num_bins, 2] = q_x_per_m
    design[num_bins:, 1] = 1.0
    design[num_bins:, 2] = q_y_per_m
    observed = numpy.concatenate((shift_x_m, shift_y_m))
    solution, _, _, _ = numpy.linalg.lstsq(design, observed, rcond=None)
    residual = observed - design @ solution
    variance = float(residual @ residual) / (2 * num_bins - 3)
    slope_variance = variance * float(numpy.linalg.inv(design.T @ design)[2, 2])
    defocus_distance_m = float(-solution[2] / photon_wavelength_m)

    return ParallaxEstimate(
        defocus_distance_m=defocus_distance_m,
        standard_error_m=math.sqrt(slope_variance) / photon_wavelength_m,
        depth_of_focus_m=estimate_probe_extent(
            moments.mean_pattern,
            probe_geometry,
            photon_wavelength_m=photon_wavelength_m,
            defocus_distance_m=defocus_distance_m,
            bad_pixels=moments.bad_pixels,
        ).depth_of_focus_m,
        num_bins=num_bins,
    )


@dataclass(frozen=True)
class AutofocusEstimate:
    """A defocus chosen by :func:`estimate_defocus_by_autofocus`, with what it measured."""

    defocus_distance_m: float
    """The defocus chosen, or the initial one where autofocus kept it."""
    candidates_m: tuple[float, ...]
    """Every defocus evaluated, in order of distance from focus; empty where none was."""
    residuals: tuple[float, ...]
    """The normalized amplitude residual at each candidate."""
    num_patterns: int
    """How many patterns the residuals were measured over."""
    kept_initial: bool
    """Whether the initial defocus was kept: it, or the residual's minimum, lay too near focus."""
    on_edge: bool
    """Whether the minimum lay on the outermost candidate, short of a bracketed minimum."""


def _select_central_rows(positions: ProbePositionSequence, fraction: float) -> IntegerArrayType:
    """The rows of *positions* inside a box centred on the scan holding *fraction* of its area."""
    xy = numpy.array([[position.x_m, position.y_m] for position in positions])

    if fraction >= 1.0:
        return numpy.arange(len(xy))

    centre = 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    half = 0.5 * (xy.max(axis=0) - xy.min(axis=0)) * math.sqrt(fraction)
    return numpy.flatnonzero(numpy.all(numpy.abs(xy - centre) <= half, axis=1))


def _fit_parabola_vertex(x: RealArrayType, y: RealArrayType) -> float:
    """The vertex abscissa of the parabola through three points, clipped to their span."""
    x0, x1, x2 = (float(v) for v in x)
    y0, y1, y2 = (float(v) for v in y)
    denominator = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denominator
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denominator

    if a <= 0.0:
        return x1

    return min(max(-b / (2.0 * a), x0), x2)


class _AutofocusResidual:
    """The amplitude residual of a one-product spectral refinement, for any probe.

    Built once from what every candidate shares: the paired patterns of a central block of
    positions, their placements, the STXM-DPC start restricted to the illuminated region,
    and the detector propagator.
    """

    def __init__(
        self,
        assembled_data: AssembledDiffractionData,
        probe_positions: Iterable[ProbePosition],
        geometry: ObjectGeometry,
        start_probe: Probe,
        *,
        photon_wavelength_m: float,
        detector_distance_m: float,
        max_patterns: int,
        support_fraction: float,
        max_patterns_per_chunk: int,
    ) -> None:
        indexes = numpy.asarray(assembled_data.get_indexes())
        paired, positions = ProbePositionSequence(list(probe_positions)).interpolate_at_indexes(
            indexes
        )

        if not paired.any():
            raise ValueError('No probe position pairs with any diffraction pattern index!')

        rows = numpy.flatnonzero(paired)
        keep = _select_central_rows(positions, min(1.0, max_patterns / rows.size))
        self._pattern_rows = rows[keep]
        self._positions = ProbePositionSequence([positions[i] for i in keep])
        self._geometry = geometry
        self._chunks = [
            slice(begin, min(begin + max_patterns_per_chunk, self._pattern_rows.size))
            for begin in range(0, self._pattern_rows.size, max_patterns_per_chunk)
        ]
        height_px, width_px = assembled_data.get_patterns().shape[-2:]
        pixel_geometry = start_probe.get_pixel_geometry()
        self._propagator = FresnelTransformPropagator(
            PropagatorParameters(
                wavelength_m=photon_wavelength_m,
                width_px=width_px,
                height_px=height_px,
                pixel_width_m=pixel_geometry.width_m,
                pixel_height_m=pixel_geometry.height_m,
                propagation_distance_m=detector_distance_m,
            )
        )
        self._sandwich = FourierSandwich(self._propagator)
        ensemble = ProbeEnsemble.from_probe(start_probe)
        self._bounds = resolve_probe_patch_bounds_on(
            ensemble.get_geometry(), self._positions, geometry
        )
        illumination = accumulate_probe_intensity_on(ensemble, self._positions, geometry)
        self._region = illumination >= support_fraction * illumination.max()
        start = generate_stxm_dpc_object(
            geometry,
            compute_stxm_dpc_signals(assembled_data),
            probe_positions,
            detector_extent=ImageExtent(width_px=width_px, height_px=height_px),
            probe=start_probe,
        ).get_layers_flattened()
        self._start = numpy.where(self._region, start, 0.0).astype(numpy.complex128)
        self._start_modulus = numpy.abs(start)
        corrections = DetectorCorrections(
            background=assembled_data.get_background(),
            flat_field=assembled_data.get_flat_field(),
            subtract_background=True,
            divide_flat_field=True,
        )
        self._good_pixels = numpy.logical_not(
            corrections.apply_to_bad_pixels(assembled_data.get_bad_pixels())
        )
        self._intensities = partial(
            measure_corrected_intensities,
            patterns=assembled_data.get_patterns(),
            pattern_rows=self._pattern_rows,
            corrections=corrections,
            i0_scale=numpy.ones(self._pattern_rows.size),
        )

    @property
    def num_patterns(self) -> int:
        return int(self._pattern_rows.size)

    def measure(self, probe: Probe) -> float:
        """``sum ||sqrt(y) - s |D(P O)|||^2 / sum y`` over the good pixels, ``s`` fitted, where
        ``O`` keeps the start's modulus and takes its phase from one product with *probe*."""
        modes = probe.get_array()
        covariance = SpectralCovariance(
            intensities=self._intensities,
            chunks=self._chunks,
            bounds=self._bounds,
            geometry=self._geometry,
            probes=ShiftedProbeStack(
                modes,
                self._bounds,
                self._chunks,
                prepare=self._sandwich.prepare_probes,
                cached=True,
            ),
            sandwich=self._sandwich,
            good_pixels=self._good_pixels,
            preprocessing=SpectralPreprocessing.IDENTITY,
            lower_bound=0.0,
            threshold_fraction=1.0,
            weight_offset=0.0,
            cache_weights=True,
        )
        refined = covariance.matvec(self._start.ravel()).reshape(self._start.shape)
        estimate = numpy.where(
            self._region, self._start_modulus * numpy.exp(1j * numpy.angle(refined)), 1.0
        ).ravel()
        cross = model_power = data_power = 0.0
        amplitudes = []

        for chunk in self._chunks:
            pixel_indexes = compute_patch_pixel_indexes(self._bounds[chunk], self._geometry)
            exit_waves = (
                shift_probe_onto_patches(modes, self._bounds[chunk])
                * estimate[pixel_indexes][:, numpy.newaxis]
            )
            predicted = numpy.sqrt(
                numpy.sum(numpy.square(numpy.abs(self._propagator.propagate(exit_waves))), axis=1)
            )[:, self._good_pixels]
            measured = numpy.sqrt(numpy.maximum(self._intensities(chunk), 0.0))[
                :, self._good_pixels
            ]
            cross += float(numpy.sum(predicted * measured))
            model_power += float(numpy.sum(predicted * predicted))
            data_power += float(numpy.sum(measured * measured))
            amplitudes.append((predicted, measured))

        if not (model_power > 0.0 and data_power > 0.0):
            return math.inf

        scale = cross / model_power
        misfit = sum(float(numpy.sum(numpy.square(m - scale * p))) for p, m in amplitudes)
        return misfit / data_power


def estimate_defocus_by_autofocus(
    assembled_data: AssembledDiffractionData,
    probe_positions: Iterable[ProbePosition],
    object_geometry: ObjectGeometry,
    probe_geometry: ProbeGeometry,
    *,
    photon_wavelength_m: float,
    detector_distance_m: float,
    initial_defocus_m: float,
    depth_of_focus_m: float,
    num_candidates: int = 7,
    span_factor: float = 3.0,
    min_defocus_dof: float = 3.0,
    max_patterns: int = 2500,
    support_fraction: float = 0.05,
    max_patterns_per_chunk: int = 64,
    workers: int = -1,
) -> AutofocusEstimate:
    """Refine a far-field defocus to the one whose average-pattern probe best explains the data.

    Candidates lie on a log grid of `num_candidates` defoci from ``initial / span_factor``
    to ``initial * span_factor``, the sign of *initial_defocus_m* (typically the parallax
    estimate). For each the average-pattern probe is generated at that defocus
    (:func:`~ptychodus.api.simulate.probe.generate_average_pattern_probe`), one product of
    the spectral covariance (:class:`~ptychodus.api.spectral.SpectralCovariance`, identity weights) is
    taken from the STXM-DPC estimate built once with the initial probe, and the refined
    object -- the start's modulus, the product's phase -- is scored by its amplitude
    residual ``sum_j ||sqrt(y_j) - s |D(P O_j)|||^2 / sum_j y_j`` with the scale ``s``
    fitted. Two more candidates halfway (in log) to the best one's neighbours refine it,
    and a parabola in log defocus through the best three gives the estimate. More than one
    product sharpens the probe's errors rather than the defocus, and the spectral Rayleigh
    quotient is no criterion: it favours a compact probe whatever the defocus.

    The initial defocus is kept, and nothing evaluated, when it lies within one
    *depth_of_focus_m* of focus; it is also kept when the residual's minimum lies within
    `min_defocus_dof` depths of focus of focus, where the residual is too flat to place the
    defocus better than the initial estimate does. A minimum on the outermost candidate is
    returned, flagged in :attr:`AutofocusEstimate.on_edge`.

    The residuals are measured over the central block of the scan holding about
    `max_patterns` patterns, whatever the detector crop, so the cost grows with the pixels
    per pattern; `max_patterns_per_chunk` bounds the working memory and `workers` is the number of FFT
    threads, as :func:`scipy.fft.set_workers` takes it. Detector background and flat field
    are corrected as the spectral refinement corrects them; I0 is not applied.

    Raises:
        ValueError: if the detector distance or depth of focus is not positive, a tunable is
            out of range, or no position pairs with any pattern.
    """
    if detector_distance_m == 0.0:
        raise ValueError('Autofocus needs a nonzero detector distance!')

    if not depth_of_focus_m > 0.0:
        raise ValueError('Autofocus needs a positive depth of focus!')

    if num_candidates < 3 or not span_factor > 1.0 or max_patterns < 1:
        raise ValueError(
            'Autofocus needs at least 3 candidates, a span above 1 and at least one pattern!'
        )

    if abs(initial_defocus_m) <= depth_of_focus_m:
        return AutofocusEstimate(initial_defocus_m, (), (), 0, kept_initial=True, on_edge=False)

    average_pattern_probe = partial(
        generate_average_pattern_probe,
        probe_geometry,
        assembled_data.get_mean_pattern(),
        bad_pixels=assembled_data.get_bad_pixels(),
        photon_wavelength_m=photon_wavelength_m,
        detector_distance_m=detector_distance_m,
        detector_pixel_geometry=assembled_data.get_pixel_geometry(),
        background=assembled_data.get_background(),
        flat_field=assembled_data.get_flat_field(),
    )
    log_span = math.log(span_factor)
    log_grid = math.log(abs(initial_defocus_m)) + numpy.linspace(
        -log_span, log_span, num_candidates
    )
    sign = math.copysign(1.0, initial_defocus_m)

    with set_workers(workers):
        residual = _AutofocusResidual(
            assembled_data,
            probe_positions,
            object_geometry,
            average_pattern_probe(defocus_distance_m=initial_defocus_m),
            photon_wavelength_m=photon_wavelength_m,
            detector_distance_m=detector_distance_m,
            max_patterns=max_patterns,
            support_fraction=support_fraction,
            max_patterns_per_chunk=max_patterns_per_chunk,
        )
        values = [
            residual.measure(average_pattern_probe(defocus_distance_m=sign * math.exp(x)))
            for x in log_grid
        ]
        best = int(numpy.argmin(values))

        if 0 < best < num_candidates - 1:
            # Halfway (in log) to each neighbour of the best candidate.
            extra = 0.5 * (log_grid[best] + log_grid[[best - 1, best + 1]])
            values += [
                residual.measure(average_pattern_probe(defocus_distance_m=sign * math.exp(x)))
                for x in extra
            ]
            log_grid = numpy.concatenate([log_grid, extra])

    order = numpy.argsort(log_grid)
    log_grid = log_grid[order]
    residuals = numpy.asarray(values)[order]
    best = int(numpy.argmin(residuals))
    on_edge = best in (0, len(log_grid) - 1)

    if on_edge:
        chosen = sign * math.exp(float(log_grid[best]))
    else:
        window = slice(best - 1, best + 2)
        chosen = sign * math.exp(_fit_parabola_vertex(log_grid[window], residuals[window]))

    kept_initial = abs(chosen) < min_defocus_dof * depth_of_focus_m
    logger.debug(f'Autofocus residuals {residuals} at defocus {sign * numpy.exp(log_grid)} m.')

    return AutofocusEstimate(
        defocus_distance_m=initial_defocus_m if kept_initial else chosen,
        candidates_m=tuple(float(sign * math.exp(x)) for x in log_grid),
        residuals=tuple(float(r) for r in residuals),
        num_patterns=residual.num_patterns,
        kept_initial=kept_initial,
        on_edge=on_edge,
    )
