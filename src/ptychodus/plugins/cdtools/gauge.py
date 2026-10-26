"""Measure the scalar that carries a probe between ptychodus's gauge and cdtools'.

ptychodus stores a probe in photons per pixel, so an energy-conserving propagator takes
it to a detector total equal to the measured counts. cdtools' far field is an
orthonormal FFT, so the factor is one for a probe already matched to its data -- but it
is still measured rather than assumed, which is what catches a probe that arrived in
some other gauge, such as a warm start from a product written before ptychodus adopted
its own or one from another tool.

**A deliberate copy.** This is pty-chi's measurement in
:mod:`ptychodus.plugins.ptychi.task`, adapted rather than shared: it takes the forward
propagator as a callable instead of building one, has no corner-ordered plane to roll,
since cdtools' far field is centered in and out, and multiplies the prediction by a flat
field, which FancyPtycho models and pty-chi does not. A fix to either belongs in both.

Pure numpy apart from the callable, so it is testable without cdtools installed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import logging
import math

import numpy

from ptychodus.api.assemble import compute_total_counts
from ptychodus.api.diffraction import FlatField
from ptychodus.api.patch import iter_probe_patches
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.product import Product
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.api.typing import BooleanArrayType, ComplexArrayType, RealArrayType

__all__ = [
    'ProbeGauge',
    'PropagateIntensity',
    'measure_probe_gauge',
]

logger = logging.getLogger(__name__)

PropagateIntensity = Callable[[ComplexArrayType], RealArrayType]
"""Detector-plane intensity of a stack of exit-wave modes, one plane per mode."""


@dataclass(frozen=True)
class ProbeGauge:
    """The scalar that carries a probe between ptychodus's gauge and cdtools'.

    `to_cdtools` multiplies a probe on the way in and divides it on the way out. Only the
    probe moves: the object is left alone, which is what keeps the pair consistent, since
    the product of the two is what the data constrains.

    `expected_factor` is what the propagator alone implies, carried for diagnostics.
    `anomaly` is how far the measurement strays from it -- one for a probe already
    matched to its data, and far from one when the probe is in a foreign gauge.
    """

    to_cdtools: float
    expected_factor: float
    sampled_positions: int

    @property
    def anomaly(self) -> float:
        return self.to_cdtools / self.expected_factor


def _measure_propagator_energy_ratio(
    propagate_intensity: PropagateIntensity, probe: ProbeEnsemble
) -> float:
    """Factor by which the propagator scales total intensity, from one flat field."""
    probe_field = numpy.ones((1, probe.height_px, probe.width_px), dtype=complex)
    return float(numpy.sum(propagate_intensity(probe_field)) / probe_field.size)


def _measure_predicted_intensity(
    propagate_intensity: PropagateIntensity,
    product: Product,
    good_pixels: BooleanArrayType,
    flat_field: FlatField | None,
    num_sample_positions: int,
) -> tuple[float, int]:
    """Mean detector total the forward model would predict, over sampled positions.

    The exit wave is the probe times the object patch beneath it, incoherent modes
    summed after propagation, and multiplied by the flat field when the model applies
    one. Only the good pixels are summed, because the measured totals it will be
    compared against are summed the same way.
    """
    object_array = product.object_.get_layers_flattened()
    num_positions = len(product.probe_positions)

    if num_positions == 0:
        return 0.0, 0

    stride = max(1, num_positions // num_sample_positions)
    total = 0.0
    sampled = 0

    for index, patch in enumerate(iter_probe_patches(product)):
        if index % stride != 0:
            continue

        object_patch = object_array[patch.bounds.y_slice, patch.bounds.x_slice]
        intensity = numpy.sum(propagate_intensity(patch.modes * object_patch), axis=0)

        if flat_field is not None:
            intensity = intensity * flat_field

        total += float(numpy.sum(intensity[good_pixels]))
        sampled += 1

    if sampled == 0:
        return 0.0, 0

    return total / sampled, sampled


def _measure_detector_intensity(
    parameters: ReconstructInput, good_pixels: BooleanArrayType
) -> float:
    """Mean per-pattern total the detector actually recorded, over good pixels.

    A carried background has not been applied to the patterns, so it is subtracted here:
    left in, it would inflate the measurement and shrink the probe to compensate.
    """
    total_counts = compute_total_counts(parameters.diffraction_patterns, parameters.bad_pixels)

    if total_counts.size == 0:
        return 0.0

    measured = float(numpy.mean(total_counts))

    if parameters.background is not None:
        measured -= float(numpy.sum(numpy.asarray(parameters.background)[good_pixels]))

    return measured


def measure_probe_gauge(
    parameters: ReconstructInput,
    propagate_intensity: PropagateIntensity,
    *,
    flat_field: FlatField | None = None,
    num_sample_positions: int = 32,
) -> ProbeGauge:
    """Measure the factor that carries this product's probe into cdtools' gauge.

    *propagate_intensity* is the forward model's own far field. *flat_field* is the
    detector response the model multiplies its prediction by, or None when it applies
    none. *num_sample_positions* is how many positions, spread evenly over the scan, the
    prediction is averaged over; more costs one propagation each.

    Degrades to the propagator's own factor, which is right whenever the probe and
    object already agree with the data, rather than raising: an initial guess is not
    worth failing a reconstruction over.
    """
    energy_ratio = _measure_propagator_energy_ratio(propagate_intensity, parameters.product.probes)
    expected_factor = 1.0 / math.sqrt(energy_ratio) if energy_ratio > 0.0 else 1.0

    good_pixels = numpy.logical_not(parameters.bad_pixels)
    measured = _measure_detector_intensity(parameters, good_pixels)

    try:
        predicted, sampled = _measure_predicted_intensity(
            propagate_intensity,
            parameters.product,
            good_pixels,
            flat_field,
            num_sample_positions,
        )
    except ValueError as exc:
        logger.warning(
            'Could not sample the probe against the object (%s); scaling the probe by the'
            ' propagator convention alone.',
            exc,
        )
        predicted, sampled = 0.0, 0

    if predicted <= 0.0 or measured <= 0.0:
        logger.warning(
            'Probe gauge measurement is unusable (predicted=%.6g, measured=%.6g); scaling'
            ' the probe by the propagator convention alone.',
            predicted,
            measured,
        )
        return ProbeGauge(expected_factor, expected_factor, 0)

    gauge = ProbeGauge(math.sqrt(measured / predicted), expected_factor, sampled)
    anomaly_warn_factor = 10.0

    if not 1.0 / anomaly_warn_factor < gauge.anomaly < anomaly_warn_factor:
        logger.warning(
            'Probe gauge is %.6g where this geometry implies %.6g (a factor of %.3g).'
            ' The probe may be in another gauge -- a product written by an older'
            ' ptychodus -- or come from another tool. Scaling by the measured value'
            ' anyway.',
            gauge.to_cdtools,
            expected_factor,
            gauge.anomaly,
        )
    else:
        logger.info(
            'Probe gauge %.6g (convention %.6g, anomaly %.4g) from %d sampled position(s).',
            gauge.to_cdtools,
            expected_factor,
            gauge.anomaly,
            sampled,
        )

    return gauge
