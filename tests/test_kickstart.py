"""Unit tests for the kickstart estimate: a probe and object from diffraction moments alone.

Behaviors verified:
  - the defocus decision table: a stated defocus is used, the product's focus position
    overrides the option, a resolved parallax fills in when none is stated, and with
    neither the probe keeps a flat phase with a warning; a parallax that disagrees with
    the stated defocus warns
  - the probe is a single mode, the true probe to a high overlap, carrying the incident
    count as its total power
  - the object's phase follows the sample's, and it is resliced into the input's layers
  - totals are normalized by the positions' measured incident photon counts, which removes a
    beam decay from the incident photon count
  - near-field metadata is rejected
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
import logging

import numpy
import pytest

from parallax_scan_fixtures import (
    METADATA,
    OBJECT_GEOMETRY,
    PROBE_GEOMETRY,
    accumulate_scan,
    make_positions,
    make_probe,
    make_transmission,
    simulate_scan,
)
from ptychodus.api.accumulate import DiffractionMoments, compute_diffraction_moments
from ptychodus.api.kickstart import (
    DefocusSource,
    KickstartEstimate,
    KickstartParameters,
    estimate_product_from_moments,
)
from ptychodus.api.preprocess.diffraction import BinningStep
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import ProductMetadata

_LOGGER_NAME = 'ptychodus.api.kickstart'


def _kickstart(
    moments: DiffractionMoments,
    *,
    metadata: ProductMetadata = METADATA,
    positions: list[ProbePosition] | None = None,
    **parameters: Any,
) -> KickstartEstimate:
    return estimate_product_from_moments(
        moments,
        ProbePositionSequence(make_positions() if positions is None else positions),
        metadata=metadata,
        probe_geometry=PROBE_GEOMETRY,
        object_geometry=OBJECT_GEOMETRY,
        parameters=KickstartParameters(**parameters),
    )


def _overlap(a: numpy.ndarray, b: numpy.ndarray) -> float:
    return float(abs(numpy.vdot(a, b)) / (numpy.linalg.norm(a) * numpy.linalg.norm(b)))


# ---------------------------------------------------------------------------
# The defocus decision table
# ---------------------------------------------------------------------------


def test_a_stated_defocus_is_used_and_agrees_with_the_parallax(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        estimate = _kickstart(accumulate_scan(2.0e-3), defocus_distance_m=2.0e-3)

    assert estimate.defocus_source is DefocusSource.STATED
    assert estimate.defocus_distance_m == 2.0e-3
    assert estimate.parallax is not None and estimate.parallax.is_resolved()
    assert caplog.records == []


def test_the_products_focus_position_overrides_the_option() -> None:
    metadata = replace(METADATA, focus_object_distance_m=-2.0e-3)

    estimate = _kickstart(accumulate_scan(2.0e-3), metadata=metadata, defocus_distance_m=5.0e-3)

    assert estimate.defocus_source is DefocusSource.STATED
    assert estimate.defocus_distance_m == pytest.approx(2.0e-3)


def test_a_disagreeing_parallax_warns_but_the_stated_defocus_is_used(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        estimate = _kickstart(accumulate_scan(2.0e-3), defocus_distance_m=-2.0e-3)

    assert estimate.defocus_distance_m == -2.0e-3
    assert any('parallax measures' in record.message for record in caplog.records)


@pytest.mark.parametrize('defocus_m', [2.0e-3, -2.0e-3])
def test_a_resolved_parallax_fills_in_when_none_is_stated(defocus_m: float) -> None:
    estimate = _kickstart(accumulate_scan(defocus_m))

    assert estimate.defocus_source is DefocusSource.PARALLAX
    assert estimate.defocus_distance_m == pytest.approx(defocus_m, rel=0.06)


def test_with_neither_the_probe_keeps_a_flat_phase(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER_NAME):
        estimate = _kickstart(accumulate_scan(0.0))

    assert estimate.defocus_source is DefocusSource.NONE
    assert estimate.defocus_distance_m == 0.0
    assert estimate.parallax is not None and not estimate.parallax.is_resolved()
    assert any('flat' in record.message for record in caplog.records)


def test_unbinned_moments_give_no_parallax() -> None:
    estimate = _kickstart(accumulate_scan(2.0e-3, bin_size_px=None), defocus_distance_m=2.0e-3)

    assert estimate.parallax is None
    assert estimate.defocus_source is DefocusSource.STATED


def test_a_stricter_depth_of_focus_threshold_rejects_the_parallax() -> None:
    estimate = _kickstart(accumulate_scan(2.0e-3), parallax_min_defocus_dof=3.0)

    assert estimate.defocus_source is DefocusSource.NONE


# ---------------------------------------------------------------------------
# The probe and object
# ---------------------------------------------------------------------------


def test_the_probe_is_the_true_probe_with_the_incident_power() -> None:
    estimate = _kickstart(accumulate_scan(2.0e-3))
    probes = estimate.product.probes.get_array()
    assert probes.shape == (1, 1, PROBE_GEOMETRY.height_px, PROBE_GEOMETRY.width_px)
    probe = probes[0, 0]

    # Measured at 0.978 on this scan.
    assert _overlap(probe, make_probe(2.0e-3).get_array()[0]) > 0.95
    assert numpy.sum(numpy.abs(probe) ** 2) == pytest.approx(
        estimate.incident_photon_count, rel=1e-9
    )
    assert estimate.product.probes.get_pixel_geometry() == PROBE_GEOMETRY.get_pixel_geometry()


def test_the_object_phase_follows_the_sample() -> None:
    estimate = _kickstart(accumulate_scan(0.0))
    interior = (slice(50, 110), slice(50, 110))
    phase = numpy.angle(estimate.product.object_.get_array()[0][interior])
    truth = numpy.angle(make_transmission(featureless=False)[interior])

    # Measured at 0.92 for the focused probe; DPC resolves only to the probe's width.
    assert numpy.corrcoef(phase.ravel(), truth.ravel())[0, 1] > 0.85


def test_the_object_is_resliced_into_the_inputs_layers() -> None:
    estimate = estimate_product_from_moments(
        accumulate_scan(2.0e-3),
        ProbePositionSequence(make_positions()),
        metadata=METADATA,
        probe_geometry=PROBE_GEOMETRY,
        object_geometry=OBJECT_GEOMETRY,
        layer_spacing_m=(1.0e-6, 1.0e-6),
    )

    assert estimate.product.object_.num_layers == 3


# ---------------------------------------------------------------------------
# Monitor normalization and the regime
# ---------------------------------------------------------------------------


def test_measured_incident_photon_counts_remove_a_beam_decay() -> None:
    data = simulate_scan(2.0e-3)
    indexes = numpy.array(data.get_indexes())
    decay = numpy.linspace(1.0, 0.5, indexes.size)
    decayed = compute_diffraction_moments(
        indexes,
        numpy.array(data.get_patterns()) * decay[:, numpy.newaxis, numpy.newaxis],
        bad_pixels=numpy.array(data.get_bad_pixels()),
        binning=BinningStep(bin_size_x=2, bin_size_y=2),
    )
    assert decayed is not None
    monitored = [
        replace(position, probe_photon_count=1.0e6 * float(scale))
        for position, scale in zip(make_positions(), decay)
    ]

    reference = _kickstart(accumulate_scan(2.0e-3), defocus_distance_m=2.0e-3)
    unmonitored = _kickstart(decayed, defocus_distance_m=2.0e-3)
    normalized = _kickstart(decayed, positions=monitored, defocus_distance_m=2.0e-3)

    assert not unmonitored.monitor_normalized
    assert normalized.monitor_normalized
    # Normalized to the mean monitor count, three quarters of the undecayed beam.
    assert normalized.incident_photon_count == pytest.approx(
        0.75 * reference.incident_photon_count, rel=0.02
    )
    assert unmonitored.incident_photon_count > 1.2 * normalized.incident_photon_count


def test_near_field_is_rejected() -> None:
    with pytest.raises(ValueError, match='far-field'):
        _kickstart(accumulate_scan(2.0e-3), metadata=replace(METADATA, far_field=False))
