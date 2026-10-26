"""The probe gauge cdtools is handed a product in, measured against its own far field.

Needs no backend installed: the fixture's numpy far field is the one cdtools
implements, which ``test_cdtools_reconstruct.py`` checks where cdtools is present.
"""

from __future__ import annotations

from dataclasses import replace
import logging

import numpy
import pytest

from cdtools_fixtures import (
    PROBE_HEIGHT_PX,
    PROBE_WIDTH_PX,
    far_field_intensity,
    make_input,
    make_probes,
    make_product,
    simulate_patterns,
)
from ptychodus.api.reconstruct import ReconstructInput
from ptychodus.plugins.cdtools.gauge import measure_probe_gauge


def test_a_matched_product_reads_as_one() -> None:
    parameters = make_input()
    gauge = measure_probe_gauge(parameters, far_field_intensity)
    assert gauge.expected_factor == pytest.approx(1.0)
    assert gauge.to_cdtools == pytest.approx(1.0, rel=1e-9)
    assert gauge.sampled_positions == len(parameters.product.probe_positions)


@pytest.mark.parametrize('scale', [0.5, 3.0])
def test_a_probe_in_another_gauge_reads_as_the_factor_that_restores_it(scale: float) -> None:
    patterns = simulate_patterns(make_product())
    product = make_product(probes=make_probes(scale=scale))
    parameters = replace(make_input(product), diffraction_patterns=patterns)
    gauge = measure_probe_gauge(parameters, far_field_intensity)
    assert gauge.to_cdtools == pytest.approx(1.0 / scale, rel=1e-9)


def test_a_far_off_gauge_is_corrected_with_a_warning(caplog: pytest.LogCaptureFixture) -> None:
    patterns = simulate_patterns(make_product())
    product = make_product(probes=make_probes(scale=100.0))
    parameters = replace(make_input(product), diffraction_patterns=patterns)

    with caplog.at_level(logging.WARNING):
        gauge = measure_probe_gauge(parameters, far_field_intensity)

    assert gauge.to_cdtools == pytest.approx(0.01, rel=1e-9)
    assert 'another gauge' in caplog.text


def test_a_carried_background_is_subtracted_from_the_measurement() -> None:
    clean = make_input()
    background = numpy.full((PROBE_HEIGHT_PX, PROBE_WIDTH_PX), 2.5)
    parameters = replace(
        clean,
        diffraction_patterns=clean.diffraction_patterns + background,
        background=background,
    )
    assert measure_probe_gauge(parameters, far_field_intensity).to_cdtools == pytest.approx(1.0)


def test_a_flat_field_the_model_applies_multiplies_the_prediction() -> None:
    clean = make_input()
    flat_field = numpy.linspace(0.5, 1.5, PROBE_HEIGHT_PX * PROBE_WIDTH_PX).reshape(
        PROBE_HEIGHT_PX, PROBE_WIDTH_PX
    )
    parameters = replace(clean, diffraction_patterns=clean.diffraction_patterns * flat_field)

    with_it = measure_probe_gauge(parameters, far_field_intensity, flat_field=flat_field)
    without = measure_probe_gauge(parameters, far_field_intensity)

    assert with_it.to_cdtools == pytest.approx(1.0, rel=1e-9)
    assert without.to_cdtools != pytest.approx(1.0, rel=1e-3)


def test_bad_pixels_drop_from_both_sides() -> None:
    clean = make_input()
    bad_pixels = numpy.zeros_like(clean.bad_pixels)
    bad_pixels[PROBE_HEIGHT_PX // 2, :] = True
    corrupted = clean.diffraction_patterns.copy()
    corrupted[:, bad_pixels] = 1.0e9
    parameters = replace(clean, diffraction_patterns=corrupted, bad_pixels=bad_pixels)
    assert measure_probe_gauge(parameters, far_field_intensity).to_cdtools == pytest.approx(1.0)


def test_an_unnormalized_propagator_is_the_expected_factor() -> None:
    """A propagator that is not energy-conserving is the convention, not an anomaly."""

    def unnormalized(wavefield: numpy.ndarray) -> numpy.ndarray:
        return far_field_intensity(wavefield) * (PROBE_HEIGHT_PX * PROBE_WIDTH_PX)

    gauge = measure_probe_gauge(make_input(), unnormalized)
    assert gauge.expected_factor == pytest.approx((PROBE_HEIGHT_PX * PROBE_WIDTH_PX) ** -0.5)
    assert gauge.anomaly == pytest.approx(1.0, rel=1e-9)


def test_an_unusable_measurement_falls_back_to_the_convention(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clean = make_input()
    parameters = ReconstructInput(
        diffraction_patterns=numpy.zeros_like(clean.diffraction_patterns),
        bad_pixels=clean.bad_pixels,
        product=clean.product,
    )

    with caplog.at_level(logging.WARNING):
        gauge = measure_probe_gauge(parameters, far_field_intensity)

    assert gauge.to_cdtools == gauge.expected_factor
    assert gauge.sampled_positions == 0
    assert 'unusable' in caplog.text
