"""Tests for compute_probe_photon_counts_by_index (assemble.py) and its downstream propagation.

Covers:
- Diffraction-side wins when both sides supply per-pattern counts.
- Position-side counts key by scan index when the diffraction side lacks measurements.
- Per-pattern total-counts fallback for pattern indexes with no measured counts on either side.
- prepare_reconstruct_input averages duplicate-index position counts arithmetically and linearly
  interpolates counts for pattern indexes that lie between anchor positions -- symmetric with the
  existing (x, y) handling.
"""

from __future__ import annotations

import numpy
import pytest

from ptychodus.api.assemble import (
    AssembledDiffractionData,
    compute_probe_photon_counts_by_index,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.reconstruct import prepare_reconstruct_input


def _pixel() -> PixelGeometry:
    return PixelGeometry(width_m=1e-6, height_m=1e-6)


def _metadata() -> ProductMetadata:
    return ProductMetadata(
        name='test',
        comments='',
        detector_distance_m=1.0,
        probe_energy_eV=10_000.0,
        probe_photon_count=1,
        exposure_time_s=1.0,
        mass_attenuation_m2_kg=0.0,
        tomography_angle_deg=0.0,
    )


def _make_assembled_data(
    indexes: list[int],
    *,
    probe_photon_counts: numpy.ndarray | None = None,
    pattern_hw: int = 4,
) -> AssembledDiffractionData:
    indexes_arr = numpy.asarray(indexes, dtype=numpy.intp)
    n = indexes_arr.size
    # Tag each pattern row with its index so a later comparison can prove which
    # patterns survived the reconstruct-input merge.
    patterns = numpy.broadcast_to(indexes_arr.reshape(n, 1, 1), (n, pattern_hw, pattern_hw)).astype(
        numpy.intp
    )
    bad_pixels = numpy.zeros((pattern_hw, pattern_hw), dtype=numpy.bool_)
    return AssembledDiffractionData(
        indexes=indexes_arr,
        patterns=patterns,
        pixel_geometry=_pixel(),
        bad_pixels=bad_pixels,
        probe_photon_counts=probe_photon_counts,
    )


def _make_product(specs: list[tuple[int, float, float, float | None]]) -> Product:
    """Build a Product from (index, x_m, y_m, probe_photon_count) specs.

    Uses a minimally-sized probe/object; the tests only exercise position wiring, not physics.
    """
    points = [
        ProbePosition(index=i, x_m=x, y_m=y, probe_photon_count=c)
        for i, x, y, c in specs
    ]
    positions = ProbePositionSequence(points)

    probe_array = numpy.zeros((1, 1, 4, 4), dtype=numpy.complex128)
    probes = ProbeSequence(array=probe_array, opr_weights=None, pixel_geometry=_pixel())

    obj_array = numpy.zeros((1, 16, 16), dtype=numpy.complex128)
    object_ = Object(
        array=obj_array,
        pixel_geometry=_pixel(),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[],
    )
    return Product(
        metadata=_metadata(),
        probe_positions=positions,
        probes=probes,
        object_=object_,
        losses=[],
    )


class TestComputeProbePhotonCountsByIndex:
    def test_diffraction_side_wins_when_both_supply_counts(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # Diffraction side carries measured counts for every valid pattern.
        diff_counts = numpy.array([10.0, 20.0, 30.0], dtype=numpy.float64)
        data = _make_assembled_data([0, 1, 2], probe_photon_counts=diff_counts)
        # Position side supplies different counts at the same indexes -- must be ignored.
        product = _make_product([(0, 0.0, 0.0, 111.0), (1, 1.0, 0.0, 222.0), (2, 2.0, 0.0, 333.0)])

        with caplog.at_level('DEBUG', logger='ptychodus.api.assemble'):
            result = compute_probe_photon_counts_by_index(data, product.probe_positions)

        assert result == {0: 10.0, 1: 20.0, 2: 30.0}
        assert 'overridden' in caplog.text

    def test_position_side_used_when_diffraction_side_absent(self) -> None:
        data = _make_assembled_data([0, 1, 2])
        product = _make_product([(0, 0.0, 0.0, 111.0), (1, 1.0, 0.0, 222.0), (2, 2.0, 0.0, 333.0)])
        result = compute_probe_photon_counts_by_index(data, product.probe_positions)
        assert result == {0: 111.0, 1: 222.0, 2: 333.0}

    def test_total_counts_fallback_for_pattern_indexes_outside_position_coverage(self) -> None:
        # Positions cover indexes {1, 2} with counts on every one (all-or-nothing).
        # The assembled data adds pattern index 5, which no position covers -- that
        # pattern should fall through to per-pattern total counts. Pattern rows are
        # tagged with their index, so total_counts per pattern = index * (H*W) = index * 16.
        data = _make_assembled_data([1, 2, 5])
        product = _make_product([(1, 1.0, 0.0, 100.0), (2, 2.0, 0.0, 200.0)])
        result = compute_probe_photon_counts_by_index(data, product.probe_positions)
        assert result[1] == pytest.approx(100.0)
        assert result[2] == pytest.approx(200.0)
        assert result[5] == pytest.approx(5 * 16)

    def test_returns_total_counts_when_neither_side_supplies_measurements(self) -> None:
        # No measured counts anywhere: every index falls through to total_counts,
        # preserving the pre-refactor illumination-map weighting behavior.
        data = _make_assembled_data([0, 1, 2])
        product = _make_product([(0, 0.0, 0.0, None), (1, 1.0, 0.0, None), (2, 2.0, 0.0, None)])
        result = compute_probe_photon_counts_by_index(data, product.probe_positions)
        assert set(result.keys()) == {0, 1, 2}
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(1 * 16)
        assert result[2] == pytest.approx(2 * 16)


class TestPrepareReconstructInputPhotonCountPropagation:
    def test_position_counts_propagate_verbatim_for_exact_matches(self) -> None:
        data = _make_assembled_data([0, 1, 2])
        product = _make_product([(0, 0.0, 0.0, 10.0), (1, 1.0, 0.0, 20.0), (2, 2.0, 0.0, 30.0)])
        result = prepare_reconstruct_input(data, product)
        counts = [p.probe_photon_count for p in result.product.probe_positions]
        assert counts == [pytest.approx(10.0), pytest.approx(20.0), pytest.approx(30.0)]

    def test_duplicate_index_counts_are_arithmetically_averaged(self) -> None:
        # Two duplicate anchors at index 1 -> their counts (12, 20) mean to 16.
        data = _make_assembled_data([1])
        product = _make_product([(1, 1.0, 0.0, 12.0), (1, 3.0, 0.0, 20.0)])
        result = prepare_reconstruct_input(data, product)
        [only] = list(result.product.probe_positions)
        assert only.probe_photon_count == pytest.approx(16.0)

    def test_missing_index_counts_are_linearly_interpolated_between_anchors(self) -> None:
        # Anchors at 0 -> 100 and 4 -> 200; pattern 2 has no matching position
        # so it should linearly interpolate to 150. Coordinates use the same
        # linear-interp path -- confirms symmetric handling.
        data = _make_assembled_data([0, 2, 4])
        product = _make_product([(0, 0.0, 0.0, 100.0), (4, 4.0, 0.0, 200.0)])
        result = prepare_reconstruct_input(data, product)
        by_index = {p.index: p.probe_photon_count for p in result.product.probe_positions}
        assert by_index[0] == pytest.approx(100.0)
        assert by_index[2] == pytest.approx(150.0)
        assert by_index[4] == pytest.approx(200.0)

    def test_no_position_counts_yields_none_photon_counts_on_output(self) -> None:
        # If neither the diffraction nor the position side carries measurements,
        # the reconstruct-input positions should have probe_photon_count == None so
        # downstream code that inspects them explicitly can distinguish that case.
        data = _make_assembled_data([0, 1, 2])
        product = _make_product([(0, 0.0, 0.0, None), (1, 1.0, 0.0, None), (2, 2.0, 0.0, None)])
        result = prepare_reconstruct_input(data, product)
        for p in result.product.probe_positions:
            assert p.probe_photon_count is None
