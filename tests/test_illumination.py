"""Unit tests for the illumination map dataclass and compute function in
ptychodus.api.illumination."""

from __future__ import annotations

from pathlib import Path
import math

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.illumination import (
    IlluminationMap,
    ProbeOverlapMetrics,
    compute_illumination_map,
    compute_probe_overlap,
)
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    *,
    probe_energy_eV: float = 10000.0,  # noqa: N803
    probe_photon_count: float = 1.0e9,
    exposure_time_s: float = 0.1,
    mass_attenuation_m2_kg: float = 5.0,
) -> ProductMetadata:
    return ProductMetadata(
        name='test',
        comments='',
        detector_distance_m=1.0,
        probe_energy_eV=probe_energy_eV,
        probe_photon_count=probe_photon_count,
        exposure_time_s=exposure_time_s,
        mass_attenuation_m2_kg=mass_attenuation_m2_kg,
        tomography_angle_deg=0.0,
    )


def _make_product(
    *,
    object_array: numpy.ndarray,
    probe_array: numpy.ndarray,
    positions: list[ProbePosition],
    pixel_size_m: float = 1.0e-7,
    object_center: tuple[float, float] = (0.0, 0.0),
    metadata: ProductMetadata | None = None,
    opr_weights: numpy.ndarray | None = None,
) -> Product:
    """Build a minimal Product. ``probe_array`` may be 2D (single mode) or
    3D (multi-mode); supply ``opr_weights`` of shape (N, num_coherent_modes)
    when iterating over multiple positions."""
    pixel_geometry = PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m)
    obj = Object(
        array=object_array.astype(complex),
        pixel_geometry=pixel_geometry,
        center=ObjectCenter(x_m=object_center[0], y_m=object_center[1]),
    )
    probes = ProbeSequence(
        array=probe_array.astype(complex),
        opr_weights=opr_weights,
        pixel_geometry=pixel_geometry,
    )
    return Product(
        metadata=metadata if metadata is not None else _make_metadata(),
        probe_positions=ProbePositionSequence(positions),
        probes=probes,
        object_=obj,
        losses=[],
    )


def _delta_object(height_px: int, width_px: int) -> numpy.ndarray:
    """A trivial complex object: zeros (only the canvas dimensions matter)."""
    return numpy.zeros((height_px, width_px), dtype=complex)


def _gaussian_probe(height_px: int, width_px: int, sigma: float = 2.0) -> numpy.ndarray:
    y = numpy.arange(height_px).reshape(-1, 1) - (height_px - 1) / 2
    x = numpy.arange(width_px).reshape(1, -1) - (width_px - 1) / 2
    return numpy.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(complex)


# ---------------------------------------------------------------------------
# IlluminationMap — derived properties
# ---------------------------------------------------------------------------


def _make_illumination_map(
    *,
    photon_number: numpy.ndarray | None = None,
    pixel_size_m: float = 2.0e-7,
    photon_energy_J: float = 1.6e-15,  # noqa: N803
    exposure_time_s: float = 0.5,
    mass_attenuation_m2_kg: float = 3.0,
    photon_flux_per_s: float = 1.0e10,  # noqa: N803
) -> IlluminationMap:
    if photon_number is None:
        photon_number = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    return IlluminationMap(
        photon_number=photon_number,
        photon_flux_per_s=photon_flux_per_s,
        photon_energy_J=photon_energy_J,
        exposure_time_s=exposure_time_s,
        mass_attenuation_m2_kg=mass_attenuation_m2_kg,
        pixel_geometry=PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
    )


class TestIlluminationMap:
    def test_photon_fluence_divides_by_pixel_area(self) -> None:
        m = _make_illumination_map(pixel_size_m=2.0e-7)  # area = 4e-14 m^2
        expected = m.photon_number / 4.0e-14
        numpy.testing.assert_allclose(m.photon_fluence_1_m2, expected)

    def test_photon_fluence_rate_divides_by_exposure_time(self) -> None:
        m = _make_illumination_map(exposure_time_s=0.5)
        numpy.testing.assert_allclose(m.photon_fluence_rate_per_s_m2, m.photon_fluence_1_m2 / 0.5)

    def test_save_npz_writes_the_documented_keys(self, tmp_path: Path) -> None:
        """The archive is an external interface: these names reach users' own scripts,
        and nothing in ptychodus reads it back to catch a rename."""
        m = _make_illumination_map()
        file_path = tmp_path / 'illumination.npz'
        m.save_npz(file_path)

        with numpy.load(file_path) as contents:
            assert set(contents.files) == {
                'photon_number',
                'photon_fluence_1_m2',
                'photon_fluence_rate_per_s_m2',
                'energy_fluence_J_m2',
                'energy_fluence_rate_W_m2',
                'dose_Gy',
                'dose_rate_Gy_s',
                'pixel_height_m',
                'pixel_width_m',
                'center_x_m',
                'center_y_m',
            }
            numpy.testing.assert_allclose(
                contents['photon_fluence_rate_per_s_m2'], m.photon_fluence_rate_per_s_m2
            )

    def test_energy_fluence_multiplies_by_photon_energy(self) -> None:
        m = _make_illumination_map(photon_energy_J=1.6e-15)
        numpy.testing.assert_allclose(m.energy_fluence_J_m2, m.photon_fluence_1_m2 * 1.6e-15)

    def test_energy_fluence_rate_equals_intensity_alias(self) -> None:
        m = _make_illumination_map()
        numpy.testing.assert_array_equal(m.intensity_W_m2, m.energy_fluence_rate_W_m2)

    def test_dose_Gy_equals_energy_fluence_times_mass_attenuation(self) -> None:  # noqa: N802
        m = _make_illumination_map(mass_attenuation_m2_kg=3.0)
        numpy.testing.assert_allclose(m.dose_Gy, m.energy_fluence_J_m2 * 3.0)

    def test_dose_rate_consistent_with_dose_over_exposure_time(self) -> None:
        m = _make_illumination_map(exposure_time_s=0.5)
        numpy.testing.assert_allclose(m.dose_rate_Gy_s, m.dose_Gy / 0.5)


# ---------------------------------------------------------------------------
# compute_illumination_map — algorithm
# ---------------------------------------------------------------------------


class TestComputeIlluminationMap:
    def test_single_position_integer_offset_places_probe_exactly(self) -> None:
        """With a probe placed exactly at the object center (integer pixel offset and
        zero subpixel residual), the canvas patch equals sum(|probe|^2) over modes."""
        probe = _gaussian_probe(8, 8, sigma=1.5)
        # Object is 16x16, probe is 8x8. Object center is at world (0, 0) so the world
        # origin maps to object-pixel (8, 8). A scan position at world (0, 0) puts the
        # probe corner at object-pixel (4, 4) — an integer-aligned placement.
        product = _make_product(
            object_array=_delta_object(16, 16),
            probe_array=probe,
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
        )
        m = compute_illumination_map(product)
        expected_patch = numpy.abs(probe) ** 2
        numpy.testing.assert_allclose(m.photon_number[4:12, 4:12], expected_patch, atol=1e-12)
        # Everything outside the patch is zero.
        masked = m.photon_number.copy()
        masked[4:12, 4:12] = 0.0
        numpy.testing.assert_allclose(masked, 0.0, atol=1e-12)

    def test_total_photons_conserved_under_subpixel_shift(self) -> None:
        """Fourier subpixel shifts are unitary, so the canvas integrates to the same
        total photon count as the original probe intensity."""
        probe = _gaussian_probe(16, 16, sigma=3.0)
        # 64x64 object with margin so the shifted probe doesn't wrap into the canvas
        # edge. Half-pixel subpixel offset along x and y.
        pixel_size_m = 1.0e-7
        product = _make_product(
            object_array=_delta_object(64, 64),
            probe_array=probe,
            positions=[
                ProbePosition(
                    index=0,
                    x_m=0.5 * pixel_size_m,
                    y_m=0.5 * pixel_size_m,
                )
            ],
            pixel_size_m=pixel_size_m,
        )
        m = compute_illumination_map(product)
        total_in = float(numpy.sum(numpy.abs(probe) ** 2))
        total_out = float(numpy.sum(m.photon_number))
        assert total_out == pytest.approx(total_in, rel=1e-6)

    def test_two_disjoint_positions_accumulate_additively(self) -> None:
        """With two well-separated scan positions, each contributes its own probe
        intensity to the canvas independently of the other."""
        probe = _gaussian_probe(8, 8, sigma=1.0)
        pixel_size_m = 1.0e-7
        # Object: 16 rows x 32 cols, centered at world (0, 0), so world x in
        # [-16, +16] * pixel_size maps to columns [0, 32]. Place patches with object
        # x-center at column 8 (world x = -8 * pixel) and column 24 (world x = +8 * pixel).
        # The two 8x8 patches then span cols [4:12] and [20:28] — disjoint.
        positions = [
            ProbePosition(index=0, x_m=-8 * pixel_size_m, y_m=0.0),
            ProbePosition(index=1, x_m=+8 * pixel_size_m, y_m=0.0),
        ]
        product = _make_product(
            object_array=_delta_object(16, 32),
            probe_array=probe,
            positions=positions,
            pixel_size_m=pixel_size_m,
        )
        m = compute_illumination_map(product)
        expected_patch = numpy.abs(probe) ** 2
        numpy.testing.assert_allclose(m.photon_number[4:12, 4:12], expected_patch, atol=1e-12)
        numpy.testing.assert_allclose(m.photon_number[4:12, 20:28], expected_patch, atol=1e-12)

    def test_processes_all_positions_without_opr_weights(self) -> None:
        """Regression test for a silent bug where ``zip(probe_positions, probes)``
        truncated to a single position whenever the product lacked OPR weights.
        Three well-separated positions, no OPR — all three patches must appear.
        """
        probe = _gaussian_probe(8, 8, sigma=1.0)
        pixel_size_m = 1.0e-7
        positions = [
            ProbePosition(index=0, x_m=-12 * pixel_size_m, y_m=0.0),
            ProbePosition(index=1, x_m=0.0, y_m=0.0),
            ProbePosition(index=2, x_m=+12 * pixel_size_m, y_m=0.0),
        ]
        product = _make_product(
            object_array=_delta_object(16, 48),
            probe_array=probe,
            positions=positions,
            pixel_size_m=pixel_size_m,
            # opr_weights deliberately omitted — this is the bug-triggering case.
        )
        m = compute_illumination_map(product)
        expected_patch = numpy.abs(probe) ** 2
        # 48-wide object, center col 24. Patches span: cols [8:16], [20:28], [32:40].
        numpy.testing.assert_allclose(m.photon_number[4:12, 8:16], expected_patch, atol=1e-12)
        numpy.testing.assert_allclose(m.photon_number[4:12, 20:28], expected_patch, atol=1e-12)
        numpy.testing.assert_allclose(m.photon_number[4:12, 32:40], expected_patch, atol=1e-12)

    def test_metadata_passthrough(self) -> None:
        metadata = _make_metadata(
            probe_energy_eV=8000.0,
            probe_photon_count=2.0e9,
            exposure_time_s=0.25,
            mass_attenuation_m2_kg=7.5,
        )
        probe = _gaussian_probe(8, 8)
        pixel_size_m = 1.0e-7
        product = _make_product(
            object_array=_delta_object(16, 16),
            probe_array=probe,
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
            pixel_size_m=pixel_size_m,
            object_center=(3.0e-7, -2.0e-7),
            metadata=metadata,
        )
        m = compute_illumination_map(product)
        assert m.exposure_time_s == 0.25
        assert m.mass_attenuation_m2_kg == 7.5
        assert m.photon_energy_J == pytest.approx(metadata.probe_energy_J)
        assert m.photon_flux_per_s == pytest.approx(2.0e9 / 0.25)
        assert m.pixel_geometry == PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m)
        assert m.center == ObjectCenter(x_m=3.0e-7, y_m=-2.0e-7)

    def test_zero_exposure_time_diverges_the_flux(self) -> None:
        """A real photon count over a vanishing exposure is unbounded, not unknown.

        The flux comes from compute_product_geometry, so this and the product property
        table cannot disagree about the degenerate case.
        """
        metadata = _make_metadata(probe_photon_count=1.0e9, exposure_time_s=0.0)
        product = _make_product(
            object_array=_delta_object(16, 16),
            probe_array=_gaussian_probe(8, 8),
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
            metadata=metadata,
        )
        m = compute_illumination_map(product)
        assert math.isinf(m.photon_flux_per_s)
        assert m.exposure_time_s == 0.0

    def test_nothing_recorded_at_all_gives_nan_flux(self) -> None:
        """0/0 is the state of a product whose flux was never measured."""
        metadata = _make_metadata(probe_photon_count=0.0, exposure_time_s=0.0)
        product = _make_product(
            object_array=_delta_object(16, 16),
            probe_array=_gaussian_probe(8, 8),
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
            metadata=metadata,
        )
        m = compute_illumination_map(product)
        assert math.isnan(m.photon_flux_per_s)

    def test_sums_intensity_across_incoherent_modes(self) -> None:
        """A 2-mode probe is reduced by summing |mode|^2 across the incoherent-mode axis."""
        mode_a = _gaussian_probe(8, 8, sigma=1.5)
        mode_b = 0.5 * _gaussian_probe(8, 8, sigma=2.5)
        probe_array = numpy.stack([mode_a, mode_b], axis=0)
        product = _make_product(
            object_array=_delta_object(16, 16),
            probe_array=probe_array,
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
        )
        m = compute_illumination_map(product)
        expected_patch = numpy.abs(mode_a) ** 2 + numpy.abs(mode_b) ** 2
        numpy.testing.assert_allclose(m.photon_number[4:12, 4:12], expected_patch, atol=1e-12)

    def test_probe_photon_counts_weight_patches_and_preserve_budget(self) -> None:
        """Per-scan-index weights redistribute photons across positions without
        changing the total photon budget on the canvas."""
        probe = _gaussian_probe(8, 8, sigma=1.0)
        pixel_size_m = 1.0e-7
        positions = [
            ProbePosition(index=0, x_m=-8 * pixel_size_m, y_m=0.0),
            ProbePosition(index=1, x_m=+8 * pixel_size_m, y_m=0.0),
        ]
        product = _make_product(
            object_array=_delta_object(16, 32),
            probe_array=probe,
            positions=positions,
            pixel_size_m=pixel_size_m,
        )
        uniform = compute_illumination_map(product)
        # Weight index 0 at 100, index 1 at 300 -> mean 200 -> factors 0.5, 1.5.
        weighted = compute_illumination_map(
            product, probe_photon_counts_by_index={0: 100.0, 1: 300.0}
        )
        expected_patch = numpy.abs(probe) ** 2
        numpy.testing.assert_allclose(weighted.photon_number[4:12, 4:12], 0.5 * expected_patch)
        numpy.testing.assert_allclose(weighted.photon_number[4:12, 20:28], 1.5 * expected_patch)
        # Preserved total budget.
        assert float(weighted.photon_number.sum()) == pytest.approx(
            float(uniform.photon_number.sum())
        )

    def test_probe_photon_counts_missing_index_gets_unit_weight(self) -> None:
        """A scan index absent from the mapping is weighted by the provided mean."""
        probe = _gaussian_probe(8, 8, sigma=1.0)
        pixel_size_m = 1.0e-7
        positions = [
            ProbePosition(index=0, x_m=-8 * pixel_size_m, y_m=0.0),
            ProbePosition(index=1, x_m=+8 * pixel_size_m, y_m=0.0),
        ]
        product = _make_product(
            object_array=_delta_object(16, 32),
            probe_array=probe,
            positions=positions,
            pixel_size_m=pixel_size_m,
        )
        # Only index 0 is in the mapping; index 1 gets the mean of the mapping (=100)
        # divided by the mean (=100) -> unit weight.
        weighted = compute_illumination_map(product, probe_photon_counts_by_index={0: 100.0})
        expected_patch = numpy.abs(probe) ** 2
        numpy.testing.assert_allclose(weighted.photon_number[4:12, 20:28], expected_patch)


# ---------------------------------------------------------------------------
# ProbeOverlapMetrics — derived properties
# ---------------------------------------------------------------------------


def _disc_probe(height_px: int, width_px: int, radius_px: float) -> numpy.ndarray:
    """A binary top-hat probe. Its participation-ratio area equals its pixel count
    exactly, which makes the effective-area assertions closed-form rather than
    approximate."""
    y = numpy.arange(height_px).reshape(-1, 1) - (height_px - 1) / 2
    x = numpy.arange(width_px).reshape(1, -1) - (width_px - 1) / 2
    return ((x**2 + y**2) <= radius_px * radius_px).astype(complex)


def _raster_positions(
    num_per_side: int, step_px: float, pixel_size_m: float
) -> list[ProbePosition]:
    return [
        ProbePosition(
            index=row * num_per_side + column,
            x_m=(column - (num_per_side - 1) / 2) * step_px * pixel_size_m,
            y_m=(row - (num_per_side - 1) / 2) * step_px * pixel_size_m,
        )
        for row in range(num_per_side)
        for column in range(num_per_side)
    ]


def _make_overlap_metrics(
    *,
    redundancy: numpy.ndarray | None = None,
    pairwise_overlap_by_position: numpy.ndarray | None = None,
    effective_probe_area_m2: float = 4.0e-14,
    effective_covered_area_m2: float = 1.6e-13,
    num_positions: int = 4,
) -> ProbeOverlapMetrics:
    return ProbeOverlapMetrics(
        redundancy=numpy.ones((2, 2)) if redundancy is None else redundancy,
        pairwise_overlap_by_position=numpy.array([0.2, 0.4, 0.6, 0.8])
        if pairwise_overlap_by_position is None
        else pairwise_overlap_by_position,
        effective_probe_area_m2=effective_probe_area_m2,
        effective_covered_area_m2=effective_covered_area_m2,
        num_positions=num_positions,
        pixel_geometry=PixelGeometry(width_m=1.0e-7, height_m=1.0e-7),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
    )


class TestProbeOverlapMetrics:
    def test_effective_diameter_is_the_equal_area_disc(self) -> None:
        m = _make_overlap_metrics(effective_probe_area_m2=4.0e-14)
        assert m.effective_probe_diameter_m == pytest.approx(2.0 * math.sqrt(4.0e-14 / math.pi))

    def test_effective_step_is_the_equal_area_square_per_position(self) -> None:
        m = _make_overlap_metrics(effective_covered_area_m2=1.6e-13, num_positions=4)
        assert m.effective_step_size_m == pytest.approx(math.sqrt(1.6e-13 / 4))

    def test_areal_redundancy_is_total_probe_area_over_covered_area(self) -> None:
        m = _make_overlap_metrics(
            effective_probe_area_m2=4.0e-14, effective_covered_area_m2=1.6e-13, num_positions=4
        )
        assert m.areal_redundancy == pytest.approx(4 * 4.0e-14 / 1.6e-13)

    def test_equivalent_linear_overlap_is_the_classic_formula(self) -> None:
        """``1 - step / diameter`` and ``1 - sqrt(pi / 4R)`` are one quantity."""
        m = _make_overlap_metrics()
        assert m.equivalent_linear_overlap == pytest.approx(
            1.0 - m.effective_step_size_m / m.effective_probe_diameter_m
        )
        assert m.equivalent_linear_overlap == pytest.approx(
            1.0 - math.sqrt(math.pi / (4.0 * m.areal_redundancy))
        )

    def test_equivalent_linear_overlap_goes_negative_when_footprints_leave_gaps(self) -> None:
        """Sparse scans are reported as computed rather than clamped at zero."""
        m = _make_overlap_metrics(
            effective_probe_area_m2=1.0e-14, effective_covered_area_m2=1.0e-12, num_positions=4
        )
        assert m.areal_redundancy < math.pi / 4.0
        assert m.equivalent_linear_overlap < 0.0

    def test_pairwise_statistics(self) -> None:
        m = _make_overlap_metrics(pairwise_overlap_by_position=numpy.array([0.1, 0.3, 0.5, 0.9]))
        assert m.mean_pairwise_overlap == pytest.approx(0.45)
        assert m.median_pairwise_overlap == pytest.approx(0.4)
        assert m.minimum_pairwise_overlap == pytest.approx(0.1)
        assert m.maximum_pairwise_overlap == pytest.approx(0.9)

    def test_pairwise_statistics_are_nan_for_a_single_position(self) -> None:
        m = _make_overlap_metrics(pairwise_overlap_by_position=numpy.zeros(1), num_positions=1)
        assert math.isnan(m.mean_pairwise_overlap)
        assert math.isnan(m.median_pairwise_overlap)
        assert math.isnan(m.minimum_pairwise_overlap)
        assert math.isnan(m.maximum_pairwise_overlap)

    def test_areal_redundancy_is_nan_without_coverage(self) -> None:
        m = _make_overlap_metrics(effective_covered_area_m2=0.0)
        assert math.isnan(m.areal_redundancy)

    def test_save_npz_round_trip(self, tmp_path: Path) -> None:
        m = _make_overlap_metrics()
        file_path = tmp_path / 'overlap.npz'
        m.save_npz(file_path)

        with numpy.load(file_path) as contents:
            numpy.testing.assert_allclose(contents['redundancy'], m.redundancy)
            numpy.testing.assert_allclose(
                contents['pairwise_overlap_by_position'], m.pairwise_overlap_by_position
            )
            assert float(contents['areal_redundancy']) == pytest.approx(m.areal_redundancy)
            assert float(contents['equivalent_linear_overlap']) == pytest.approx(
                m.equivalent_linear_overlap
            )
            assert int(contents['num_positions']) == m.num_positions


# ---------------------------------------------------------------------------
# compute_probe_overlap — algorithm
# ---------------------------------------------------------------------------


class TestComputeProbeOverlap:
    def test_redundancy_counts_coincident_probes(self) -> None:
        """k probes stacked on one spot means every illuminated pixel is seen k times."""
        probe = _disc_probe(32, 32, radius_px=8.0)

        for num_positions in (1, 2, 3):
            product = _make_product(
                object_array=_delta_object(64, 64),
                probe_array=probe,
                positions=[ProbePosition(index=i, x_m=0.0, y_m=0.0) for i in range(num_positions)],
            )
            m = compute_probe_overlap(product)
            illuminated = m.redundancy[numpy.isfinite(m.redundancy)]
            numpy.testing.assert_allclose(illuminated, float(num_positions), atol=1e-9)

    def test_redundancy_is_a_soft_count_for_unequal_probes(self) -> None:
        """Unequal contributions degrade gracefully rather than rounding to a whole
        probe: powers 1 and 3 give ``(1 + 3)^2 / (1 + 9) = 1.6``, not 2."""
        probe = _disc_probe(32, 32, radius_px=8.0)
        product = _make_product(
            object_array=_delta_object(64, 64),
            probe_array=probe,
            positions=[ProbePosition(index=i, x_m=0.0, y_m=0.0) for i in range(2)],
            opr_weights=numpy.array([[1.0], [math.sqrt(3.0)]]),
        )
        m = compute_probe_overlap(product)
        illuminated = m.redundancy[numpy.isfinite(m.redundancy)]
        numpy.testing.assert_allclose(illuminated, 1.6, atol=1e-9)

    def test_disjoint_probes_have_unit_redundancy(self) -> None:
        probe = _disc_probe(16, 16, radius_px=4.0)
        pixel_size_m = 1.0e-7
        product = _make_product(
            object_array=_delta_object(32, 64),
            probe_array=probe,
            positions=[
                ProbePosition(index=0, x_m=-12 * pixel_size_m, y_m=0.0),
                ProbePosition(index=1, x_m=+12 * pixel_size_m, y_m=0.0),
            ],
            pixel_size_m=pixel_size_m,
        )
        m = compute_probe_overlap(product)
        illuminated = m.redundancy[numpy.isfinite(m.redundancy)]
        numpy.testing.assert_allclose(illuminated, 1.0, atol=1e-9)

    def test_metrics_are_invariant_under_probe_intensity_rescale(self) -> None:
        """Overlap is geometry, not photometry: a brighter probe changes the
        illumination map but must not change any overlap number."""
        probe = _gaussian_probe(32, 32, sigma=3.0)
        pixel_size_m = 1.0e-7
        # Half-pixel offsets so the subpixel Fourier shift is exercised too.
        positions = [
            ProbePosition(index=0, x_m=(-4 + 0.5) * pixel_size_m, y_m=0.0),
            ProbePosition(index=1, x_m=(+4 + 0.5) * pixel_size_m, y_m=0.0),
        ]

        def overlap_for(probe_array: numpy.ndarray) -> ProbeOverlapMetrics:
            return compute_probe_overlap(
                _make_product(
                    object_array=_delta_object(64, 64),
                    probe_array=probe_array,
                    positions=positions,
                    pixel_size_m=pixel_size_m,
                )
            )

        dim = overlap_for(probe)
        bright = overlap_for(10.0 * probe)

        numpy.testing.assert_allclose(dim.redundancy, bright.redundancy, rtol=1e-9, equal_nan=True)
        numpy.testing.assert_allclose(
            dim.pairwise_overlap_by_position, bright.pairwise_overlap_by_position, rtol=1e-9
        )
        assert dim.areal_redundancy == pytest.approx(bright.areal_redundancy)
        assert dim.equivalent_linear_overlap == pytest.approx(bright.equivalent_linear_overlap)
        assert dim.effective_probe_area_m2 == pytest.approx(bright.effective_probe_area_m2)

    def test_effective_diameter_matches_the_disc_area(self) -> None:
        """For a top-hat the participation ratio is the support area exactly, so the
        effective diameter is the equal-area disc of the discretized footprint."""
        radius_px = 8.0
        probe = _disc_probe(32, 32, radius_px=radius_px)
        num_probe_px = int(numpy.sum(numpy.abs(probe) ** 2))
        pixel_size_m = 1.0e-7
        product = _make_product(
            object_array=_delta_object(64, 64),
            probe_array=probe,
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
            pixel_size_m=pixel_size_m,
        )
        m = compute_probe_overlap(product)
        assert m.effective_probe_area_m2 == pytest.approx(num_probe_px * pixel_size_m**2)
        assert m.effective_probe_diameter_m == pytest.approx(
            2.0 * math.sqrt(num_probe_px / math.pi) * pixel_size_m
        )

    def test_raster_interior_redundancy_is_probe_area_over_step_squared(self) -> None:
        """Away from the scan edges, the number of probes covering a pixel is the probe
        footprint area divided by the area each position is responsible for."""
        step_px = 4
        probe = _disc_probe(32, 32, radius_px=8.0)
        num_probe_px = int(numpy.sum(numpy.abs(probe) ** 2))
        product = _make_product(
            object_array=_delta_object(160, 160),
            probe_array=probe,
            positions=_raster_positions(13, step_px, 1.0e-7),
        )
        m = compute_probe_overlap(product)
        interior = m.redundancy[70:90, 70:90]
        assert float(numpy.nanmean(interior)) == pytest.approx(num_probe_px / step_px**2, rel=0.01)

    def test_denser_raster_gives_larger_equivalent_overlap(self) -> None:
        probe = _disc_probe(32, 32, radius_px=8.0)
        overlaps = []

        for step_px in (8, 6, 4):
            product = _make_product(
                object_array=_delta_object(160, 160),
                probe_array=probe,
                positions=_raster_positions(9, step_px, 1.0e-7),
            )
            m = compute_probe_overlap(product)
            assert m.equivalent_linear_overlap == pytest.approx(
                1.0 - math.sqrt(math.pi / (4.0 * m.areal_redundancy))
            )
            overlaps.append(m.equivalent_linear_overlap)

        assert overlaps == sorted(overlaps)

    def test_pairwise_overlap_matches_the_circle_lens_area(self) -> None:
        """This is the claim that the metric generalizes the classic overlap ratio: for
        two top-hat discs it reproduces the circle-circle lens area over the disc area."""
        radius_px = 12.0
        probe = _disc_probe(48, 48, radius_px=radius_px)
        num_probe_px = int(numpy.sum(numpy.abs(probe) ** 2))
        pixel_size_m = 1.0e-7

        for separation_px in (6.0, 10.0, 14.0):
            product = _make_product(
                object_array=_delta_object(96, 160),
                probe_array=probe,
                positions=[
                    ProbePosition(index=0, x_m=-0.5 * separation_px * pixel_size_m, y_m=0.0),
                    ProbePosition(index=1, x_m=+0.5 * separation_px * pixel_size_m, y_m=0.0),
                ],
                pixel_size_m=pixel_size_m,
            )
            m = compute_probe_overlap(product)
            lens_area_px = 2.0 * radius_px**2 * math.acos(
                separation_px / (2.0 * radius_px)
            ) - 0.5 * separation_px * math.sqrt(4.0 * radius_px**2 - separation_px**2)
            # Tolerance is set by the discretization of the disc boundary.
            numpy.testing.assert_allclose(
                m.pairwise_overlap_by_position, lens_area_px / num_probe_px, rtol=0.05
            )

    def test_pairwise_overlap_is_zero_for_disjoint_probes(self) -> None:
        probe = _disc_probe(16, 16, radius_px=4.0)
        pixel_size_m = 1.0e-7
        product = _make_product(
            object_array=_delta_object(32, 64),
            probe_array=probe,
            positions=[
                ProbePosition(index=0, x_m=-12 * pixel_size_m, y_m=0.0),
                ProbePosition(index=1, x_m=+12 * pixel_size_m, y_m=0.0),
            ],
            pixel_size_m=pixel_size_m,
        )
        m = compute_probe_overlap(product)
        numpy.testing.assert_allclose(m.pairwise_overlap_by_position, 0.0, atol=1e-12)

    def test_pairwise_overlap_flags_an_orphan_position(self) -> None:
        """A global average hides a position no neighbor constrains; the per-position
        array does not."""
        probe = _disc_probe(32, 32, radius_px=8.0)
        pixel_size_m = 1.0e-7
        product = _make_product(
            object_array=_delta_object(64, 192),
            probe_array=probe,
            positions=[
                ProbePosition(index=0, x_m=-4 * pixel_size_m, y_m=0.0),
                ProbePosition(index=1, x_m=+4 * pixel_size_m, y_m=0.0),
                ProbePosition(index=2, x_m=+60 * pixel_size_m, y_m=0.0),
            ],
            pixel_size_m=pixel_size_m,
        )
        m = compute_probe_overlap(product)
        assert m.pairwise_overlap_by_position[0] > 0.3
        assert m.pairwise_overlap_by_position[1] > 0.3
        assert m.pairwise_overlap_by_position[2] == pytest.approx(0.0)
        assert m.minimum_pairwise_overlap == pytest.approx(0.0)

    def test_metrics_are_invariant_under_rigid_translation(self) -> None:
        probe = _gaussian_probe(32, 32, sigma=3.0)
        pixel_size_m = 1.0e-7
        offsets_px = [(-6, -3), (0, 0), (6, 3)]
        results = []

        for shift_px in (0, 5):
            product = _make_product(
                object_array=_delta_object(96, 96),
                probe_array=probe,
                positions=[
                    ProbePosition(
                        index=i,
                        x_m=(dx + shift_px) * pixel_size_m,
                        y_m=(dy + shift_px) * pixel_size_m,
                    )
                    for i, (dx, dy) in enumerate(offsets_px)
                ],
                pixel_size_m=pixel_size_m,
            )
            results.append(compute_probe_overlap(product))

        original, translated = results
        assert original.areal_redundancy == pytest.approx(translated.areal_redundancy)
        assert original.equivalent_linear_overlap == pytest.approx(
            translated.equivalent_linear_overlap
        )
        numpy.testing.assert_allclose(
            original.pairwise_overlap_by_position,
            translated.pairwise_overlap_by_position,
            rtol=1e-9,
        )

    def test_redundancy_matches_the_illumination_map_canvas(self) -> None:
        """The redundancy map lives on the object canvas, finite where the probes reach
        and NaN where they do not."""
        probe = _disc_probe(16, 16, radius_px=4.0)
        product = _make_product(
            object_array=_delta_object(32, 32),
            probe_array=probe,
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
        )
        m = compute_probe_overlap(product)
        illumination = compute_illumination_map(product)

        assert m.redundancy.shape == illumination.photon_number.shape
        assert m.pixel_geometry == illumination.pixel_geometry
        assert m.center == illumination.center
        assert m.num_positions == 1
        assert m.pairwise_overlap_by_position.shape == (1,)
        # Finite exactly where the footprint lands, NaN in the untouched corners.
        assert numpy.isfinite(m.redundancy[16, 16])
        assert numpy.isnan(m.redundancy[0, 0])
        assert numpy.isnan(m.redundancy[-1, -1])

    def test_requires_probe_positions(self) -> None:
        product = _make_product(
            object_array=_delta_object(32, 32),
            probe_array=_disc_probe(16, 16, radius_px=4.0),
            positions=[],
        )

        with pytest.raises(ValueError, match='probe positions'):
            compute_probe_overlap(product)

    def test_single_position_has_nan_pairwise_statistics(self) -> None:
        product = _make_product(
            object_array=_delta_object(32, 32),
            probe_array=_disc_probe(16, 16, radius_px=4.0),
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
        )
        m = compute_probe_overlap(product)
        numpy.testing.assert_allclose(m.pairwise_overlap_by_position, 0.0)
        assert math.isnan(m.mean_pairwise_overlap)
        assert m.areal_redundancy == pytest.approx(1.0)

    def test_roundoff_floor_controls_the_unilluminated_mask(self) -> None:
        """The floor only decides which pixels are reported as NaN; raising it trims the
        faint halo of a Gaussian probe without touching any of the scalars."""
        product = _make_product(
            object_array=_delta_object(64, 64),
            probe_array=_gaussian_probe(32, 32, sigma=3.0),
            positions=[ProbePosition(index=0, x_m=0.0, y_m=0.0)],
        )
        permissive = compute_probe_overlap(product, roundoff_floor=1.0e-12)
        strict = compute_probe_overlap(product, roundoff_floor=1.0e-2)

        assert numpy.isfinite(strict.redundancy).sum() < numpy.isfinite(permissive.redundancy).sum()
        assert strict.effective_probe_area_m2 == pytest.approx(permissive.effective_probe_area_m2)
        assert strict.areal_redundancy == pytest.approx(permissive.areal_redundancy)
