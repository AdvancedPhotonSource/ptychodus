"""Unit tests for the illumination map dataclass and compute function in
ptychodus.api.illumination."""

from __future__ import annotations

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.illumination import IlluminationMap, compute_illumination_map
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
        center=ObjectCenter(coordinate_x_m=object_center[0], coordinate_y_m=object_center[1]),
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
    photon_flux_Hz: float = 1.0e10,  # noqa: N803
) -> IlluminationMap:
    if photon_number is None:
        photon_number = numpy.array([[1.0, 2.0], [3.0, 4.0]])
    return IlluminationMap(
        photon_number=photon_number,
        photon_flux_Hz=photon_flux_Hz,
        photon_energy_J=photon_energy_J,
        exposure_time_s=exposure_time_s,
        mass_attenuation_m2_kg=mass_attenuation_m2_kg,
        pixel_geometry=PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m),
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
    )


class TestIlluminationMap:
    def test_photon_fluence_divides_by_pixel_area(self) -> None:
        m = _make_illumination_map(pixel_size_m=2.0e-7)  # area = 4e-14 m^2
        expected = m.photon_number / 4.0e-14
        numpy.testing.assert_allclose(m.photon_fluence_1_m2, expected)

    def test_photon_fluence_rate_divides_by_exposure_time(self) -> None:
        m = _make_illumination_map(exposure_time_s=0.5)
        numpy.testing.assert_allclose(m.photon_fluence_rate_Hz_m2, m.photon_fluence_1_m2 / 0.5)

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
            positions=[ProbePosition(index=0, coordinate_x_m=0.0, coordinate_y_m=0.0)],
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
                    coordinate_x_m=0.5 * pixel_size_m,
                    coordinate_y_m=0.5 * pixel_size_m,
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
            ProbePosition(index=0, coordinate_x_m=-8 * pixel_size_m, coordinate_y_m=0.0),
            ProbePosition(index=1, coordinate_x_m=+8 * pixel_size_m, coordinate_y_m=0.0),
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
            ProbePosition(index=0, coordinate_x_m=-12 * pixel_size_m, coordinate_y_m=0.0),
            ProbePosition(index=1, coordinate_x_m=0.0, coordinate_y_m=0.0),
            ProbePosition(index=2, coordinate_x_m=+12 * pixel_size_m, coordinate_y_m=0.0),
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
            positions=[ProbePosition(index=0, coordinate_x_m=0.0, coordinate_y_m=0.0)],
            pixel_size_m=pixel_size_m,
            object_center=(3.0e-7, -2.0e-7),
            metadata=metadata,
        )
        m = compute_illumination_map(product)
        assert m.exposure_time_s == 0.25
        assert m.mass_attenuation_m2_kg == 7.5
        assert m.photon_energy_J == pytest.approx(metadata.probe_energy_J)
        assert m.photon_flux_Hz == pytest.approx(2.0e9 / 0.25)
        assert m.pixel_geometry == PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m)
        assert m.center == ObjectCenter(coordinate_x_m=3.0e-7, coordinate_y_m=-2.0e-7)

    def test_zero_exposure_time_gives_nan_flux(self) -> None:
        metadata = _make_metadata(probe_photon_count=1.0e9, exposure_time_s=0.0)
        product = _make_product(
            object_array=_delta_object(16, 16),
            probe_array=_gaussian_probe(8, 8),
            positions=[ProbePosition(index=0, coordinate_x_m=0.0, coordinate_y_m=0.0)],
            metadata=metadata,
        )
        m = compute_illumination_map(product)
        assert numpy.isnan(m.photon_flux_Hz)
        assert m.exposure_time_s == 0.0

    def test_sums_intensity_across_incoherent_modes(self) -> None:
        """A 2-mode probe is reduced by summing |mode|^2 across the incoherent-mode axis."""
        mode_a = _gaussian_probe(8, 8, sigma=1.5)
        mode_b = 0.5 * _gaussian_probe(8, 8, sigma=2.5)
        probe_array = numpy.stack([mode_a, mode_b], axis=0)
        product = _make_product(
            object_array=_delta_object(16, 16),
            probe_array=probe_array,
            positions=[ProbePosition(index=0, coordinate_x_m=0.0, coordinate_y_m=0.0)],
        )
        m = compute_illumination_map(product)
        expected_patch = numpy.abs(mode_a) ** 2 + numpy.abs(mode_b) ** 2
        numpy.testing.assert_allclose(m.photon_number[4:12, 4:12], expected_patch, atol=1e-12)
