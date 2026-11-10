"""Tests for the probe, object and probe-position metrics and comparison reports."""

from __future__ import annotations
from dataclasses import asdict, replace
import json

import numpy
import numpy.testing
import pytest

from ptychodus.api.analyze import (
    compute_object_comparison_report,
    compute_object_metrics_report,
    compute_probe_comparison_report,
    compute_probe_metrics_report,
    compute_probe_positions_comparison_report,
    compute_probe_positions_metrics_report,
)
from ptychodus.api.fourier import fourier_shift_2d
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.api.probe import ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.scattering.geometry import ScatteringGeometry

_PIXEL_M = 1.0e-8
_OBJECT_PX = 96
_PROBE_PX = 16
_STEP_PX = 4
_GRID_SIZE = 7


def _metadata(**kwargs: object) -> ProductMetadata:
    metadata = ProductMetadata(
        name='test',
        comments='',
        detector_distance_m=1.0,
        photon_energy_eV=10_000.0,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
    )
    return replace(metadata, **kwargs)  # type: ignore[arg-type]


def _smooth_field(seed: int, shape: tuple[int, int] = (_OBJECT_PX, _OBJECT_PX)) -> numpy.ndarray:
    """A real, band-limited random field with unit standard deviation."""
    rng = numpy.random.default_rng(seed)
    spectrum = numpy.fft.fft2(rng.standard_normal(shape))
    fy = numpy.fft.fftfreq(shape[0])[:, numpy.newaxis]
    fx = numpy.fft.fftfreq(shape[1])[numpy.newaxis, :]
    field = numpy.real(numpy.fft.ifft2(spectrum * numpy.exp(-(fx**2 + fy**2) / (2 * 0.05**2))))
    return field / numpy.std(field)


def _object_array(seed: int, *, num_layers: int = 1, phase: float = 0.5) -> numpy.ndarray:
    layers = []

    for layer in range(num_layers):
        amplitude = 0.8 + 0.05 * _smooth_field(seed + 100 * layer)
        layers.append(amplitude * numpy.exp(1j * phase * _smooth_field(seed + 100 * layer + 1)))

    return numpy.stack(layers).astype(numpy.complex128)


def _make_object(array: numpy.ndarray, *, pixel_m: float = _PIXEL_M) -> Object:
    return Object(
        array=array,
        pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[2.0e-6] * (array.shape[0] - 1),
    )


def _probe_modes(num_modes: int, *, size_px: int = _PROBE_PX, seed: int = 4) -> numpy.ndarray:
    rng = numpy.random.default_rng(seed)
    coordinate = numpy.arange(size_px) - (size_px - 1) / 2
    y, x = numpy.meshgrid(coordinate, coordinate, indexing='ij')
    envelope = numpy.exp(-(x**2 + y**2) / (2 * (size_px / 6) ** 2))
    modes = [envelope * (1.0 + 0.3j * x / size_px)]

    for _ in range(num_modes - 1):
        coefficients = 0.3 * (rng.standard_normal(3) + 1j * rng.standard_normal(3))
        modes.append(envelope * (coefficients[0] + coefficients[1] * x + coefficients[2] * y) / 4)

    return numpy.stack(modes)


def _raster_positions(*, indexes: list[int] | None = None) -> ProbePositionSequence:
    offsets = (numpy.arange(_GRID_SIZE) - (_GRID_SIZE - 1) / 2) * _STEP_PX * _PIXEL_M
    points = []

    for count, (y_m, x_m) in enumerate((y, x) for y in offsets for x in offsets):
        index = count if indexes is None else indexes[count]
        points.append(ProbePosition(index=index, x_m=float(x_m), y_m=float(y_m)))

    return ProbePositionSequence(points)


def _make_product(
    *,
    object_array: numpy.ndarray | None = None,
    probe_array: numpy.ndarray | None = None,
    positions: ProbePositionSequence | None = None,
    pixel_m: float = _PIXEL_M,
    **metadata: object,
) -> Product:
    if object_array is None:
        object_array = _object_array(0)

    if probe_array is None:
        probe_array = _probe_modes(2)

    return Product(
        metadata=_metadata(**metadata),
        probe_positions=_raster_positions() if positions is None else positions,
        probes=ProbeEnsemble(
            array=probe_array.astype(numpy.complex128),
            opr_weights=None,
            pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
        ),
        object_=_make_object(object_array, pixel_m=pixel_m),
        losses=[],
    )


def _assert_json_ready(scalars: object) -> None:
    json.dumps(asdict(scalars))  # type: ignore[call-overload]


# --------------------------------------------------------------------------- probe


class TestProbeMetricsReport:
    def test_describes_the_probe_and_its_overlap(self) -> None:
        report = compute_probe_metrics_report(_make_product())

        assert report.scalars.profile.num_modes == 2
        assert report.scalars.profile.size is not None
        assert 0.5 < report.scalars.profile.purity <= 1.0
        assert report.scalars.overlap is not None
        assert report.scalars.overlap.num_positions == _GRID_SIZE**2
        assert report.scalars.has_opr_weights is False
        numpy.testing.assert_allclose(report.profile.relative_power.sum(), 1.0)
        _assert_json_ready(report.scalars)

    def test_overlap_is_absent_without_positions(self) -> None:
        report = compute_probe_metrics_report(_make_product(positions=ProbePositionSequence()))

        assert report.overlap is None
        assert report.scalars.overlap is None


class TestProbeComparisonReport:
    def test_a_probe_agrees_with_itself(self) -> None:
        product = _make_product()

        report = compute_probe_comparison_report(product, product)

        assert report.scalars.normalized_fidelity == pytest.approx(1.0, abs=1e-9)
        assert report.scalars.normalized_rms_error == pytest.approx(0.0, abs=1e-4)
        assert report.scalars.probe_shift_x_px == pytest.approx(0.0, abs=0.02)
        assert report.scalars.amplitude.root_mean_square_error == pytest.approx(0.0, abs=1e-9)
        assert report.scalars.phase.root_mean_square_error == pytest.approx(0.0, abs=1e-9)
        assert report.scalars.test_object_resampled_from is None
        assert report.scalars.test_probe_resampled_from is None
        assert report.phase_support.any()
        _assert_json_ready(report.scalars)

    def test_a_global_phase_on_the_top_mode_is_aligned_away(self) -> None:
        reference = _make_product()
        modes = _probe_modes(2)
        modes[0] *= numpy.exp(0.7j)
        test = _make_product(probe_array=modes)

        report = compute_probe_comparison_report(reference, test)

        numpy.testing.assert_allclose(report.test_top_mode, report.reference_top_mode, atol=1e-6)

    def test_mode_counts_may_differ(self) -> None:
        reference = _make_product(probe_array=_probe_modes(2))
        test = _make_product(probe_array=_probe_modes(3))

        report = compute_probe_comparison_report(reference, test)

        assert report.scalars.reference.num_modes == 2
        assert report.scalars.test.num_modes == 3
        assert report.test.relative_power.size == 3

    def test_probe_shape_mismatch_is_resampled_or_refused(self) -> None:
        reference = _make_product()
        test = _make_product(probe_array=_probe_modes(2, size_px=_PROBE_PX + 4))

        report = compute_probe_comparison_report(reference, test)

        assert report.scalars.test_probe_resampled_from is not None
        assert report.scalars.test_probe_resampled_from.width_px == _PROBE_PX + 4
        assert report.scalars.test_object_resampled_from is None

        with pytest.raises(ValueError, match='Probe geometry mismatch'):
            compute_probe_comparison_report(reference, test, resample_test=False)


# -------------------------------------------------------------------------- object


class TestObjectMetricsReport:
    def test_recovers_a_phase_ramp_on_a_flat_object(self) -> None:
        flat = _object_array(1, phase=0.0)
        coordinate = (numpy.arange(_OBJECT_PX) - (_OBJECT_PX - 1) / 2) * _PIXEL_M
        ramp_x_rad_per_m, ramp_y_rad_per_m = 2.0e6, -1.0e6
        ramp = numpy.exp(
            1j
            * (
                ramp_x_rad_per_m * coordinate[numpy.newaxis, :]
                + ramp_y_rad_per_m * coordinate[:, numpy.newaxis]
            )
        )

        report = compute_object_metrics_report(_make_product(object_array=flat * ramp))

        ambiguities = report.scalars.ambiguities
        assert ambiguities.phase_ramp_x_rad_per_m == pytest.approx(ramp_x_rad_per_m, rel=1e-6)
        assert ambiguities.phase_ramp_y_rad_per_m == pytest.approx(ramp_y_rad_per_m, rel=1e-6)
        assert report.scalars.projected.phase_deviation_rad == pytest.approx(0.0, abs=1e-6)
        _assert_json_ready(report.scalars)

    def test_describes_every_layer(self) -> None:
        report = compute_object_metrics_report(
            _make_product(object_array=_object_array(2, num_layers=3))
        )

        assert len(report.scalars.layers) == 3
        assert len(report.layer_spectra) == 3
        assert report.scalars.layer_depths_m == pytest.approx((0.0, 2.0e-6, 4.0e-6))
        assert 0.0 < report.scalars.region_area_fraction < 1.0

    def test_refuses_a_product_without_a_scoring_region(self) -> None:
        dark = _make_product(probe_array=numpy.zeros((1, _PROBE_PX, _PROBE_PX)))

        with pytest.raises(ValueError):
            compute_object_metrics_report(dark)


class TestObjectComparisonReport:
    def test_an_object_agrees_with_itself(self) -> None:
        product = _make_product()

        report = compute_object_comparison_report(product, product)

        correlation = report.fourier_ring_correlation.correlation
        numpy.testing.assert_allclose(correlation[numpy.isfinite(correlation)], 1.0, atol=1e-9)
        assert report.scalars.amplitude.root_mean_square_error == pytest.approx(0.0, abs=1e-9)
        assert report.scalars.amplitude.structural_similarity == pytest.approx(1.0)
        assert report.scalars.phase.root_mean_square_error == pytest.approx(0.0, abs=1e-9)
        _assert_json_ready(report.scalars)

    def test_recovers_a_subpixel_shift(self) -> None:
        reference = _make_product()
        shifted = fourier_shift_2d(reference.object_.get_array(), dx=0.4, dy=-0.3)
        test = _make_product(object_array=shifted * numpy.exp(0.5j))

        report = compute_object_comparison_report(reference, test)

        assert report.scalars.shift_x_px == pytest.approx(-0.4, abs=0.02)
        assert report.scalars.shift_y_px == pytest.approx(0.3, abs=0.02)
        assert report.scalars.ambiguities.phase_offset_rad == pytest.approx(0.5, abs=0.02)

    def test_resolution_uses_the_requested_threshold(self) -> None:
        reference = _make_product()
        test = _make_product(object_array=_object_array(0) + 0.05 * _object_array(9))

        report = compute_object_comparison_report(reference, test, snr_threshold=3.0)

        frc = report.fourier_ring_correlation
        assert report.scalars.snr_threshold == 3.0
        assert report.scalars.resolution_m_at_snr == pytest.approx(
            frc.get_resolution_m_at_signal_to_noise_threshold(3.0), nan_ok=True
        )
        assert report.scalars.resolution_m_at_half_bit == pytest.approx(
            frc.get_resolution_m_at_bit_threshold(0.5), nan_ok=True
        )

    def test_pixel_size_mismatch_is_resampled_or_refused(self) -> None:
        reference = _make_product()
        test = _make_product(pixel_m=1.25 * _PIXEL_M)

        report = compute_object_comparison_report(reference, test)

        assert report.scalars.test_object_resampled_from == PixelGeometry(
            width_m=1.25 * _PIXEL_M, height_m=1.25 * _PIXEL_M
        )
        assert report.scalars.test_probe_resampled_from is not None

        with pytest.raises(ValueError, match='pixel geometry mismatch'):
            compute_object_comparison_report(reference, test, resample_test=False)


# ---------------------------------------------------------------------- positions


class TestProbePositionsMetricsReport:
    def test_regular_grid(self) -> None:
        report = compute_probe_positions_metrics_report(_make_product())
        scalars = report.scalars
        step_m = _STEP_PX * _PIXEL_M
        side_m = (_GRID_SIZE - 1) * step_m

        assert scalars.num_positions == _GRID_SIZE**2
        assert scalars.nearest_neighbor_mean_m == pytest.approx(step_m)
        assert scalars.nearest_neighbor_maximum_m == pytest.approx(step_m)
        assert scalars.convex_hull_area_m2 == pytest.approx(side_m**2)
        assert scalars.width_m == pytest.approx(side_m)
        assert scalars.num_duplicated_indexes == 0
        assert scalars.num_missing_indexes == 0
        assert scalars.is_exit_frame is False
        assert report.i0 is None
        _assert_json_ready(scalars)

    def test_duplicates_are_averaged_and_counted(self) -> None:
        indexes = list(range(_GRID_SIZE**2))
        indexes[1] = 0  # positions 0 and 1 share index 0
        indexes[-1] = 100  # leaves a gap of 52 indexes
        report = compute_probe_positions_metrics_report(
            _make_product(positions=_raster_positions(indexes=indexes))
        )

        assert report.scalars.num_duplicated_indexes == 1
        numpy.testing.assert_array_equal(report.duplicated_indexes, [0])
        assert report.scalars.num_distinct_indexes == _GRID_SIZE**2 - 1
        assert report.scalars.num_missing_indexes == 1 + 52
        numpy.testing.assert_array_equal(report.index_gaps, [[1, 1], [48, 99]])
        numpy.testing.assert_allclose(report.anchor_xy_m[0], [-10.0e-8, -12.0e-8])

    def test_i0_is_calibrated_only_when_the_metadata_says_how(self) -> None:
        positions = ProbePositionSequence(
            [
                ProbePosition(index=i, x_m=float(i) * 1e-8, y_m=0.0, probe_photon_count=2.0)
                for i in range(4)
            ]
        )

        raw = compute_probe_positions_metrics_report(_make_product(positions=positions))
        calibrated = compute_probe_positions_metrics_report(
            _make_product(positions=positions, i0_photons_per_count=10.0)
        )

        assert raw.scalars.i0_is_calibrated is False
        assert raw.scalars.i0_mean == pytest.approx(2.0)
        assert calibrated.scalars.i0_is_calibrated is True
        assert calibrated.scalars.i0_mean == pytest.approx(20.0)
        assert calibrated.scalars.convex_hull_area_m2 == 0.0

    def test_exit_frame_follows_the_geometry(self) -> None:
        bragg = ScatteringGeometry(
            two_theta_deg=47.35, detector_gamma_deg=-0.27, incidence_angle_deg=24.15
        )

        report = compute_probe_positions_metrics_report(_make_product(scattering_geometry=bragg))

        assert report.scalars.is_exit_frame is True

    def test_refuses_a_product_without_positions(self) -> None:
        with pytest.raises(ValueError, match='without probe positions'):
            compute_probe_positions_metrics_report(_make_product(positions=ProbePositionSequence()))


class TestProbePositionsComparisonReport:
    def test_positions_agree_with_themselves(self) -> None:
        product = _make_product()

        report = compute_probe_positions_comparison_report(product, product)

        assert report.scalars.num_matched == _GRID_SIZE**2
        assert report.scalars.maximum_magnitude_m == 0.0
        _assert_json_ready(report.scalars)

    def test_recovers_a_uniform_offset_and_counts_unmatched_indexes(self) -> None:
        reference = _make_product()
        test_positions = ProbePositionSequence(
            [
                ProbePosition(index=point.index, x_m=point.x_m + 2e-9, y_m=point.y_m - 1e-9)
                for point in reference.probe_positions
                if point.index >= 3
            ]
            + [ProbePosition(index=1000, x_m=0.0, y_m=0.0)]
        )
        test = replace(reference, probe_positions=test_positions)

        report = compute_probe_positions_comparison_report(reference, test)

        assert report.scalars.num_matched == _GRID_SIZE**2 - 3
        assert report.scalars.num_reference_only == 3
        assert report.scalars.num_test_only == 1
        assert report.scalars.mean_displacement_x_m == pytest.approx(2e-9)
        assert report.scalars.mean_displacement_y_m == pytest.approx(-1e-9)
        numpy.testing.assert_allclose(report.magnitude_m, numpy.hypot(2e-9, 1e-9))

    def test_refuses_positions_in_different_frames(self) -> None:
        reference = _make_product()
        bragg = _make_product(
            scattering_geometry=ScatteringGeometry(
                two_theta_deg=47.35, detector_gamma_deg=-0.27, incidence_angle_deg=24.15
            )
        )

        with pytest.raises(ValueError, match='different frames'):
            compute_probe_positions_comparison_report(reference, bragg)

    def test_refuses_positions_sharing_no_index(self) -> None:
        reference = _make_product()
        test = _make_product(positions=_raster_positions(indexes=list(range(1000, 1049))))

        with pytest.raises(ValueError, match='share no scan index'):
            compute_probe_positions_comparison_report(reference, test)
