"""Unit tests for ptychodus.api.probe."""

import numpy
import numpy.testing
import pytest

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.probe import (
    FocusPolarity,
    PatchBounds,
    Probe,
    ProbeFocusCurves,
    ProbeFocusMetric,
    ProbeGeometry,
    ProbeSequence,
    ProbeSizeMetrics,
    compute_amplitude_deviation,
    compute_phase_deviation_rad,
    compute_probe_focus_curves,
    compute_rms_contrast,
    compute_shannon_entropy,
    estimate_focal_plane,
    estimate_probe_entropy,
    estimate_probe_size,
    shift_probe,
)
from ptychodus.api.propagate import PropagatedWavefield


PIXEL_M = 1e-9  # 1 nm per pixel — keeps lengths interpretable as "px == nm"
PIXEL_GEOMETRY = PixelGeometry(width_m=PIXEL_M, height_m=PIXEL_M)


def _gaussian_2d(
    height: int,
    width: int,
    sigma_major_px: float,
    sigma_minor_px: float,
    tilt_rad: float,
) -> numpy.ndarray:
    """Construct a tilted anisotropic 2D Gaussian intensity on a centered grid."""
    y_idx, x_idx = numpy.mgrid[:height, :width]
    dx = x_idx - (width - 1) / 2.0
    dy = y_idx - (height - 1) / 2.0
    cos_t = numpy.cos(tilt_rad)
    sin_t = numpy.sin(tilt_rad)
    dx_p = cos_t * dx + sin_t * dy  # along the requested major axis
    dy_p = -sin_t * dx + cos_t * dy  # along the requested minor axis
    return numpy.exp(-0.5 * ((dx_p / sigma_major_px) ** 2 + (dy_p / sigma_minor_px) ** 2))


class TestEstimateProbeSize:
    def test_isotropic_gaussian_recovers_2sigma_fwhm_and_encircled_diameter(self) -> None:
        """For a known Gaussian, all three widths should match analytic values."""
        sigma_px = 10.0
        intensity = _gaussian_2d(128, 128, sigma_px, sigma_px, tilt_rad=0.0)

        metrics = estimate_probe_size(intensity, PIXEL_GEOMETRY)

        expected_2sigma = 2.0 * sigma_px * PIXEL_M
        expected_fwhm = 2.0 * numpy.sqrt(2.0 * numpy.log(2.0)) * sigma_px * PIXEL_M
        # 2D Gaussian encircled-energy radius at fraction f: r = sigma * sqrt(-2 ln(1 - f))
        expected_encircled = 2.0 * sigma_px * numpy.sqrt(-2.0 * numpy.log(0.2)) * PIXEL_M

        numpy.testing.assert_allclose(metrics.rms_major_axis_length_m, expected_2sigma, rtol=0.02)
        numpy.testing.assert_allclose(metrics.rms_minor_axis_length_m, expected_2sigma, rtol=0.02)
        numpy.testing.assert_allclose(metrics.fwhm_major_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(metrics.fwhm_minor_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(
            metrics.encircled_energy_diameter_m, expected_encircled, rtol=0.05
        )

    def test_rms_and_fwhm_share_full_width_scale(self) -> None:
        """Now that RMS is reported as the full 2-sigma width, the FWHM / RMS ratio
        for a Gaussian should be sqrt(2 ln 2) ~ 1.1774 — *not* 2*sqrt(2 ln 2) ~ 2.355.
        """
        intensity = _gaussian_2d(128, 128, sigma_major_px=10.0, sigma_minor_px=10.0, tilt_rad=0.0)
        metrics = estimate_probe_size(intensity, PIXEL_GEOMETRY)

        ratio = metrics.fwhm_major_axis_length_m / metrics.rms_major_axis_length_m
        numpy.testing.assert_allclose(ratio, numpy.sqrt(2.0 * numpy.log(2.0)), rtol=0.03)

    def test_anisotropic_tilted_gaussian_round_trip(self) -> None:
        """Major/minor axis lengths, ordering, and tilt should all round-trip."""
        sigma_major_px = 12.0
        sigma_minor_px = 6.0
        tilt = numpy.pi / 6.0  # 30 deg, comfortably inside [-pi/2, pi/2)
        intensity = _gaussian_2d(128, 128, sigma_major_px, sigma_minor_px, tilt)

        metrics = estimate_probe_size(intensity, PIXEL_GEOMETRY)

        assert metrics.rms_major_axis_length_m > metrics.rms_minor_axis_length_m
        assert metrics.fwhm_major_axis_length_m > metrics.fwhm_minor_axis_length_m

        numpy.testing.assert_allclose(
            metrics.rms_major_axis_length_m, 2.0 * sigma_major_px * PIXEL_M, rtol=0.02
        )
        numpy.testing.assert_allclose(
            metrics.rms_minor_axis_length_m, 2.0 * sigma_minor_px * PIXEL_M, rtol=0.02
        )

        numpy.testing.assert_allclose(metrics.major_axis_tilt_rad, tilt, atol=0.05)
        # Minor axis is perpendicular to major: |sin(major - minor)| ~ 1.
        delta = metrics.major_axis_tilt_rad - metrics.minor_axis_tilt_rad
        numpy.testing.assert_allclose(abs(numpy.sin(delta)), 1.0, atol=0.02)

    def test_uniform_disk_recovers_radius_and_encircled_diameter(self) -> None:
        """For a uniform disk of radius R: sigma = R/2 (so 2-sigma = R),
        the 1D projection is a semicircle profile with FWHM = R*sqrt(3),
        and the 80%-encircled diameter is 2 R sqrt(0.8).
        """
        radius_px = 20.0
        size = 128
        y_idx, x_idx = numpy.mgrid[:size, :size]
        r_px = numpy.hypot(x_idx - (size - 1) / 2.0, y_idx - (size - 1) / 2.0)
        disk = (r_px <= radius_px).astype(numpy.float64)

        metrics = estimate_probe_size(disk, PIXEL_GEOMETRY)

        expected_2sigma = radius_px * PIXEL_M
        expected_fwhm = radius_px * numpy.sqrt(3.0) * PIXEL_M
        expected_encircled = 2.0 * radius_px * numpy.sqrt(0.8) * PIXEL_M

        numpy.testing.assert_allclose(metrics.rms_major_axis_length_m, expected_2sigma, rtol=0.05)
        numpy.testing.assert_allclose(metrics.rms_minor_axis_length_m, expected_2sigma, rtol=0.05)
        numpy.testing.assert_allclose(metrics.fwhm_major_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(metrics.fwhm_minor_axis_length_m, expected_fwhm, rtol=0.05)
        numpy.testing.assert_allclose(
            metrics.encircled_energy_diameter_m, expected_encircled, rtol=0.05
        )

    def test_uniform_disk_rms_to_fwhm_ratio(self) -> None:
        """For a uniform disk: RMS = R and FWHM = R*sqrt(3), so RMS/FWHM = 1/sqrt(3).
        Contrast with the Gaussian ratio of sqrt(2 ln 2) ~ 1.1774 — the shape of
        the intensity distribution materially changes the RMS-vs-FWHM relationship.
        """
        radius_px = 20.0
        size = 128
        y_idx, x_idx = numpy.mgrid[:size, :size]
        r_px = numpy.hypot(x_idx - (size - 1) / 2.0, y_idx - (size - 1) / 2.0)
        disk = (r_px <= radius_px).astype(numpy.float64)

        metrics = estimate_probe_size(disk, PIXEL_GEOMETRY)

        ratio = metrics.rms_major_axis_length_m / metrics.fwhm_major_axis_length_m
        numpy.testing.assert_allclose(ratio, 1.0 / numpy.sqrt(3.0), rtol=0.05)

    def test_elliptical_disk_round_trip(self) -> None:
        """For an axis-aligned uniform ellipse with semi-axes a >= b: the variance
        along each principal axis is (semi-axis)^2 / 4, so the reported 2-sigma
        length equals the semi-axis length.
        """
        a_px = 30.0  # semi-major along x
        b_px = 15.0  # semi-minor along y
        size = 192
        y_idx, x_idx = numpy.mgrid[:size, :size]
        dx = x_idx - (size - 1) / 2.0
        dy = y_idx - (size - 1) / 2.0
        ellipse = ((dx / a_px) ** 2 + (dy / b_px) ** 2 <= 1.0).astype(numpy.float64)

        metrics = estimate_probe_size(ellipse, PIXEL_GEOMETRY)

        assert metrics.rms_major_axis_length_m > metrics.rms_minor_axis_length_m
        numpy.testing.assert_allclose(metrics.rms_major_axis_length_m, a_px * PIXEL_M, rtol=0.05)
        numpy.testing.assert_allclose(metrics.rms_minor_axis_length_m, b_px * PIXEL_M, rtol=0.05)
        numpy.testing.assert_allclose(metrics.major_axis_tilt_rad, 0.0, atol=0.05)

    def test_zero_signal_returns_all_zeros(self) -> None:
        """When the cleaned image has no power, all metrics fall through to 0.0."""
        metrics = estimate_probe_size(numpy.zeros((64, 64)), PIXEL_GEOMETRY)

        assert metrics == ProbeSizeMetrics(
            major_axis_tilt_rad=0.0,
            minor_axis_tilt_rad=0.0,
            fwhm_major_axis_length_m=0.0,
            fwhm_minor_axis_length_m=0.0,
            rms_major_axis_length_m=0.0,
            rms_minor_axis_length_m=0.0,
            encircled_energy_diameter_m=0.0,
        )

    def test_rejects_non_2d_input(self) -> None:
        with pytest.raises(ValueError, match='2-dimensional'):
            estimate_probe_size(numpy.zeros((4, 8, 8)), PIXEL_GEOMETRY)

    @pytest.mark.parametrize('bad_fraction', [0.0, -0.1, 1.1, 2.0])
    def test_rejects_invalid_energy_fraction(self, bad_fraction: float) -> None:
        with pytest.raises(ValueError, match='energy_fraction'):
            estimate_probe_size(numpy.ones((16, 16)), PIXEL_GEOMETRY, energy_fraction=bad_fraction)

    def test_rejects_negative_mad_threshold(self) -> None:
        with pytest.raises(ValueError, match='mad_threshold'):
            estimate_probe_size(numpy.ones((16, 16)), PIXEL_GEOMETRY, mad_threshold=-1.0)


class TestComputeShannonEntropy:
    def test_uniform_distribution_has_max_normalized_entropy(self) -> None:
        numpy.testing.assert_allclose(compute_shannon_entropy(numpy.ones((16, 16))), 1.0)

    def test_one_hot_distribution_has_zero_entropy(self) -> None:
        values = numpy.zeros((16, 16))
        values[3, 5] = 1.0
        assert compute_shannon_entropy(values) == 0.0

    def test_empty_mass_returns_zero(self) -> None:
        assert compute_shannon_entropy(numpy.zeros((8, 8))) == 0.0

    def test_negative_values_are_clipped(self) -> None:
        # After clipping, one positive element remains -> zero entropy.
        values = numpy.full((4, 4), -1.0)
        values[0, 0] = 2.0
        assert compute_shannon_entropy(values) == 0.0

    def test_concentrated_has_lower_entropy_than_broad(self) -> None:
        narrow = _gaussian_2d(128, 128, 4.0, 4.0, tilt_rad=0.0)
        broad = _gaussian_2d(128, 128, 32.0, 32.0, tilt_rad=0.0)
        assert compute_shannon_entropy(narrow) < compute_shannon_entropy(broad)

    def test_raw_entropy_of_uniform_equals_log2_n(self) -> None:
        values = numpy.ones((8, 8))
        numpy.testing.assert_allclose(
            compute_shannon_entropy(values, normalize=False), numpy.log2(values.size)
        )


class TestEstimateProbeEntropy:
    def _gaussian_probe(self, sigma_px: float) -> Probe:
        amplitude = numpy.sqrt(_gaussian_2d(128, 128, sigma_px, sigma_px, tilt_rad=0.0))
        return Probe(amplitude.astype(numpy.complex128), PIXEL_GEOMETRY)

    def test_metrics_are_normalized_to_unit_interval(self) -> None:
        metrics = estimate_probe_entropy(self._gaussian_probe(16.0))
        assert 0.0 <= metrics.real_space_intensity_entropy <= 1.0
        assert 0.0 <= metrics.spectral_entropy <= 1.0

    def test_fourier_duality_direction(self) -> None:
        """A tightly focused probe is concentrated in real space (low real-space
        entropy) but spread in frequency (high spectral entropy); a broad probe
        is the reverse."""
        narrow = estimate_probe_entropy(self._gaussian_probe(4.0))
        broad = estimate_probe_entropy(self._gaussian_probe(32.0))

        assert narrow.real_space_intensity_entropy < broad.real_space_intensity_entropy
        assert narrow.spectral_entropy > broad.spectral_entropy


class TestProbeSequenceSlicing:
    """Regression tests for ProbeSequence.__getitem__.

    The slice path used to build its indices from
    ``range(index.start, index.stop, index.step)``, so every slice with an
    implicit bound raised TypeError. Only a fully-specified positive-step slice
    such as ``seq[0:2:1]`` worked.
    """

    def _sequence(self, num_positions: int = 5) -> ProbeSequence:
        """Two coherent modes over *num_positions* scan positions, so len() > 1."""
        array = numpy.arange(2 * 1 * 4 * 4, dtype=numpy.complex128).reshape(2, 1, 4, 4)
        opr_weights = numpy.arange(num_positions * 2, dtype=float).reshape(num_positions, 2)
        return ProbeSequence(array, opr_weights, PIXEL_GEOMETRY)

    @pytest.mark.parametrize(
        ('index', 'expected_length'),
        [
            (slice(None, 2), 2),
            (slice(1, 3), 2),
            (slice(None), 5),
            (slice(None, None, 2), 3),
            (slice(-2, None), 2),
            (slice(None, 99), 5),
        ],
    )
    def test_slice_returns_expected_number_of_probes(
        self, index: slice, expected_length: int
    ) -> None:
        probes = self._sequence()[index]
        assert len(probes) == expected_length
        assert all(isinstance(probe, Probe) for probe in probes)

    def test_empty_slice_returns_empty_list(self) -> None:
        assert len(self._sequence()[3:1]) == 0

    def test_negative_step_reverses(self) -> None:
        sequence = self._sequence()
        reversed_probes = sequence[::-1]
        assert len(reversed_probes) == 5

        for offset, probe in enumerate(reversed_probes):
            expected = sequence[4 - offset]
            numpy.testing.assert_array_equal(probe.get_array(), expected.get_array())

    def test_slice_agrees_with_integer_indexing(self) -> None:
        sequence = self._sequence()

        for offset, probe in enumerate(sequence[1:4]):
            expected = sequence[1 + offset]
            numpy.testing.assert_array_equal(probe.get_array(), expected.get_array())

    def test_slice_without_opr_weights(self) -> None:
        """__len__ reports 1 when there are no OPR weights, so [:] yields one probe."""
        array = numpy.ones((1, 1, 4, 4), dtype=numpy.complex128)
        sequence = ProbeSequence(array, None, PIXEL_GEOMETRY)

        assert len(sequence) == 1
        assert len(sequence[:]) == 1


class TestProbeGeometryFromFarField:
    """The Fraunhofer sample-plane relation dx_sample = lambda * |z| / (N * dx_detector)."""

    def test_returns_expected_pixel_size_for_square_setup(self) -> None:
        detector = PixelGeometry(width_m=75e-6, height_m=75e-6)
        wavelength_m = 1.55e-10
        distance_m = 0.1

        geometry = ProbeGeometry.from_far_field(
            detector,
            ImageExtent(width_px=256, height_px=256),
            wavelength_m=wavelength_m,
            distance_m=distance_m,
        )

        expected_pixel_m = wavelength_m * distance_m / (256 * 75e-6)
        assert geometry.width_px == 256
        assert geometry.height_px == 256
        numpy.testing.assert_allclose(geometry.pixel_width_m, expected_pixel_m, rtol=1.0e-12)
        numpy.testing.assert_allclose(geometry.pixel_height_m, expected_pixel_m, rtol=1.0e-12)

    def test_maps_each_axis_independently_for_non_square(self) -> None:
        detector = PixelGeometry(width_m=50e-6, height_m=100e-6)
        wavelength_m = 1.0e-10
        distance_m = 2.0

        geometry = ProbeGeometry.from_far_field(
            detector,
            ImageExtent(width_px=256, height_px=128),
            wavelength_m=wavelength_m,
            distance_m=distance_m,
        )

        assert geometry.height_px == 128
        assert geometry.width_px == 256
        numpy.testing.assert_allclose(
            geometry.pixel_width_m, wavelength_m * distance_m / (256 * 50e-6), rtol=1.0e-12
        )
        numpy.testing.assert_allclose(
            geometry.pixel_height_m, wavelength_m * distance_m / (128 * 100e-6), rtol=1.0e-12
        )

    def test_negative_distance_uses_magnitude(self) -> None:
        detector = PixelGeometry(width_m=75e-6, height_m=75e-6)
        wavelength_m = 1.55e-10
        extent = ImageExtent(width_px=256, height_px=256)

        forward = ProbeGeometry.from_far_field(
            detector, extent, wavelength_m=wavelength_m, distance_m=0.1
        )
        backward = ProbeGeometry.from_far_field(
            detector, extent, wavelength_m=wavelength_m, distance_m=-0.1
        )

        assert forward == backward


class TestProbeGeometryStr:
    def test_square_pixels_collapse_to_single_value(self) -> None:
        geometry = ProbeGeometry(
            width_px=256, height_px=256, pixel_width_m=8.06e-9, pixel_height_m=8.06e-9
        )
        rendered = str(geometry)
        assert rendered.startswith('256 x 256 px @ ')
        # Single "x" between px count and pitch means no second axis rendered.
        assert rendered.count(' x ') == 1
        assert '/px' in rendered

    def test_non_square_pixels_render_both_axes(self) -> None:
        geometry = ProbeGeometry(
            width_px=128, height_px=256, pixel_width_m=8.06e-9, pixel_height_m=4.03e-9
        )
        rendered = str(geometry)
        assert rendered.startswith('128 x 256 px @ ')
        # Two "x" separators: one for the pixel count, one for the pitch axes.
        assert rendered.count(' x ') == 2
        assert '/px' in rendered


class TestProbeGeometryResolvePatchBounds:
    """The (N-1)/2 patch-center split shared with ``get_transverse_coordinates``."""

    def test_even_extent_integer_center_yields_zero_offset(self) -> None:
        geometry = ProbeGeometry(
            width_px=4, height_px=4, pixel_width_m=PIXEL_M, pixel_height_m=PIXEL_M
        )

        bounds = geometry.resolve_patch_bounds(cx=10.5, cy=10.5)

        assert bounds == PatchBounds(
            x_slice=slice(9, 13),
            y_slice=slice(9, 13),
            dx=0.0,
            dy=0.0,
        )

    def test_odd_extent_integer_center_yields_half_pixel_offset(self) -> None:
        geometry = ProbeGeometry(
            width_px=5, height_px=5, pixel_width_m=PIXEL_M, pixel_height_m=PIXEL_M
        )

        bounds = geometry.resolve_patch_bounds(cx=10.5, cy=10.5)

        assert bounds.x_slice == slice(8, 13)
        assert bounds.y_slice == slice(8, 13)
        numpy.testing.assert_allclose(bounds.dx, 0.5, rtol=1.0e-12)
        numpy.testing.assert_allclose(bounds.dy, 0.5, rtol=1.0e-12)

    def test_negative_center_truncates_toward_zero_not_floor(self) -> None:
        """Pins Python ``int()`` semantics: for ``cx - (W-1)/2 < 0`` the lower
        corner is ``ceil``, not ``floor``. Callers must keep object-canvas
        coordinates non-negative in practice; the helper's docstring says so.
        """
        geometry = ProbeGeometry(
            width_px=3, height_px=5, pixel_width_m=PIXEL_M, pixel_height_m=2 * PIXEL_M
        )

        bounds = geometry.resolve_patch_bounds(cx=-0.4, cy=-1.6)

        # int(-0.4 - 1.0) == int(-1.4) == -1  (floor would be -2)
        # int(-1.6 - 2.0) == int(-3.6) == -3  (floor would be -4)
        assert bounds.x_slice == slice(-1, 2)
        assert bounds.y_slice == slice(-3, 2)
        # dx = cx - (x_lower + rx) = -0.4 - (-1 + 1.0) = -0.4
        # dy = cy - (y_lower + ry) = -1.6 - (-3 + 2.0) = -0.6
        numpy.testing.assert_allclose(bounds.dx, -0.4, atol=1.0e-12)
        numpy.testing.assert_allclose(bounds.dy, -0.6, atol=1.0e-12)


class TestProbeSequenceFromProbe:
    def test_wraps_probe_as_length_1_sequence_with_matching_array_and_geometry(self) -> None:
        rng = numpy.random.default_rng(0)
        pixel_geometry = PixelGeometry(width_m=1e-9, height_m=2e-9)
        array = (rng.standard_normal((3, 8, 8)) + 1j * rng.standard_normal((3, 8, 8))).astype(
            numpy.complex128
        )
        probe = Probe(array=array, pixel_geometry=pixel_geometry)

        sequence = ProbeSequence.from_probe(probe)

        assert len(sequence) == 1
        assert sequence.num_coherent_modes == 1
        assert sequence.num_incoherent_modes == 3
        assert sequence.get_opr_weights_or_none() is None
        assert sequence.get_pixel_geometry() == pixel_geometry
        numpy.testing.assert_array_equal(sequence.get_probe_no_opr().get_array(), array)


def _gaussian_caustic_stack(
    coordinate_m: numpy.ndarray,
    waist_m: float,
    waist_coordinate_m: float,
    rayleigh_range_m: float,
    *,
    size: int = 48,
) -> numpy.ndarray:
    """Build a ``(num_steps, 1, size, size)`` stack holding a Gaussian beam.

    The width follows the caustic ``w(z) = w0 sqrt(1 + ((z - z0) / zR)^2)``, normalized
    so total power is conserved along z, and each plane carries the matching spherical
    phase ``-k r^2 / 2R(z)``. Curvature is written as ``1 / R = z / (z^2 + zR^2)``,
    which stays finite at the waist, where it vanishes and leaves a flat wavefront.

    The phase is what makes this a beam rather than a stack of blurred spots: without
    it the wavefront metrics have nothing to measure and read zero at every plane.
    """
    y_idx, x_idx = numpy.mgrid[:size, :size]
    dx = (x_idx - (size - 1) / 2.0) * PIXEL_M
    dy = (y_idx - (size - 1) / 2.0) * PIXEL_M
    r2 = numpy.square(dx) + numpy.square(dy)
    # Implied by the waist and Rayleigh range via zR = pi w0^2 / lambda.
    wavenumber = 2.0 * rayleigh_range_m / numpy.square(waist_m)

    planes = []

    for z_m in coordinate_m:
        offset_m = z_m - waist_coordinate_m
        scale = numpy.sqrt(1.0 + (offset_m / rayleigh_range_m) ** 2)
        width_m = waist_m * scale
        inverse_radius = offset_m / (numpy.square(offset_m) + numpy.square(rayleigh_range_m))
        phase = -0.5 * wavenumber * r2 * inverse_radius
        # Amplitude, not intensity: dividing by the width keeps sum(|psi|^2) constant.
        envelope = numpy.exp(-r2 / numpy.square(width_m)) / width_m
        planes.append(envelope * numpy.exp(1j * phase))

    return numpy.asarray(planes, dtype=numpy.complex128)[:, numpy.newaxis]


def _caustic_probe(
    waist_coordinate_m: float,
    *,
    num_steps: int = 21,
    begin_coordinate_m: float = -100.0 * PIXEL_M,
    end_coordinate_m: float = 100.0 * PIXEL_M,
) -> PropagatedWavefield:
    coordinate_m = numpy.linspace(begin_coordinate_m, end_coordinate_m, num_steps)
    wavefield = _gaussian_caustic_stack(
        coordinate_m,
        waist_m=8.0 * PIXEL_M,
        waist_coordinate_m=waist_coordinate_m,
        rayleigh_range_m=40.0 * PIXEL_M,
    )
    return PropagatedWavefield(
        wavefield=wavefield,
        begin_coordinate_m=begin_coordinate_m,
        end_coordinate_m=end_coordinate_m,
        pixel_geometry=PIXEL_GEOMETRY,
    )


class TestComputeRmsContrast:
    def test_constant_image_has_no_contrast(self) -> None:
        assert compute_rms_contrast(numpy.full((16, 16), 3.5)) == 0.0

    def test_two_level_image_matches_closed_form(self) -> None:
        """Half zeros and half 2a has mean a and standard deviation a, so a contrast of 1."""
        image = numpy.concatenate([numpy.zeros(128), numpy.full(128, 7.0)])

        numpy.testing.assert_allclose(compute_rms_contrast(image), 1.0, rtol=1e-12)

    def test_zero_image_returns_zero_rather_than_dividing(self) -> None:
        assert compute_rms_contrast(numpy.zeros((8, 8))) == 0.0

    def test_invariant_under_rescaling(self) -> None:
        rng = numpy.random.default_rng(3)
        image = rng.random((32, 32))

        numpy.testing.assert_allclose(
            compute_rms_contrast(image), compute_rms_contrast(1e6 * image), rtol=1e-12
        )


class TestComputeAmplitudeDeviation:
    def test_uniform_amplitude_has_no_deviation(self) -> None:
        wavefield = numpy.exp(1j * numpy.linspace(0.0, 6.0, 64)).reshape(8, 8)

        numpy.testing.assert_allclose(compute_amplitude_deviation(wavefield), 0.0, atol=1e-12)

    def test_zero_wavefield_returns_zero_rather_than_dividing(self) -> None:
        assert compute_amplitude_deviation(numpy.zeros((8, 8), dtype=numpy.complex128)) == 0.0

    def test_invariant_under_rescaling(self) -> None:
        rng = numpy.random.default_rng(4)
        wavefield = rng.random((16, 16)) + 1j * rng.random((16, 16))

        numpy.testing.assert_allclose(
            compute_amplitude_deviation(wavefield),
            compute_amplitude_deviation(1e6 * wavefield),
            rtol=1e-12,
        )


class TestComputePhaseDeviation:
    def test_flat_wavefront_has_no_deviation(self) -> None:
        wavefield = numpy.full((16, 16), 2.0 + 0.0j)

        numpy.testing.assert_allclose(compute_phase_deviation_rad(wavefield), 0.0, atol=1e-12)

    def test_recovers_sigma_of_a_wrapped_normal_phase(self) -> None:
        """For phase ~ N(0, sigma), the resultant is exp(-sigma^2/2), so sqrt(-2 ln R)
        returns sigma itself."""
        sigma = 0.3
        rng = numpy.random.default_rng(5)
        wavefield = numpy.exp(1j * rng.normal(0.0, sigma, size=1 << 16))

        numpy.testing.assert_allclose(compute_phase_deviation_rad(wavefield), sigma, rtol=0.02)

    def test_is_insensitive_to_the_branch_cut(self) -> None:
        """Phase tightly clustered about pi straddles the +/-pi wrap. The circular
        statistic must see a narrow distribution where a plain standard deviation of
        the angle sees a maximally wide one."""
        rng = numpy.random.default_rng(6)
        phase = numpy.pi + rng.normal(0.0, 0.05, size=(64, 64))
        wavefield = numpy.exp(1j * phase)

        deviation = compute_phase_deviation_rad(wavefield)

        numpy.testing.assert_allclose(deviation, 0.05, rtol=0.1)
        assert numpy.std(numpy.angle(wavefield)) > 3.0  # what the naive approach reports

    def test_is_dominated_by_the_bright_pixels(self) -> None:
        """A bright coherent core plus a dim incoherent halo is a low-deviation
        wavefield, because the weighting is by intensity."""
        rng = numpy.random.default_rng(7)
        wavefield = 1e-3 * numpy.exp(2j * numpy.pi * rng.random((32, 32)))
        wavefield[12:20, 12:20] = 1.0

        assert compute_phase_deviation_rad(wavefield) < 0.01

    def test_scattered_phase_exceeds_clustered_phase(self) -> None:
        rng = numpy.random.default_rng(8)
        scattered = numpy.exp(2j * numpy.pi * rng.random((64, 64)))
        clustered = numpy.exp(1j * rng.normal(0.0, 0.1, size=(64, 64)))

        assert compute_phase_deviation_rad(scattered) > compute_phase_deviation_rad(clustered)

    def test_zero_wavefield_returns_zero(self) -> None:
        assert compute_phase_deviation_rad(numpy.zeros((8, 8), dtype=numpy.complex128)) == 0.0


class TestEstimateFocalPlane:
    def test_recovers_an_off_grid_parabola_vertex_exactly(self) -> None:
        """Three-point parabolic interpolation is exact for a parabola, so a vertex
        placed deliberately between samples comes back to machine precision."""
        coordinate_m = numpy.linspace(-10.0, 10.0, 11)  # spacing 2.0
        vertex_m = 1.3
        value = 3.0 * numpy.square(coordinate_m - vertex_m) + 0.5

        plane = estimate_focal_plane(coordinate_m, value, FocusPolarity.MINIMUM)

        assert plane.is_refined
        numpy.testing.assert_allclose(plane.coordinate_m, vertex_m, rtol=1e-12)
        numpy.testing.assert_allclose(plane.value, 0.5, rtol=1e-12)
        assert plane.step == 6  # the sample nearest 1.3

    def test_maximum_polarity_mirrors_minimum(self) -> None:
        coordinate_m = numpy.linspace(-10.0, 10.0, 11)
        vertex_m = 1.3
        value = 3.0 * numpy.square(coordinate_m - vertex_m) + 0.5

        maximum = estimate_focal_plane(coordinate_m, -value, FocusPolarity.MAXIMUM)

        assert maximum.is_refined
        numpy.testing.assert_allclose(maximum.coordinate_m, vertex_m, rtol=1e-12)

    def test_extremum_at_either_endpoint_is_not_refined(self) -> None:
        coordinate_m = numpy.linspace(0.0, 10.0, 11)

        rising = estimate_focal_plane(coordinate_m, coordinate_m, FocusPolarity.MINIMUM)
        falling = estimate_focal_plane(coordinate_m, -coordinate_m, FocusPolarity.MINIMUM)

        assert not rising.is_refined
        assert rising.step == 0
        assert not falling.is_refined
        assert falling.step == 10

    def test_flat_curve_falls_back_to_the_grid(self) -> None:
        coordinate_m = numpy.linspace(0.0, 10.0, 11)

        plane = estimate_focal_plane(coordinate_m, numpy.zeros(11), FocusPolarity.MINIMUM)

        assert not plane.is_refined

    def test_short_curves_return_the_extremal_sample(self) -> None:
        single = estimate_focal_plane(numpy.array([4.0]), numpy.array([7.0]), FocusPolarity.MINIMUM)
        pair = estimate_focal_plane(
            numpy.array([0.0, 1.0]), numpy.array([5.0, 2.0]), FocusPolarity.MINIMUM
        )

        assert not single.is_refined
        assert single.step == 0
        numpy.testing.assert_allclose(single.coordinate_m, 4.0, rtol=1e-12)
        numpy.testing.assert_allclose(single.value, 7.0, rtol=1e-12)
        assert not pair.is_refined
        assert pair.step == 1
        numpy.testing.assert_allclose(pair.coordinate_m, 1.0, rtol=1e-12)

    def test_mismatched_or_empty_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            estimate_focal_plane(numpy.zeros(4), numpy.zeros(5), FocusPolarity.MINIMUM)

        with pytest.raises(ValueError):
            estimate_focal_plane(numpy.zeros(0), numpy.zeros(0), FocusPolarity.MINIMUM)


class TestComputeProbeFocusCurves:
    def test_every_metric_locates_a_known_off_grid_waist(self) -> None:
        """All ten metrics measure the same beam, so all ten should agree on where it
        is narrowest, to within a fraction of the step size."""
        waist_m = 13.0 * PIXEL_M  # off-grid: the samples are spaced 10 nm apart
        curves = compute_probe_focus_curves(_caustic_probe(waist_m))

        for series in curves.series:
            plane = curves.get_focal_plane(series.metric)
            assert abs(plane.coordinate_m - waist_m) < 10.0 * PIXEL_M, series.metric

    def test_polarity_matches_how_each_metric_behaves_at_focus(self) -> None:
        """Widths and entropy bottom out at the waist; peak intensity and contrast top
        out there. A wrong polarity would send the search to the opposite end."""
        curves = compute_probe_focus_curves(_caustic_probe(0.0))
        focus_step = curves.series[0].value.argmin()

        for series in curves.series:
            if series.metric.polarity is FocusPolarity.MINIMUM:
                assert series.value.argmin() == focus_step, series.metric
            else:
                assert series.value.argmax() == focus_step, series.metric

    def test_series_cover_every_metric_and_span_the_requested_range(self) -> None:
        curves = compute_probe_focus_curves(_caustic_probe(0.0, num_steps=21))

        assert {series.metric for series in curves.series} == set(ProbeFocusMetric)
        assert curves.coordinate_m.shape == (21,)
        numpy.testing.assert_allclose(curves.coordinate_m[0], -100.0 * PIXEL_M)
        numpy.testing.assert_allclose(curves.coordinate_m[-1], 100.0 * PIXEL_M)

        for series in curves.series:
            assert series.value.shape == (21,), series.metric

    def test_lengths_are_reported_in_meters(self) -> None:
        """The waist amplitude is exp(-r^2/w^2) with w = 8 px at 1 nm per pixel, so the
        intensity has sigma = w/2 and the reported 2-sigma width is w itself, 8 nm --
        a value only plausible if the series is in meters rather than pixels."""
        curves = compute_probe_focus_curves(_caustic_probe(0.0))

        plane = curves.get_focal_plane(ProbeFocusMetric.RMS_MAJOR)

        assert ProbeFocusMetric.RMS_MAJOR.si_unit == 'm'
        numpy.testing.assert_allclose(plane.value, 8.0 * PIXEL_M, rtol=0.05)

    def test_get_series_and_get_focal_plane_reject_an_unsampled_metric(self) -> None:
        curves = compute_probe_focus_curves(_caustic_probe(0.0))
        pruned = ProbeFocusCurves(
            coordinate_m=curves.coordinate_m,
            series=[s for s in curves.series if s.metric is not ProbeFocusMetric.RMS_CONTRAST],
        )

        with pytest.raises(KeyError):
            pruned.get_series(ProbeFocusMetric.RMS_CONTRAST)

        with pytest.raises(KeyError):
            pruned.get_focal_plane(ProbeFocusMetric.RMS_CONTRAST)

    def test_out_of_range_mode_raises(self) -> None:
        probe = _caustic_probe(0.0)

        with pytest.raises(ValueError):
            compute_probe_focus_curves(probe, mode=1)

        with pytest.raises(ValueError):
            compute_probe_focus_curves(probe, mode=-1)


class TestShiftProbe:
    """shift_probe mirrors shift_object: a Fourier phase ramp over the last two axes."""

    @staticmethod
    def _probe(array: numpy.ndarray) -> Probe:
        return Probe(array=array.astype(complex), pixel_geometry=PIXEL_GEOMETRY)

    def test_zero_shift_returns_the_same_object(self) -> None:
        probe = self._probe(numpy.ones((2, 8, 8)))

        assert shift_probe(probe, shift_y_px=0.0, shift_x_px=0.0) is probe

    def test_integer_shift_rolls_the_array(self) -> None:
        array = numpy.zeros((1, 8, 8), dtype=complex)
        array[0, 2, 3] = 1.0
        shifted = shift_probe(self._probe(array), shift_y_px=1.0, shift_x_px=2.0)

        numpy.testing.assert_allclose(
            shifted.get_array(), numpy.roll(array, (1, 2), axis=(-2, -1)), atol=1e-10
        )

    def test_subpixel_shift_is_exact_for_a_bandlimited_signal(self) -> None:
        """The reason for a phase ramp rather than bilinear interpolation."""
        n = 16
        x = numpy.arange(n)
        array = numpy.exp(2j * numpy.pi * x / n)[None, None, :] * numpy.ones((1, n, 1))
        shifted = shift_probe(self._probe(array), shift_y_px=0.0, shift_x_px=0.5)
        expected = numpy.exp(2j * numpy.pi * (x - 0.5) / n)[None, None, :] * numpy.ones((1, n, 1))

        numpy.testing.assert_allclose(shifted.get_array(), expected, atol=1e-10)

    def test_every_incoherent_mode_is_shifted(self) -> None:
        array = numpy.zeros((3, 8, 8), dtype=complex)
        array[:, 4, 4] = [1.0, 2.0, 3.0]
        shifted = shift_probe(self._probe(array), shift_y_px=1.0, shift_x_px=2.0)

        numpy.testing.assert_allclose(
            numpy.abs(shifted.get_array()[:, 5, 6]), [1.0, 2.0, 3.0], atol=1e-10
        )

    def test_total_intensity_is_conserved(self) -> None:
        rng = numpy.random.default_rng(4)
        array = rng.normal(size=(2, 8, 8)) + 1j * rng.normal(size=(2, 8, 8))
        shifted = shift_probe(self._probe(array), shift_y_px=1.7, shift_x_px=-0.4)

        numpy.testing.assert_allclose(
            numpy.sum(numpy.abs(shifted.get_array()) ** 2),
            numpy.sum(numpy.abs(array) ** 2),
            rtol=1e-10,
        )

    def test_the_shift_wraps_around(self) -> None:
        """Documented departure from scipy.ndimage.shift's edge extension."""
        array = numpy.zeros((1, 8, 8), dtype=complex)
        array[0, 0, 0] = 1.0
        shifted = shift_probe(self._probe(array), shift_y_px=-1.0, shift_x_px=0.0)

        numpy.testing.assert_allclose(numpy.abs(shifted.get_array()[0, 7, 0]), 1.0, atol=1e-10)

    def test_pixel_geometry_is_preserved(self) -> None:
        shifted = shift_probe(self._probe(numpy.ones((1, 8, 8))), shift_y_px=1.0, shift_x_px=1.0)

        assert shifted.get_pixel_geometry().width_m == PIXEL_M
