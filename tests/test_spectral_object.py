"""The spectral object estimate: its operator, its eigenvector, and what it recovers.

Recovery is checked on data simulated by the multislice forward model, with Poisson
noise and detector gaps, so the operator is pinned against the model that produced the
data rather than against itself.
"""

from collections.abc import Sequence
from functools import cache
from typing import Any

import numpy
import numpy.testing
import pytest
from scipy.ndimage import gaussian_filter

from ptychodus.api.accumulate import DiffractionMoments, compute_diffraction_moments
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.geometry import GeometryNotDefinedError, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.patch import accumulate_probe_intensity_on, iter_probe_patches_on
from ptychodus.api.probe import Probe, ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import FresnelTransformPropagator, PropagatorParameters
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.simulate.object import (
    SpectralPreprocessing,
    _build_propagator_pair,
    _compute_spectral_weights,
    compute_stxm_dpc_signals,
    generate_spectral_object,
    generate_stxm_dpc_object,
)
from ptychodus.api.simulate.probe import propagate_probe
from ptychodus.plugins.object.phase_retrieval import SpectralObjectGenerator, SpectralOptions

_PHOTON_ENERGY_EV = 10_000.0
_WAVELENGTH_M = energy_eV_to_wavelength_m(_PHOTON_ENERGY_EV)
_PIXEL_SIZE_M = 50e-9
_PIXEL_GEOMETRY = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
_FAR_FIELD_DISTANCE_M = 2.0
_NEAR_FIELD_DISTANCE_M = 1.0e-3
_OBJECT_PX = 64
_PROBE_PX = 32
_INTERIOR = (slice(20, 44), slice(20, 44))


def _geometry(extent_px: int = _OBJECT_PX) -> ObjectGeometry:
    return ObjectGeometry(
        width_px=extent_px,
        height_px=extent_px,
        pixel_width_m=_PIXEL_SIZE_M,
        pixel_height_m=_PIXEL_SIZE_M,
        center_x_m=0.0,
        center_y_m=0.0,
    )


def _grid_positions(half_span_px: int, step_px: int, *, seed: int = 1) -> list[ProbePosition]:
    rng = numpy.random.default_rng(seed)
    positions: list[ProbePosition] = []

    for iy in range(-half_span_px, half_span_px + 1, step_px):
        for ix in range(-half_span_px, half_span_px + 1, step_px):
            positions.append(
                ProbePosition(
                    index=len(positions),
                    x_m=(ix + rng.uniform(-0.5, 0.5)) * _PIXEL_SIZE_M,
                    y_m=(iy + rng.uniform(-0.5, 0.5)) * _PIXEL_SIZE_M,
                )
            )

    return positions


def _radius_px(extent_px: int) -> numpy.ndarray:
    yy, xx = numpy.indices((extent_px, extent_px)) - (extent_px - 1) / 2
    return numpy.hypot(xx, yy)


def _defocused_disk_probe(*, extent_px: int = _PROBE_PX, amplitude: float = 30.0) -> Probe:
    """A disk carried a little past focus: the illumination of a zone plate or pinhole."""
    disk = (_radius_px(extent_px) < extent_px / 5) * amplitude
    focused = Probe(array=disk.astype(complex), pixel_geometry=_PIXEL_GEOMETRY)
    return propagate_probe(focused, photon_wavelength_m=_WAVELENGTH_M, propagation_distance_m=20e-6)


def _diffuser_probe(*, amplitude: float = 10.0) -> Probe:
    """A disk through a random phase screen: the structured illumination of near field."""
    rng = numpy.random.default_rng(7)
    screen = gaussian_filter(rng.standard_normal((_PROBE_PX, _PROBE_PX)), 1.5)
    phase = 2.0 * screen / screen.std()
    array = (_radius_px(_PROBE_PX) < 0.3 * _PROBE_PX) * amplitude * numpy.exp(1j * phase)
    return Probe(array=array.astype(complex), pixel_geometry=_PIXEL_GEOMETRY)


@cache
def _truth() -> numpy.ndarray:
    """A weakly absorbing object of smooth random phase, one radian rms."""
    rng = numpy.random.default_rng(0)
    phase = gaussian_filter(rng.standard_normal((_OBJECT_PX, _OBJECT_PX)), 3)
    absorber = gaussian_filter((rng.uniform(size=(_OBJECT_PX, _OBJECT_PX)) > 0.5) * 1.0, 2)
    return (1.0 - 0.2 * absorber) * numpy.exp(1j * phase / phase.std())


def _simulate(
    probe: Probe,
    positions: Sequence[ProbePosition],
    *,
    far_field: bool = True,
    with_noise: bool = True,
) -> AssembledDiffractionData:
    metadata = ProductMetadata(
        name='spectral',
        comments='',
        detector_distance_m=_FAR_FIELD_DISTANCE_M if far_field else _NEAR_FIELD_DISTANCE_M,
        photon_energy_eV=_PHOTON_ENERGY_EV,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
        far_field=far_field,
    )
    product = Product(
        metadata=metadata,
        probe_positions=ProbePositionSequence(list(positions)),
        probes=ProbeEnsemble(
            array=probe.get_array()[numpy.newaxis], opr_weights=None, pixel_geometry=_PIXEL_GEOMETRY
        ),
        object_=Object(
            array=_truth()[numpy.newaxis],
            pixel_geometry=_PIXEL_GEOMETRY,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )
    data = generate_diffraction_data(product, numpy.random.default_rng(1) if with_noise else None)
    # Module gaps, reading zero as a photon-counting detector's do.
    bad_pixels = numpy.zeros((_PROBE_PX, _PROBE_PX), dtype=bool)
    bad_pixels[:, 21:23] = True
    bad_pixels[11, :] = True
    patterns = numpy.array(data.get_patterns())
    patterns[:, bad_pixels] = 0.0
    return _with(data, patterns=patterns, bad_pixels=bad_pixels)


def _with(data: AssembledDiffractionData, **changes: Any) -> AssembledDiffractionData:
    arguments: dict[str, Any] = dict(
        indexes=numpy.array(data.get_indexes()),
        patterns=numpy.array(data.get_patterns()),
        pixel_geometry=data.get_pixel_geometry(),
        bad_pixels=numpy.array(data.get_bad_pixels()),
        background=data.get_background(),
        flat_field=data.get_flat_field(),
    )
    arguments.update(changes)
    return AssembledDiffractionData(**arguments)


@cache
def _far_field_case() -> tuple[Probe, list[ProbePosition], AssembledDiffractionData]:
    probe = _defocused_disk_probe()
    positions = _grid_positions(14, 4)
    return probe, positions, _simulate(probe, positions)


def _estimate(
    data: AssembledDiffractionData,
    positions: Sequence[ProbePosition],
    probe: Probe,
    *,
    far_field: bool = True,
    geometry: ObjectGeometry | None = None,
    **kwargs: Any,
) -> numpy.ndarray:
    object_ = generate_spectral_object(
        _geometry() if geometry is None else geometry,
        data,
        positions,
        probe=probe,
        photon_wavelength_m=_WAVELENGTH_M,
        propagation_distance_m=_FAR_FIELD_DISTANCE_M if far_field else _NEAR_FIELD_DISTANCE_M,
        far_field=far_field,
        **kwargs,
    )
    assert object_.num_layers == 1
    return object_.get_array()[0]


def _correlation(estimate: numpy.ndarray, reference: numpy.ndarray) -> float:
    """``|<a, b>| / (|a| |b|)`` over the interior: one when equal up to a complex scale."""
    a = estimate[_INTERIOR].ravel()
    b = reference[_INTERIOR].ravel()
    return float(abs(numpy.vdot(a, b)) / numpy.linalg.norm(a) / numpy.linalg.norm(b))


def _support(probe: Probe, positions: Sequence[ProbePosition]) -> numpy.ndarray:
    illumination = accumulate_probe_intensity_on(
        ProbeEnsemble.from_probe(probe), positions, _geometry()
    )
    return illumination >= 0.05 * illumination.max()


# -- the preprocessing -----------------------------------------------------------------


def test_the_optimal_preprocessing_is_floored_without_dividing_by_zero() -> None:
    normalized = numpy.array([[0.0, 0.5, 2.0 / 3.0, 1.0, 4.0]])

    with numpy.errstate(all='raise'):
        weights = _compute_spectral_weights(
            normalized, preprocessing=SpectralPreprocessing.OPTIMAL, lower_bound=0.5, threshold=0.0
        )

    # The floor -0.5 holds at and below y = 1 / 1.5; above it the map is 1 - 1/y.
    numpy.testing.assert_allclose(weights, [[-0.5, -0.5, -0.5, 0.0, 0.75]])


def test_the_threshold_preprocessing_keeps_what_exceeds_the_cut() -> None:
    normalized = numpy.array([[0.5, 2.0, 3.0, 2.5]])

    weights = _compute_spectral_weights(
        normalized, preprocessing=SpectralPreprocessing.THRESHOLD, lower_bound=0.5, threshold=2.5
    )

    numpy.testing.assert_array_equal(weights, [[0.0, 0.0, 1.0, 0.0]])


def test_the_identity_preprocessing_returns_a_copy() -> None:
    normalized = numpy.array([[0.5, 2.0]])

    weights = _compute_spectral_weights(
        normalized, preprocessing=SpectralPreprocessing.IDENTITY, lower_bound=0.5, threshold=0.0
    )

    numpy.testing.assert_array_equal(weights, normalized)
    assert not numpy.shares_memory(weights, normalized)


# -- the operator ----------------------------------------------------------------------


@pytest.mark.parametrize(
    'far_field, distance_m, pixel_size_m',
    [
        (True, 2.0, 50e-9),  # Fresnel transform onto the reciprocal grid
        (False, 1.0e-3, 1.0e-6),  # angular spectrum
        (False, 0.5, 1.0e-7),  # near field past the crossover: Fresnel transform
    ],
)
def test_the_adjoint_is_the_adjoint(
    far_field: bool, distance_m: float, pixel_size_m: float
) -> None:
    parameters = PropagatorParameters(
        wavelength_m=_WAVELENGTH_M,
        width_px=32,
        height_px=24,
        pixel_width_m=pixel_size_m,
        pixel_height_m=1.1 * pixel_size_m,
        propagation_distance_m=distance_m,
    )
    forward, backward = _build_propagator_pair(parameters, far_field=far_field)
    rng = numpy.random.default_rng(2)
    x = rng.standard_normal((3, 24, 32)) + 1j * rng.standard_normal((3, 24, 32))
    y = rng.standard_normal((3, 24, 32)) + 1j * rng.standard_normal((3, 24, 32))

    lhs = numpy.vdot(forward.propagate(x), y)
    rhs = numpy.vdot(x, backward.propagate(y))

    assert abs(lhs - rhs) <= 1e-12 * abs(lhs)


def test_the_estimate_is_the_leading_eigenvector_of_the_dense_covariance() -> None:
    """On a canvas small enough to write ``Z`` down, its top eigenvector is the estimate."""
    extent_px, probe_px = 16, 8
    geometry = _geometry(extent_px)
    probe = _defocused_disk_probe(extent_px=probe_px)
    positions = [
        ProbePosition(index=index, x_m=x_px * _PIXEL_SIZE_M, y_m=y_px * _PIXEL_SIZE_M)
        for index, (x_px, y_px) in enumerate(
            [(-3.2, -2.9), (0.1, -3.0), (3.3, -3.1), (-2.8, 0.2), (0.0, 0.0), (3.1, 0.3)]
        )
    ]
    rng = numpy.random.default_rng(4)
    patterns = rng.poisson(50.0 * rng.uniform(size=(len(positions), probe_px, probe_px)))
    data = AssembledDiffractionData(
        numpy.arange(len(positions)),
        patterns.astype(float),
        _PIXEL_GEOMETRY,
        numpy.zeros((probe_px, probe_px), dtype=bool),
    )
    lower_bound = 0.5
    normalized = patterns / patterns.mean()
    with numpy.errstate(divide='ignore'):
        weights = numpy.maximum(1.0 - 1.0 / normalized, -lower_bound)

    propagator = FresnelTransformPropagator(
        PropagatorParameters(
            wavelength_m=_WAVELENGTH_M,
            width_px=probe_px,
            height_px=probe_px,
            pixel_width_m=_PIXEL_SIZE_M,
            pixel_height_m=_PIXEL_SIZE_M,
            propagation_distance_m=_FAR_FIELD_DISTANCE_M,
        )
    )
    covariance = numpy.zeros((extent_px**2, extent_px**2), dtype=complex)
    patches = iter_probe_patches_on(ProbeEnsemble.from_probe(probe), positions, geometry)

    for patch, weight in zip(patches, weights):
        sensing = numpy.zeros((probe_px**2, extent_px**2), dtype=complex)

        for pixel in range(extent_px**2):
            canvas = numpy.zeros(extent_px**2, dtype=complex)
            canvas[pixel] = 1.0
            window = canvas.reshape(extent_px, extent_px)[
                patch.bounds.y_slice, patch.bounds.x_slice
            ]
            sensing[:, pixel] = propagator.propagate(patch.modes[0] * window).ravel()

        covariance += sensing.conj().T @ (weight.ravel()[:, None] * sensing)

    _, eigenvectors = numpy.linalg.eigh(covariance)
    expected = eigenvectors[:, -1].reshape(extent_px, extent_px)

    estimate = _estimate(
        data, positions, probe, geometry=geometry, lower_bound=lower_bound, tolerance=1e-12
    )

    illumination = accumulate_probe_intensity_on(
        ProbeEnsemble.from_probe(probe), positions, geometry
    )
    support = illumination >= 0.05 * illumination.max()
    a, b = estimate[support], expected[support]
    assert abs(numpy.vdot(a, b)) / numpy.linalg.norm(a) / numpy.linalg.norm(b) > 1 - 1e-8


def test_chunking_and_the_probe_cache_change_nothing() -> None:
    probe, positions, data = _far_field_case()
    reference = _estimate(data, positions, probe, tolerance=1e-10)

    estimate = _estimate(
        data,
        positions,
        probe,
        tolerance=1e-10,
        max_patterns_per_chunk=5,
        max_cached_bytes=0,
    )

    numpy.testing.assert_allclose(estimate, reference, atol=1e-6)


# -- recovery --------------------------------------------------------------------------


def test_far_field_recovery_beats_a_flat_object_and_stxm_dpc() -> None:
    probe, positions, data = _far_field_case()
    stxm_dpc = generate_stxm_dpc_object(
        _geometry(),
        compute_stxm_dpc_signals(data),
        positions,
        detector_extent=data.get_image_extent(),
    ).get_array()[0]
    flat = numpy.ones_like(_truth())

    spectral = _correlation(_estimate(data, positions, probe), _truth())
    corrected = _correlation(_estimate(data, positions, probe, correct_amplitude=True), _truth())

    # Measured 0.906 raw, 0.962 corrected, 0.898 for STXM-DPC and 0.662 flat. Without
    # the reference correction the estimate falls below STXM-DPC near the edge of a scan.
    assert spectral > _correlation(flat, _truth()) + 0.1
    assert corrected > _correlation(stxm_dpc, _truth()) + 0.03


def test_near_field_recovery_with_structured_illumination() -> None:
    probe = _diffuser_probe()
    positions = _grid_positions(14, 4)
    data = _simulate(probe, positions, far_field=False)
    flat = numpy.ones_like(_truth())

    spectral = _correlation(_estimate(data, positions, probe, far_field=False), _truth())

    assert spectral > _correlation(flat, _truth()) + 0.2


@pytest.mark.parametrize('preprocessing', list(SpectralPreprocessing))
def test_every_preprocessing_improves_on_a_flat_object(
    preprocessing: SpectralPreprocessing,
) -> None:
    probe = _diffuser_probe()
    positions = _grid_positions(14, 4)
    data = _simulate(probe, positions, far_field=False)

    estimate = _estimate(data, positions, probe, far_field=False, preprocessing=preprocessing)

    assert _correlation(estimate, _truth()) > _correlation(numpy.ones_like(_truth()), _truth())


def test_a_multimode_probe_is_summed_over_its_modes() -> None:
    single = _defocused_disk_probe()
    mode = single.get_array()[0]
    modes = numpy.stack([mode, 0.5 * numpy.roll(mode, 2, axis=-1)])
    probe = Probe(array=modes, pixel_geometry=_PIXEL_GEOMETRY)
    positions = _grid_positions(14, 4)
    data = _simulate(probe, positions)

    estimate = _estimate(data, positions, probe)

    assert estimate.shape == (_OBJECT_PX, _OBJECT_PX)
    assert numpy.isfinite(estimate).all()
    assert _correlation(estimate, _truth()) > _correlation(numpy.ones_like(_truth()), _truth())


# -- normalization, pairing and detector corrections ----------------------------------


@pytest.mark.parametrize('correct_amplitude', [False, True])
def test_the_support_has_unit_mean_modulus_and_zero_net_phase(correct_amplitude: bool) -> None:
    probe, positions, data = _far_field_case()
    support = _support(probe, positions)

    estimate = _estimate(data, positions, probe, correct_amplitude=correct_amplitude)

    assert numpy.mean(numpy.abs(estimate[support])) == pytest.approx(1.0)
    assert abs(numpy.angle(numpy.sum(estimate[support]))) < 1e-9
    numpy.testing.assert_array_equal(estimate[~support], 1.0 + 0.0j)


def test_patterns_pair_with_positions_by_index_not_order() -> None:
    probe, positions, data = _far_field_case()
    reference = _estimate(data, positions, probe, tolerance=1e-10)
    order = numpy.random.default_rng(3).permutation(len(positions))
    shuffled = _with(
        data,
        indexes=numpy.array(data.get_indexes())[order],
        patterns=numpy.array(data.get_patterns())[order],
    )

    numpy.testing.assert_allclose(
        _estimate(shuffled, positions, probe, tolerance=1e-10), reference, atol=1e-6
    )


def test_a_pattern_without_a_position_is_left_out() -> None:
    probe, positions, data = _far_field_case()
    reference = _estimate(data, positions, probe, tolerance=1e-10)
    stray = numpy.random.default_rng(5).uniform(0.0, 1.0e6, size=(1, _PROBE_PX, _PROBE_PX))
    with_stray = _with(
        data,
        indexes=numpy.append(data.get_indexes(), 10_000),
        patterns=numpy.concatenate([data.get_patterns(), stray]),
    )

    numpy.testing.assert_allclose(
        _estimate(with_stray, positions, probe, tolerance=1e-10), reference, atol=1e-6
    )


def test_bad_pixels_carry_no_weight() -> None:
    probe, positions, data = _far_field_case()
    reference = _estimate(data, positions, probe, tolerance=1e-10)
    patterns = numpy.array(data.get_patterns())
    patterns[:, data.get_bad_pixels()] = 1.0e6

    numpy.testing.assert_allclose(
        _estimate(_with(data, patterns=patterns), positions, probe, tolerance=1e-10),
        reference,
        atol=1e-6,
    )


def test_i0_removes_a_beam_decay() -> None:
    probe, positions, data = _far_field_case()
    reference = _estimate(data, positions, probe, tolerance=1e-10)
    decay = numpy.linspace(1.0, 0.6, len(positions))
    decayed = _with(data, patterns=numpy.array(data.get_patterns()) * decay[:, None, None])
    i0_counts_by_index = {
        int(index): 500.0 * float(d) for index, d in zip(data.get_indexes(), decay)
    }

    estimate = _estimate(
        decayed, positions, probe, tolerance=1e-10, i0_counts_by_index=i0_counts_by_index
    )

    numpy.testing.assert_allclose(estimate, reference, atol=1e-6)


def test_carried_detector_maps_are_applied_as_the_moments_apply_them() -> None:
    """``(pattern - background) / flat_field`` clipped at zero, flat-field zeros made bad."""
    probe, positions, data = _far_field_case()
    rng = numpy.random.default_rng(6)
    background = rng.uniform(0.0, 2.0, size=(_PROBE_PX, _PROBE_PX))
    flat_field = rng.uniform(0.8, 1.2, size=(_PROBE_PX, _PROBE_PX))
    # Both below the floor a flat field is divided by, so both pixels turn bad.
    flat_field[3, 4] = 1.0e-4
    flat_field[25, 6] = 0.0
    corrected = numpy.array(data.get_patterns())
    raw = corrected * flat_field + background
    raw[:, 3, 4] = 1.0e6
    raw[:, 25, 6] = 1.0e6

    carried = _with(data, patterns=raw, background=background, flat_field=flat_field)
    unusable = numpy.array(data.get_bad_pixels())
    unusable[3, 4] = unusable[25, 6] = True
    moments = compute_diffraction_moments(
        carried.get_indexes(),
        carried.get_patterns(),
        bad_pixels=carried.get_bad_pixels(),
        background=background,
        flat_field=flat_field,
    )
    assert moments is not None
    numpy.testing.assert_array_equal(moments.bad_pixels, unusable)

    expected = _estimate(_with(data, bad_pixels=unusable), positions, probe, tolerance=1e-10)

    numpy.testing.assert_allclose(
        _estimate(carried, positions, probe, tolerance=1e-10), expected, atol=1e-6
    )


# -- refusals --------------------------------------------------------------------------


def test_probe_pixels_must_match_object_pixels() -> None:
    probe, positions, data = _far_field_case()
    other = Probe(
        array=probe.get_array(),
        pixel_geometry=PixelGeometry(width_m=2 * _PIXEL_SIZE_M, height_m=2 * _PIXEL_SIZE_M),
    )

    with pytest.raises(ValueError, match='differ from object pixels'):
        _estimate(data, positions, other)


def test_the_probe_must_span_the_patterns() -> None:
    _, positions, data = _far_field_case()

    with pytest.raises(ValueError, match='do not match'):
        _estimate(data, positions, _defocused_disk_probe(extent_px=16))


@pytest.mark.parametrize(
    'kwargs',
    [
        dict(lower_bound=-0.1),
        dict(lower_bound=1.1),
        dict(threshold_fraction=0.0),
        dict(threshold_fraction=1.5),
        dict(tolerance=-1.0),
        dict(max_restarts=0),
        dict(num_lanczos_vectors=2),
        dict(support_fraction=0.0),
        dict(max_patterns_per_chunk=0),
        dict(max_cached_bytes=-1),
    ],
)
def test_out_of_range_tunables_are_refused(kwargs: dict[str, Any]) -> None:
    probe, positions, data = _far_field_case()

    with pytest.raises(ValueError):
        _estimate(data, positions, probe, **kwargs)


def test_far_field_needs_a_distance() -> None:
    probe, positions, data = _far_field_case()

    with pytest.raises(ValueError, match='nonzero distance'):
        generate_spectral_object(
            _geometry(),
            data,
            positions,
            probe=probe,
            photon_wavelength_m=_WAVELENGTH_M,
            propagation_distance_m=0.0,
            far_field=True,
        )


# -- the generator ---------------------------------------------------------------------


class _Provider(ObjectGeometryProvider):
    def __init__(
        self,
        data: AssembledDiffractionData | None,
        positions: Sequence[ProbePosition],
        *,
        with_moments: bool = True,
    ) -> None:
        self._data = data
        self._positions = positions
        self._moments = (
            compute_diffraction_moments(
                data.get_indexes(), data.get_patterns(), bad_pixels=data.get_bad_pixels()
            )
            if data is not None and with_moments
            else None
        )

    @property
    def photon_wavelength_m(self) -> float:
        return _WAVELENGTH_M

    @property
    def mean_amplitude_transmission(self) -> float:
        return 1.0

    @property
    def exit_wave_propagation_distance_m(self) -> float:
        return _FAR_FIELD_DISTANCE_M

    @property
    def far_field(self) -> bool:
        return True

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return self._positions

    def get_object_geometry(self) -> ObjectGeometry:
        return _geometry()

    def get_assembled_data(self) -> AssembledDiffractionData | None:
        return self._data

    def get_diffraction_moments(self) -> DiffractionMoments | None:
        return self._moments


def test_the_generator_matches_the_function() -> None:
    probe, positions, data = _far_field_case()
    options = SpectralOptions(eigensolver_tolerance=1e-8)

    object_ = SpectralObjectGenerator().generate(
        options, _Provider(data, positions), rng=numpy.random.default_rng(0), probe=probe
    )

    numpy.testing.assert_allclose(
        object_.get_array()[0],
        _estimate(data, positions, probe, tolerance=1e-8),
        atol=1e-6,
    )


def test_the_generator_waits_for_the_moments() -> None:
    probe, positions, data = _far_field_case()

    with pytest.raises(GeometryNotDefinedError):
        SpectralObjectGenerator().generate(
            SpectralOptions(),
            _Provider(data, positions, with_moments=False),
            rng=numpy.random.default_rng(0),
            probe=probe,
        )


def test_the_generator_needs_a_probe() -> None:
    _, positions, data = _far_field_case()

    with pytest.raises(ValueError, match="needs the product's probe"):
        SpectralObjectGenerator().generate(
            SpectralOptions(),
            _Provider(data, positions),
            rng=numpy.random.default_rng(0),
            probe=None,
        )
