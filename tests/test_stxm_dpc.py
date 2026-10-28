"""STXM differential phase contrast: signals, phase recovery, diagnostics and the builder.

Phase recovery is checked on data simulated by the far-field forward model, so the sign
and orientation conventions are pinned against that model rather than derived.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cache
from typing import Any
from unittest.mock import MagicMock
import logging

import numpy
import numpy.testing
import pytest
from scipy.ndimage import gaussian_filter

from ptychodus.api.accumulate import DiffractionMomentAccumulator
from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.accumulate import DiffractionMoments, compute_diffraction_moments
from ptychodus.api.geometry import GeometryNotDefinedError, ImageExtent, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.probe import Probe, ProbeEnsemble
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.plugins import PluginChooser, PluginRegistry
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.settings import SettingsRegistry
from ptychodus.api.simulate.diffraction import generate_diffraction_data
from ptychodus.api.simulate.object import (
    ObjectGenerator,
    STXMDPCSignals,
    compute_stxm_dpc_signals,
    generate_stxm_dpc_object,
    interpolate_scan_values,
)
from ptychodus.api.simulate.probe import propagate_probe
from ptychodus.model.product.object import (
    ObjectBuilder,
    ObjectBuilderFactory,
    ObjectRepositoryItem,
    ObjectSettings,
)
from ptychodus.plugins.object.phase_retrieval import compute_decay_corrected_signals

_LOGGER_NAME = 'ptychodus.api.simulate.object'
_PIXEL_SIZE_M = 50e-9
_OBJECT_PX = 128
_DETECTOR_DISTANCE_M = 2.0
_PHOTON_ENERGY_EV = 10_000.0
_PIXEL_GEOMETRY = PixelGeometry(width_m=_PIXEL_SIZE_M, height_m=_PIXEL_SIZE_M)
_GEOMETRY = ObjectGeometry(
    width_px=_OBJECT_PX,
    height_px=_OBJECT_PX,
    pixel_width_m=_PIXEL_SIZE_M,
    pixel_height_m=_PIXEL_SIZE_M,
    center_x_m=0.0,
    center_y_m=0.0,
)
# Well inside every scan below, so neither the hull nor the mirror padding reaches it.
_INTERIOR = (slice(34, 94), slice(34, 94))

# Measured on the fixtures below with whole-frame centroids, with headroom: 0.016 rad for
# the flat-phase scan, and 0.061 rad uncorrected versus 0.012 rad corrected (with the
# probe's illumination as the support) for the defocused one.
_PHASE_TOLERANCE_RAD = 0.03
_SHADOW_UNCORRECTED_MINIMUM_RAD = 0.04
_SHADOW_CORRECTED_TOLERANCE_RAD = 0.02


def _metadata() -> ProductMetadata:
    return ProductMetadata(
        name='stxm_dpc',
        comments='',
        detector_distance_m=_DETECTOR_DISTANCE_M,
        photon_energy_eV=_PHOTON_ENERGY_EV,
        probe_photon_count=1.0,
        exposure_time_s=1.0,
        mass_attenuation_m2_per_kg=0.0,
        tomography_angle_deg=0.0,
    )


def _gaussian_probe(size_px: int, sigma_px: float) -> Probe:
    """A flat-phase Gaussian of amplitude width `sigma_px` -- a probe at focus."""
    yy, xx = numpy.indices((size_px, size_px)) - (size_px - 1) / 2
    array = 100.0 * numpy.exp(-(xx**2 + yy**2) / (4 * sigma_px**2))
    return Probe(array=array.astype(complex), pixel_geometry=_PIXEL_GEOMETRY)


def _bump() -> numpy.ndarray:
    """An off-centre bump with unequal widths, so a sign flip, an axis swap or a quarter
    turn each give a clearly different gradient field."""
    yy, xx = numpy.indices((_OBJECT_PX, _OBJECT_PX)) - (_OBJECT_PX - 1) / 2
    return numpy.exp(-((xx - 10) ** 2 / (2 * 14**2) + (yy + 6) ** 2 / (2 * 9**2)))


def _grid_positions(half_span_px: int, step_px: int, seed: int = 1) -> list[ProbePosition]:
    rng = numpy.random.default_rng(seed)
    positions: list[ProbePosition] = list()

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


def _simulate(
    transmission: numpy.ndarray,
    probe: Probe,
    positions: Sequence[ProbePosition],
    **kwargs: Any,
) -> AssembledDiffractionData:
    product = Product(
        metadata=_metadata(),
        probe_positions=ProbePositionSequence(list(positions)),
        probes=ProbeEnsemble(
            array=probe.get_array()[numpy.newaxis],
            opr_weights=None,
            pixel_geometry=_PIXEL_GEOMETRY,
        ),
        object_=Object(
            array=transmission[numpy.newaxis],
            pixel_geometry=_PIXEL_GEOMETRY,
            center=ObjectCenter(x_m=0.0, y_m=0.0),
        ),
        losses=[],
    )
    return generate_diffraction_data(product, **kwargs)


def _with(data: AssembledDiffractionData, **changes: Any) -> AssembledDiffractionData:
    """A copy of `data` with some constructor arguments replaced."""
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


def _recover(
    data: AssembledDiffractionData,
    positions: Sequence[ProbePosition],
    signals: STXMDPCSignals | None = None,
    **kwargs: Any,
) -> Object:
    return generate_stxm_dpc_object(
        _GEOMETRY,
        compute_stxm_dpc_signals(data) if signals is None else signals,
        positions,
        detector_extent=data.get_image_extent(),
        **kwargs,
    )


def _phase(object_: Object) -> numpy.ndarray:
    return numpy.angle(object_.get_array()[0])


def _phase_error_rad(phase: numpy.ndarray, reference: numpy.ndarray) -> float:
    """RMS difference over the interior, up to a constant."""
    difference = phase[_INTERIOR] - reference[_INTERIOR]
    return float(numpy.sqrt(numpy.mean(numpy.square(difference - difference.mean()))))


def _warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _LOGGER_NAME and record.levelno >= logging.WARNING
    ]


@dataclass(frozen=True)
class _Scan:
    data: AssembledDiffractionData
    positions: list[ProbePosition]
    phase: numpy.ndarray
    """The true phase blurred by the probe intensity, which is what DPC resolves."""
    probe: Probe


@pytest.fixture(scope='module')
def phase_scan() -> _Scan:
    """A pure phase bump of about 1 rad under a flat-phase probe at focus."""
    probe_sigma_px = 3.0
    phase = 1.0 * _bump()
    positions = _grid_positions(40, 3)
    probe = _gaussian_probe(32, probe_sigma_px)
    data = _simulate(numpy.exp(1j * phase), probe, positions)
    return _Scan(data, positions, gaussian_filter(phase, probe_sigma_px / numpy.sqrt(2)), probe)


@pytest.fixture(scope='module')
def absorbing_scan() -> _Scan:
    """A single material, beta/delta = 0.1, under a flat-phase probe at focus."""
    probe_sigma_px = 3.0
    thickness = _bump()
    phase = -1.0 * thickness
    positions = _grid_positions(40, 3)
    probe = _gaussian_probe(32, probe_sigma_px)
    transmission = numpy.exp(-0.1 * thickness) * numpy.exp(1j * phase)
    data = _simulate(transmission, probe, positions)
    return _Scan(data, positions, gaussian_filter(phase, probe_sigma_px / numpy.sqrt(2)), probe)


@pytest.fixture(scope='module')
def shadow_scan() -> _Scan:
    """The absorbing sample under a probe defocused by 2 mm, whose phase curvature puts
    a shadow term of roughly 40% into every centroid."""
    focus = _gaussian_probe(48, 1.5)
    probe = propagate_probe(
        focus,
        photon_wavelength_m=_metadata().photon_wavelength_m,
        propagation_distance_m=2.0e-3,
    )
    intensity = numpy.square(numpy.abs(probe.get_array()[0]))
    xx = numpy.indices(intensity.shape)[1] - (intensity.shape[1] - 1) / 2
    sigma_px = float(numpy.sqrt(numpy.sum(intensity * xx**2) / numpy.sum(intensity)))
    thickness = _bump()
    phase = -1.0 * thickness
    positions = _grid_positions(36, 3)
    transmission = numpy.exp(-0.1 * thickness) * numpy.exp(1j * phase)
    data = _simulate(transmission, probe, positions)
    return _Scan(data, positions, gaussian_filter(phase, sigma_px), probe)


# ---------------------------------------------------------------------------
# Phase recovery
# ---------------------------------------------------------------------------


def test_recovers_the_phase_in_the_forward_model_frame(phase_scan: _Scan) -> None:
    recovered = _recover(phase_scan.data, phase_scan.positions)

    assert _phase_error_rad(_phase(recovered), phase_scan.phase) < _PHASE_TOLERANCE_RAD


def test_recovers_the_phase_of_an_absorbing_sample(absorbing_scan: _Scan) -> None:
    recovered = _recover(absorbing_scan.data, absorbing_scan.positions)

    assert _phase_error_rad(_phase(recovered), absorbing_scan.phase) < _PHASE_TOLERANCE_RAD


def test_empty_beam_is_the_phase_reference(absorbing_scan: _Scan) -> None:
    phase = _phase(_recover(absorbing_scan.data, absorbing_scan.positions))
    # Far from the bump the sample is empty beam, which is also the most transparent.
    assert numpy.abs(phase[30:40, 30:40]).max() < 0.03


# ---------------------------------------------------------------------------
# Amplitude and totals
# ---------------------------------------------------------------------------


def test_amplitude_is_the_normalized_stxm_map(phase_scan: _Scan) -> None:
    signals = compute_stxm_dpc_signals(phase_scan.data)
    totals = dict(zip(signals.indexes.tolist(), signals.total_counts.tolist()))
    grids, scanned = interpolate_scan_values(
        _GEOMETRY,
        ProbePositionSequence(phase_scan.positions),
        numpy.array([totals[position.index] for position in phase_scan.positions]),
    )
    intensity = grids[0]
    expected = numpy.sqrt(intensity / numpy.mean(intensity[scanned]))

    recovered = _recover(phase_scan.data, phase_scan.positions)

    numpy.testing.assert_allclose(numpy.abs(recovered.get_array()[0]), expected, rtol=1e-9)


def test_i0_removes_a_beam_decay(absorbing_scan: _Scan) -> None:
    data = absorbing_scan.data
    decay = numpy.linspace(1.0, 0.6, data.get_patterns().shape[0])
    decayed = _with(data, patterns=data.get_patterns() * decay[:, numpy.newaxis, numpy.newaxis])
    i0_counts_by_index = {
        int(index): 1.0e6 * float(scale) for index, scale in zip(data.get_indexes(), decay)
    }
    signals = compute_stxm_dpc_signals(decayed, i0_counts_by_index=i0_counts_by_index)

    expected = numpy.abs(_recover(data, absorbing_scan.positions).get_array())
    actual = numpy.abs(_recover(decayed, absorbing_scan.positions, signals).get_array())

    numpy.testing.assert_allclose(actual, expected, rtol=1e-9)


def test_an_opaque_position_stays_dark_with_a_finite_phase(phase_scan: _Scan) -> None:
    patterns = numpy.array(phase_scan.data.get_patterns())
    dark = len(phase_scan.positions) // 2
    patterns[dark] = 0.0
    data = _with(phase_scan.data, patterns=patterns)

    signals = compute_stxm_dpc_signals(data)
    recovered = _recover(data, phase_scan.positions, signals)

    assert numpy.isnan(signals.center_of_mass_x_px[dark])
    assert signals.total_counts[dark] == 0.0
    point = _GEOMETRY.map_coordinates_probe_to_object(phase_scan.positions[dark])
    row, column = round(point.y_px), round(point.x_px)
    amplitude = numpy.abs(recovered.get_array()[0])
    # The nearest pixel can sit most of a pixel from a jittered position, so it is dark
    # relative to its surroundings rather than exactly zero.
    assert amplitude[row, column] < 0.6 * numpy.median(
        amplitude[row - 6 : row + 7, column - 6 : column + 7]
    )
    assert numpy.isfinite(_phase(recovered)).all()


# ---------------------------------------------------------------------------
# Detector handling
# ---------------------------------------------------------------------------


def test_inpainted_bad_pixels_preserve_the_phase(phase_scan: _Scan) -> None:
    data = phase_scan.data
    mean_pattern = data.get_mean_pattern()
    yy, xx = numpy.indices(mean_pattern.shape)
    total = numpy.sum(mean_pattern)
    center_y = numpy.sum(mean_pattern * yy) / total
    center_x = numpy.sum(mean_pattern * xx) / total
    radius = numpy.hypot(yy - center_y, xx - center_x)
    # The focused probe's beam is about a pixel wide here, so the holes go on its flank,
    # inside the bright field: a dozen scattered dead pixels and a small cluster.
    candidates = numpy.argwhere((radius >= 3.0) & (radius < 7.0))
    chosen = candidates[numpy.random.default_rng(3).choice(len(candidates), 12, replace=False)]
    bad_pixels = numpy.zeros(mean_pattern.shape, dtype=bool)
    bad_pixels[chosen[:, 0], chosen[:, 1]] = True
    row, column = round(center_y), round(center_x)
    bad_pixels[row - 1 : row + 2, column + 3 : column + 5] = True
    masked = _with(data, bad_pixels=bad_pixels)

    reference = _phase(_recover(data, phase_scan.positions))

    phase = _phase(_recover(masked, phase_scan.positions))

    assert _phase_error_rad(phase, reference) < 0.005
    attenuation = numpy.std(phase[_INTERIOR]) / numpy.std(reference[_INTERIOR])
    assert attenuation == pytest.approx(1.0, abs=0.02)


def test_a_carried_background_does_not_move_the_centroids(phase_scan: _Scan) -> None:
    data = phase_scan.data
    background = numpy.full(data.get_bad_pixels().shape, 5.0)
    with_pedestal = _with(data, patterns=data.get_patterns() + background, background=background)

    expected = compute_stxm_dpc_signals(data)
    actual = compute_stxm_dpc_signals(with_pedestal)

    numpy.testing.assert_allclose(
        actual.center_of_mass_x_px, expected.center_of_mass_x_px, atol=1e-9
    )
    numpy.testing.assert_allclose(
        actual.center_of_mass_y_px, expected.center_of_mass_y_px, atol=1e-9
    )
    numpy.testing.assert_allclose(actual.total_counts, expected.total_counts, rtol=1e-9)


def test_chunk_size_does_not_change_the_signals(phase_scan: _Scan) -> None:
    expected = compute_stxm_dpc_signals(phase_scan.data)
    actual = compute_stxm_dpc_signals(phase_scan.data, max_patterns_per_chunk=1)

    # Relaxed from byte equality: numpy's per-row sums over a stack are not bit-stable
    # in the number of rows, and differ in the last place.
    numpy.testing.assert_allclose(actual.total_counts, expected.total_counts, rtol=1e-12)
    numpy.testing.assert_allclose(
        actual.center_of_mass_x_px, expected.center_of_mass_x_px, rtol=1e-12, atol=1e-12
    )
    numpy.testing.assert_allclose(
        actual.center_of_mass_y_px, expected.center_of_mass_y_px, rtol=1e-12, atol=1e-12
    )


def test_signals_are_one_pass_of_the_accumulator(phase_scan: _Scan) -> None:
    data = phase_scan.data
    indexes = numpy.asarray(data.get_indexes())
    accumulator = DiffractionMomentAccumulator(
        *data.get_bad_pixels().shape, bad_pixels=data.get_bad_pixels(), chunk_size=7
    )

    for start in range(0, indexes.size, 5):
        accumulator.add(indexes[start : start + 5], data.get_patterns()[start : start + 5])

    moments = accumulator.get_moments()
    assert moments is not None
    expected = compute_stxm_dpc_signals(data)

    numpy.testing.assert_array_equal(moments.indexes, expected.indexes)
    numpy.testing.assert_allclose(moments.total_counts, expected.total_counts, rtol=1e-12)
    numpy.testing.assert_allclose(
        moments.center_of_mass_x_px, expected.center_of_mass_x_px, rtol=1e-12, atol=1e-12
    )


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_pairing_is_by_index_not_order(phase_scan: _Scan) -> None:
    signals = compute_stxm_dpc_signals(phase_scan.data)
    positions = phase_scan.positions
    expected = _recover(phase_scan.data, positions, signals).get_array()

    shuffled = list(positions)
    numpy.random.default_rng(5).shuffle(shuffled)
    numpy.testing.assert_allclose(
        _recover(phase_scan.data, shuffled, signals).get_array(), expected, atol=1e-12
    )

    beyond = max(position.index for position in positions) + 1
    unmatched_position = [*positions, ProbePosition(index=beyond, x_m=0.0, y_m=0.0)]
    numpy.testing.assert_allclose(
        _recover(phase_scan.data, unmatched_position, signals).get_array(), expected, atol=1e-12
    )

    out_of_range = replace(
        signals,
        indexes=numpy.append(signals.indexes, beyond + 10),
        total_counts=numpy.append(signals.total_counts, 1.0),
        center_of_mass_x_px=numpy.append(signals.center_of_mass_x_px, 5.0),
        center_of_mass_y_px=numpy.append(signals.center_of_mass_y_px, 5.0),
    )
    numpy.testing.assert_allclose(
        _recover(phase_scan.data, positions, out_of_range).get_array(), expected, atol=1e-12
    )


def test_duplicate_positions_are_averaged(phase_scan: _Scan) -> None:
    signals = compute_stxm_dpc_signals(phase_scan.data)
    positions = phase_scan.positions
    expected = _recover(phase_scan.data, positions, signals).get_array()

    target = positions[len(positions) // 2]
    offset_m = 0.4 * _PIXEL_SIZE_M
    duplicated = [position for position in positions if position.index != target.index]
    duplicated.append(replace(target, x_m=target.x_m - offset_m))
    duplicated.append(replace(target, x_m=target.x_m + offset_m))

    numpy.testing.assert_allclose(
        _recover(phase_scan.data, duplicated, signals).get_array(), expected, atol=1e-9
    )


def test_a_signal_between_positions_takes_an_interpolated_one(phase_scan: _Scan) -> None:
    signals = compute_stxm_dpc_signals(phase_scan.data)
    positions = phase_scan.positions
    missing = len(positions) // 2
    before, after = positions[missing - 1], positions[missing + 1]
    midpoint = ProbePosition(
        index=missing, x_m=(before.x_m + after.x_m) / 2, y_m=(before.y_m + after.y_m) / 2
    )
    with_midpoint = [*positions[:missing], midpoint, *positions[missing + 1 :]]
    without = [*positions[:missing], *positions[missing + 1 :]]

    numpy.testing.assert_allclose(
        _recover(phase_scan.data, without, signals).get_array(),
        _recover(phase_scan.data, with_midpoint, signals).get_array(),
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Shadow correction and regularization
# ---------------------------------------------------------------------------


def test_shadow_correction_removes_the_probe_phase_term(shadow_scan: _Scan) -> None:
    uncorrected = _phase_error_rad(
        _phase(_recover(shadow_scan.data, shadow_scan.positions)), shadow_scan.phase
    )
    corrected = _phase_error_rad(
        _phase(_recover(shadow_scan.data, shadow_scan.positions, probe=shadow_scan.probe)),
        shadow_scan.phase,
    )

    assert uncorrected > _SHADOW_UNCORRECTED_MINIMUM_RAD
    assert corrected < 0.5 * uncorrected
    assert corrected < _SHADOW_CORRECTED_TOLERANCE_RAD


def test_shadow_correction_rejects_a_probe_on_other_pixels(shadow_scan: _Scan) -> None:
    probe = Probe(
        array=shadow_scan.probe.get_array(),
        pixel_geometry=PixelGeometry(width_m=2 * _PIXEL_SIZE_M, height_m=2 * _PIXEL_SIZE_M),
    )

    with pytest.raises(ValueError, match='Probe pixels'):
        _recover(shadow_scan.data, shadow_scan.positions, probe=probe)


def test_extent_must_be_positive(phase_scan: _Scan) -> None:
    with pytest.raises(ValueError, match='Detector extent'):
        generate_stxm_dpc_object(
            _GEOMETRY,
            compute_stxm_dpc_signals(phase_scan.data),
            phase_scan.positions,
            detector_extent=ImageExtent(width_px=0, height_px=32),
        )


def test_regularization(phase_scan: _Scan) -> None:
    signals = compute_stxm_dpc_signals(phase_scan.data)
    default = _recover(phase_scan.data, phase_scan.positions, signals)

    unregularized = _recover(
        phase_scan.data, phase_scan.positions, signals, regularization_fov_fraction=0.0
    )
    numpy.testing.assert_array_equal(unregularized.get_array(), default.get_array())

    # Damps wavelengths longer than about 20 pixels of the 128-pixel canvas, the scale
    # of the bump.
    regularized = _recover(
        phase_scan.data, phase_scan.positions, signals, regularization_fov_fraction=20 / 128
    )
    assert numpy.std(_phase(regularized)[_INTERIOR]) < 0.8 * numpy.std(_phase(default)[_INTERIOR])

    with pytest.raises(ValueError, match='Regularization'):
        _recover(phase_scan.data, phase_scan.positions, signals, regularization_fov_fraction=-1.0)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class _StubProvider(ObjectGeometryProvider):
    def __init__(
        self,
        positions: Sequence[ProbePosition],
        *,
        far_field: bool = True,
        assembled_data: AssembledDiffractionData | None = None,
    ) -> None:
        self._positions = positions
        self._far_field = far_field
        self._assembled_data = assembled_data

    def get_assembled_data(self) -> AssembledDiffractionData | None:
        return self._assembled_data

    def get_diffraction_moments(self) -> DiffractionMoments | None:
        data = self._assembled_data

        if data is None:
            return None

        return compute_diffraction_moments(
            data.get_indexes(),
            data.get_patterns(),
            bad_pixels=data.get_bad_pixels(),
            background=data.get_background(),
            flat_field=data.get_flat_field(),
        )

    @property
    def photon_wavelength_m(self) -> float:
        return _metadata().photon_wavelength_m

    @property
    def mean_amplitude_transmission(self) -> float:
        return 1.0

    @property
    def object_plane_propagation_distance_m(self) -> float:
        return _DETECTOR_DISTANCE_M

    @property
    def far_field(self) -> bool:
        return self._far_field

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return self._positions

    def get_object_geometry(self) -> ObjectGeometry:
        return _GEOMETRY


def _settings() -> ObjectSettings:
    return ObjectSettings(SettingsRegistry())


@cache
def _object_generators() -> PluginChooser[ObjectGenerator]:
    return PluginRegistry.load_plugins().object_generators


def _factory(settings: ObjectSettings) -> ObjectBuilderFactory:
    return ObjectBuilderFactory(
        numpy.random.default_rng(0), settings, _object_generators(), MagicMock(), MagicMock()
    )


def _builder(name: str, settings: ObjectSettings) -> ObjectBuilder:
    return _factory(settings).create(name, has_diffraction_data=True)


def _ensemble(probe: Probe) -> ProbeEnsemble:
    return ProbeEnsemble(
        array=probe.get_array()[numpy.newaxis],
        opr_weights=None,
        pixel_geometry=probe.get_pixel_geometry(),
    )


def test_factory_offers_stxm_dpc_in_place_of_stxm() -> None:
    names = list(_factory(_settings()))

    assert 'stxm_dpc' in names
    assert 'stxm' not in names


def test_builder_builds_a_single_layer_object(phase_scan: _Scan) -> None:
    builder = _builder('stxm_dpc', _settings())

    object_ = builder.build(_StubProvider(phase_scan.positions, assembled_data=phase_scan.data), [])

    assert object_.num_layers == 1
    assert numpy.isfinite(object_.get_array()).all()


def test_builder_passes_its_regularization_through(phase_scan: _Scan) -> None:
    provider = _StubProvider(phase_scan.positions, assembled_data=phase_scan.data)
    builder = _builder('stxm_dpc', _settings())
    default = builder.build(provider, []).get_array()

    builder.parameters()['regularization_fov_fraction'].set_value(20 / 128)

    assert not numpy.allclose(builder.build(provider, []).get_array(), default)


def test_builder_rejects_near_field_before_the_signal_pass(phase_scan: _Scan) -> None:
    provider = _StubProvider(phase_scan.positions, far_field=False, assembled_data=phase_scan.data)
    provider.get_assembled_data = MagicMock(return_value=phase_scan.data)  # type: ignore[method-assign]
    builder = _builder('stxm_dpc', _settings())

    with pytest.raises(ValueError, match='far-field'):
        builder.build(provider, [])

    provider.get_assembled_data.assert_not_called()


def test_builder_waits_for_data_rather_than_failing(phase_scan: _Scan) -> None:
    with pytest.raises(GeometryNotDefinedError):
        _builder('stxm_dpc', _settings()).build(_StubProvider(phase_scan.positions), [])


def test_a_copied_builder_reads_the_data_of_the_product_it_is_in(phase_scan: _Scan) -> None:
    builder = _builder('stxm_dpc', _settings())
    provider = _StubProvider(phase_scan.positions, assembled_data=phase_scan.data)

    numpy.testing.assert_array_equal(
        builder.copy().build(provider, []).get_array(), builder.build(provider, []).get_array()
    )
    assert builder.reads_diffraction_data


def test_builder_shadow_correction_uses_the_bound_probe(
    shadow_scan: _Scan, caplog: pytest.LogCaptureFixture
) -> None:
    provider = _StubProvider(shadow_scan.positions, assembled_data=shadow_scan.data)
    builder = _builder('stxm_dpc', _settings())
    uncorrected = builder.build(provider, []).get_array()
    builder.parameters()['shadow_correction'].set_value(True)

    with caplog.at_level(logging.WARNING):
        unbound = builder.build(provider, []).get_array()

    assert any('No probe' in record.getMessage() for record in caplog.records)
    numpy.testing.assert_array_equal(unbound, uncorrected)

    caplog.clear()
    mismatched = Probe(
        array=shadow_scan.probe.get_array(),
        pixel_geometry=PixelGeometry(width_m=2 * _PIXEL_SIZE_M, height_m=2 * _PIXEL_SIZE_M),
    )
    builder.bind_probe_supplier(lambda: _ensemble(mismatched))

    with caplog.at_level(logging.WARNING):
        skipped = builder.build(provider, []).get_array()

    assert any('differ from object pixels' in record.getMessage() for record in caplog.records)
    numpy.testing.assert_array_equal(skipped, uncorrected)

    builder.bind_probe_supplier(lambda: _ensemble(shadow_scan.probe))
    corrected = builder.build(provider, []).get_array()

    assert not numpy.allclose(corrected, uncorrected)


def test_item_binds_its_probe_supplier_to_every_builder(phase_scan: _Scan) -> None:
    settings = _settings()

    def supplier() -> ProbeEnsemble:
        return _ensemble(phase_scan.probe)

    first = _builder('stxm_dpc', settings)
    item = ObjectRepositoryItem(
        _StubProvider(phase_scan.positions, assembled_data=phase_scan.data),
        settings,
        first,
        probe_supplier=supplier,
    )
    assert first._probe_supplier is supplier

    copied = first.copy()
    assert copied._probe_supplier is None

    item.set_builder(copied)
    assert copied._probe_supplier is supplier


# ---------------------------------------------------------------------------
# Mapping, support, scale and I0
# ---------------------------------------------------------------------------


def test_a_probe_sets_the_support_but_not_the_values(phase_scan: _Scan) -> None:
    positions = ProbePositionSequence(phase_scan.positions)
    values = numpy.arange(len(positions), dtype=float)
    probes = _ensemble(phase_scan.probe)

    linear, hull = interpolate_scan_values(_GEOMETRY, positions, values)
    same, illuminated = interpolate_scan_values(_GEOMETRY, positions, values, probes=probes)

    numpy.testing.assert_array_equal(same, linear)
    # The footprint reaches past the outermost positions, and not to the far corners.
    assert numpy.count_nonzero(illuminated & ~hull) > 0
    assert not illuminated[0, 0] and not illuminated[-1, -1]

    with pytest.raises(ValueError, match='Support fraction'):
        interpolate_scan_values(_GEOMETRY, positions, values, probes=probes, support_fraction=0.0)


def test_the_incident_photon_count_gives_an_absolute_transmission(absorbing_scan: _Scan) -> None:
    signals = compute_stxm_dpc_signals(absorbing_scan.data)
    incident = float(numpy.max(signals.total_counts))
    relative = _recover(absorbing_scan.data, absorbing_scan.positions, signals=signals)
    absolute = _recover(
        absorbing_scan.data,
        absorbing_scan.positions,
        signals=signals,
        incident_photon_count=incident,
    )

    # Only the scale differs, and the brightest total is the incident beam.
    ratio = numpy.abs(absolute.get_array()) / numpy.abs(relative.get_array())
    assert numpy.allclose(ratio, ratio.flat[0])
    assert numpy.max(numpy.abs(absolute.get_array())) <= 1.0 + 1e-6

    with pytest.raises(ValueError, match='incident photon count must be positive'):
        _recover(absorbing_scan.data, absorbing_scan.positions, incident_photon_count=0.0)


def _with_position_i0(scan: _Scan, decay: numpy.ndarray) -> list[ProbePosition]:
    return [
        ProbePosition(
            index=position.index, x_m=position.x_m, y_m=position.y_m, probe_photon_count=float(m)
        )
        for position, m in zip(scan.positions, decay)
    ]


def test_the_positions_i0_corrects_the_builders_signals(absorbing_scan: _Scan) -> None:
    data = absorbing_scan.data
    decay = numpy.linspace(1.0, 0.6, data.get_patterns().shape[0])
    decayed = _with(data, patterns=data.get_patterns() * decay[:, numpy.newaxis, numpy.newaxis])
    reference = compute_stxm_dpc_signals(data)

    uncorrected = compute_decay_corrected_signals(
        _StubProvider(absorbing_scan.positions, assembled_data=decayed)
    )
    corrected = compute_decay_corrected_signals(
        _StubProvider(_with_position_i0(absorbing_scan, 1.0e6 * decay), assembled_data=decayed)
    )

    def spread(signals: STXMDPCSignals) -> float:
        ratio = signals.total_counts / reference.total_counts
        return float(numpy.std(ratio / numpy.mean(ratio)))

    assert spread(uncorrected) > 0.05
    assert spread(corrected) < 1e-3


def test_the_builders_wait_for_the_moments(absorbing_scan: _Scan) -> None:
    with pytest.raises(GeometryNotDefinedError):
        compute_decay_corrected_signals(_StubProvider(absorbing_scan.positions))


def test_paganin_builds_on_the_decay_corrected_totals(absorbing_scan: _Scan) -> None:
    data = absorbing_scan.data
    decay = numpy.linspace(1.0, 0.6, data.get_patterns().shape[0])
    decayed = _with(data, patterns=data.get_patterns() * decay[:, numpy.newaxis, numpy.newaxis])
    builder = _builder('paganin', _settings())

    clean = builder.build(
        _StubProvider(absorbing_scan.positions, assembled_data=data), []
    ).get_array()
    corrected = builder.build(
        _StubProvider(_with_position_i0(absorbing_scan, 1.0e6 * decay), assembled_data=decayed),
        [],
    ).get_array()

    numpy.testing.assert_allclose(numpy.abs(corrected), numpy.abs(clean), rtol=1e-3)


def test_a_bound_probe_sets_the_builders_support(phase_scan: _Scan) -> None:
    provider = _StubProvider(phase_scan.positions, assembled_data=phase_scan.data)
    builder = _builder('stxm_dpc', _settings())
    unbound = builder.build(provider, []).get_array()

    builder.bind_probe_supplier(lambda: _ensemble(phase_scan.probe))
    bound = builder.build(provider, []).get_array()

    # Shadow correction is off, so only the support differs: amplitudes agree and the
    # phase, integrated and referenced over another region, does not.
    numpy.testing.assert_allclose(numpy.abs(bound), numpy.abs(unbound), rtol=1e-6)
    assert not numpy.allclose(numpy.angle(bound), numpy.angle(unbound))
