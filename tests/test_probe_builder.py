"""Regression tests for the probe conditioning pipeline (incoherent modes -> OPR modes).

Two invariants carry the weight here.

First, conditioning is expand-only and therefore idempotent. The generators in
ptychodus.api.simulate.probe are not safe to re-apply: generate_incoherent_probe_modes
re-orthogonalizes and renormalizes every mode to the decay profile, and
generate_coherent_probe_modes fills its output with fresh Gaussian noise, keeps
only coherent mode zero of its input, and regenerates the OPR weights from
scratch. The guards in ProbeEnsembleBuilder._condition_probe are what stop a
converged OPR basis being replaced with noise.

Second, FromMemoryProbeBuilder must never condition. It holds a probe that is
already conditioned -- reconstruction output, which ProcessingTaskMonitor
re-assigns to the output product item on every reconstructor iteration, and
products loaded from HDF5/NPZ. Re-running the mode generators there would destroy
the reconstruction, once per iteration.

The photon-count rescale sits on the other side of the split: it is
generation-only, because a probe read from file already carries the intensity it
was reconstructed at.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock
import logging
import math

import numpy
import pytest

from ptychodus.api.assemble import AssembledDiffractionData
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import (
    OPRWeightPolicy,
    Probe,
    ProbeFileReader,
    ProbeGeometry,
    ProbeGeometryProvider,
    ProbeEnsemble,
)
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.probe.builder import (
    FromFileProbeBuilder,
    FromMemoryProbeBuilder,
    ProbeEnsembleBuilder,
    resolve_defocus_distance_m,
)
from ptychodus.model.product.probe import INCOHERENT_MODE_STRATEGY_NAMES
from ptychodus.api.simulate.probe import (
    generate_average_pattern_probe,
    generate_disk_probe,
    generate_rectangular_probe,
    generate_super_gaussian_probe,
    propagate_probe,
    rescale_probe_intensity,
)
from ptychodus.model.diffraction.dataset import AssembledDiffractionDataset
from ptychodus.model.diffraction.monitor import DiffractionTaskMonitor
from ptychodus.model.diffraction.settings import DetectorSettings, DiffractionSettings
from ptychodus.model.product.probe.average_pattern import AveragePatternProbeBuilder
from ptychodus.model.product.probe.builder_factory import ProbeBuilderFactory
from ptychodus.model.product.probe.disk import DiskProbeBuilder
from ptychodus.model.product.probe.kb_mirror import KBMirrorProbeBuilder
from ptychodus.model.product.probe.rect import RectangularProbeBuilder
from ptychodus.model.product.probe.settings import ProbeSettings
from ptychodus.model.product.probe.super_gaussian import SuperGaussianProbeBuilder

from test_diffraction_loader import InlineTaskManager

NUM_SCAN_POINTS = 7
PROBE_PHOTON_COUNT = 1.0e6
# Fine enough that the default 1 um disk covers several pixels; at a coarser
# pixel size the generated probe is empty and rescale_probe_intensity bails.
PIXEL_SIZE_M = 1.0e-7
PROBE_EXTENT_PX = 64
# The default KB optic has NA 3e-3, whose focus a 100 nm grid cannot hold: the pupil
# window is lambda*z/dx_probe, so the aperture only fits on a much finer grid.
KB_PIXEL_SIZE_M = 5.0e-9


def _make_settings() -> ProbeSettings:
    return ProbeSettings(SettingsRegistry())


def _make_rng() -> numpy.random.Generator:
    return numpy.random.default_rng(42)


class _StubProbeGeometryProvider(ProbeGeometryProvider):
    """A ready geometry provider, so the builders never hit the not-yet-bound guard."""

    def __init__(
        self,
        *,
        probe_photon_count: float = PROBE_PHOTON_COUNT,
        pixel_size_m: float = PIXEL_SIZE_M,
        focus_object_distance_m: float = 0.0,
    ) -> None:
        self._probe_photon_count = probe_photon_count
        self._pixel_size_m = pixel_size_m
        self._focus_object_distance_m = focus_object_distance_m

    @property
    def detector_distance_m(self) -> float:
        return 1.0

    @property
    def far_field(self) -> bool:
        return True

    @property
    def focus_object_distance_m(self) -> float:
        return self._focus_object_distance_m

    @property
    def probe_photon_count(self) -> float:
        return self._probe_photon_count

    @property
    def photon_wavelength_m(self) -> float:
        return 1.0e-10

    @property
    def probe_power_W(self) -> float:  # noqa: N802
        return 1.0

    @property
    def num_scan_points(self) -> int:
        return NUM_SCAN_POINTS

    def get_detector_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(width_m=PIXEL_SIZE_M, height_m=PIXEL_SIZE_M)

    def get_probe_geometry(self) -> ProbeGeometry:
        return ProbeGeometry(
            width_px=PROBE_EXTENT_PX,
            height_px=PROBE_EXTENT_PX,
            pixel_width_m=self._pixel_size_m,
            pixel_height_m=self._pixel_size_m,
        )


def _make_probe_seq(
    num_cmodes: int,
    num_imodes: int,
    *,
    with_opr_weights: bool = False,
    num_positions: int = NUM_SCAN_POINTS,
) -> ProbeEnsemble:
    """A deterministic, non-degenerate probe of the requested mode structure.

    ``num_positions`` sizes the OPR weights, so passing something other than
    ``NUM_SCAN_POINTS`` produces the probe of a *different* scan.
    """
    rng = numpy.random.default_rng(7)
    shape = (num_cmodes, num_imodes, 8, 8)
    array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(complex)

    opr_weights = None

    if with_opr_weights:
        opr_weights = rng.normal(size=(num_positions, num_cmodes))
        opr_weights[:, 0] = 1.0

    return ProbeEnsemble(
        array=array,
        opr_weights=opr_weights,
        pixel_geometry=PixelGeometry(width_m=PIXEL_SIZE_M, height_m=PIXEL_SIZE_M),
    )


class _StubProbeFileReader(ProbeFileReader):
    def __init__(self, probe_seq: ProbeEnsemble) -> None:
        self._probe_seq = probe_seq

    def get_file_description(self) -> str:
        return 'Stub Probe Files'

    def get_file_extensions(self) -> set[str]:
        return {'.npy'}

    def read(self, file_path: Path) -> ProbeEnsemble:
        return self._probe_seq


def _make_from_file_builder(
    settings: ProbeSettings,
    probe_seq: ProbeEnsemble,
    *,
    opr_weight_policy: OPRWeightPolicy | None = None,
) -> FromFileProbeBuilder:
    return FromFileProbeBuilder(
        _make_rng(),
        settings,
        _StubProbeFileReader(probe_seq),
        opr_weight_policy=opr_weight_policy,
    )


def _total_intensity(probe_seq: ProbeEnsemble) -> float:
    return float(numpy.sum(numpy.abs(probe_seq.get_array()) ** 2))


def test_generator_expands_incoherent_modes() -> None:
    """The refactor moved mode generation out of each generator's tail and into
    the base pipeline; generators must still come back multimodal."""
    settings = _make_settings()
    builder = DiskProbeBuilder(_make_rng(), settings)
    builder.num_incoherent_modes.set_value(4)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.num_incoherent_modes == 4
    assert probe_seq.num_coherent_modes == 1


def test_generator_expands_coherent_modes() -> None:
    settings = _make_settings()
    builder = DiskProbeBuilder(_make_rng(), settings)
    builder.num_coherent_modes.set_value(3)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.get_array().shape[:2] == (3, 1)

    opr_weights = probe_seq.get_opr_weights_or_none()
    assert opr_weights is not None
    assert opr_weights.shape == (NUM_SCAN_POINTS, 3)


def test_generator_rescales_to_photon_count() -> None:
    """rescale_probe_intensity was duplicated in all seven generator tails and is
    now a single base helper; the generative path must still be normalized."""
    settings = _make_settings()
    builder = DiskProbeBuilder(_make_rng(), settings)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert _total_intensity(probe_seq) == pytest.approx(PROBE_PHOTON_COUNT)


def test_from_file_builder_expands_modes() -> None:
    """FromFileProbeBuilder.build() used to return the reader's output verbatim,
    so the mode settings were silently ignored for every file-loaded probe even
    though the reconstructor honored them."""
    settings = _make_settings()
    builder = _make_from_file_builder(settings, _make_probe_seq(1, 1))
    builder.num_incoherent_modes.set_value(3)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.num_incoherent_modes == 3


def test_from_file_builder_does_not_rescale_intensity() -> None:
    """The photon-count rescale is generation-only. A file probe carries the
    intensity it was reconstructed at, and rescaling it without applying the
    reciprocal to a matching from-file object would break the product P*O."""
    settings = _make_settings()
    from_file = _make_probe_seq(1, 1)
    builder = _make_from_file_builder(settings, from_file)

    probe_seq = builder.build(_StubProbeGeometryProvider(probe_photon_count=1.0e12))

    assert _total_intensity(probe_seq) == pytest.approx(_total_intensity(from_file))


def test_from_file_builder_keeps_extra_incoherent_modes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Expand-only: asking for fewer modes than the file carries must not discard
    the converged ones."""
    settings = _make_settings()
    builder = _make_from_file_builder(settings, _make_probe_seq(1, 4))
    builder.num_incoherent_modes.set_value(1)

    with caplog.at_level('INFO'):
        probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.num_incoherent_modes == 4
    assert 'keeping them rather than discarding down to 1' in caplog.text


def test_from_file_builder_preserves_opr_basis(caplog: pytest.LogCaptureFixture) -> None:
    """The catastrophic case. generate_coherent_probe_modes would replace every
    coherent mode but the first with Gaussian noise and regenerate the OPR
    weights, so a solved OPR basis must never reach it."""
    settings = _make_settings()
    from_file = _make_probe_seq(3, 2, with_opr_weights=True)
    builder = _make_from_file_builder(settings, from_file)
    builder.num_coherent_modes.set_value(5)
    builder.num_incoherent_modes.set_value(6)

    with caplog.at_level('INFO'):
        probe_seq = builder.build(_StubProbeGeometryProvider())

    assert numpy.array_equal(probe_seq.get_array(), from_file.get_array())
    assert numpy.array_equal(
        probe_seq.get_opr_weights(),
        from_file.get_opr_weights(),
    )
    assert 'leaving its mode structure unchanged' in caplog.text


@pytest.mark.parametrize(('num_imodes', 'num_cmodes'), [(1, 1), (3, 1), (1, 3), (3, 2)])
def test_conditioning_is_idempotent(num_imodes: int, num_cmodes: int) -> None:
    """Conditioning an already-conditioned probe must be a no-op. Several rebuild
    paths -- a geometry-provider notification, a builder-parameter edit -- can
    re-run build() on a probe that has already been through the pipeline."""
    settings = _make_settings()
    provider = _StubProbeGeometryProvider()

    first = _make_from_file_builder(settings, _make_probe_seq(1, 1))
    first.num_incoherent_modes.set_value(num_imodes)
    first.num_coherent_modes.set_value(num_cmodes)
    conditioned = first.build(provider)

    second = _make_from_file_builder(settings, conditioned)
    second.num_incoherent_modes.set_value(num_imodes)
    second.num_coherent_modes.set_value(num_cmodes)
    reconditioned = second.build(provider)

    assert numpy.array_equal(reconditioned.get_array(), conditioned.get_array())

    weights = conditioned.get_opr_weights_or_none()
    reweights = reconditioned.get_opr_weights_or_none()

    if weights is None:
        assert reweights is None
    else:
        assert reweights is not None
        assert numpy.array_equal(reweights, weights)


def test_from_memory_builder_ignores_conditioning() -> None:
    """Guards reconstruction output: the from-memory builder must return its
    probe verbatim no matter what the mode parameters say."""
    settings = _make_settings()
    raw = _make_probe_seq(2, 3, with_opr_weights=True)
    builder = FromMemoryProbeBuilder(_make_rng(), settings, raw)
    builder.num_incoherent_modes.set_value(8)
    builder.num_coherent_modes.set_value(8)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert numpy.array_equal(probe_seq.get_array(), raw.get_array())
    assert numpy.array_equal(probe_seq.get_opr_weights(), raw.get_opr_weights())


def test_repeated_from_memory_builds_are_idempotent() -> None:
    """The reconstruct loop rebuilds the output item's probe once per iteration;
    conditioning must not accumulate across those rebuilds."""
    settings = _make_settings()
    settings.num_incoherent_modes.set_value(5)
    settings.num_coherent_modes.set_value(4)

    provider = _StubProbeGeometryProvider()
    expected = _make_probe_seq(2, 3, with_opr_weights=True)
    probe_seq = expected

    for _ in range(3):
        builder = FromMemoryProbeBuilder(_make_rng(), settings, probe_seq)
        probe_seq = builder.build(provider)

    assert numpy.array_equal(probe_seq.get_array(), expected.get_array())
    assert numpy.array_equal(probe_seq.get_opr_weights(), expected.get_opr_weights())


@pytest.mark.parametrize('builder_name', ['disk', 'from_file', 'kb_mirror'])
def test_copy_preserves_mode_parameters(builder_name: str) -> None:
    """copy() iterates parameters() generically and now also has to carry the rng
    hoisted into the base, so the copy must still build."""
    settings = _make_settings()
    builder: ProbeEnsembleBuilder

    if builder_name == 'disk':
        builder = DiskProbeBuilder(_make_rng(), settings)
    elif builder_name == 'kb_mirror':
        builder = KBMirrorProbeBuilder(_make_rng(), settings, PluginChooser())
    else:
        builder = _make_from_file_builder(settings, _make_probe_seq(1, 1))

    builder.num_incoherent_modes.set_value(3)
    builder.num_coherent_modes.set_value(2)

    duplicate = builder.copy()

    assert duplicate.num_incoherent_modes.get_value() == 3
    assert duplicate.num_coherent_modes.get_value() == 2
    assert duplicate.opr_weight_policy.get_value() == builder.opr_weight_policy.get_value()

    probe_seq = duplicate.build(_StubProbeGeometryProvider(pixel_size_m=KB_PIXEL_SIZE_M))
    assert probe_seq.get_array().shape[:2] == (2, 3)


def test_from_file_builder_resamples_a_probe_sampled_at_another_pixel_size() -> None:
    """A probe saved by a run at a different pixel size is an illumination of the wrong
    physical size; this is the regrid the builder's TODO used to defer."""
    settings = _make_settings()
    source = _make_probe_seq(1, 1)
    coarse = ProbeEnsemble(
        array=source.get_array(),
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=2.0 * PIXEL_SIZE_M, height_m=2.0 * PIXEL_SIZE_M),
    )
    builder = _make_from_file_builder(settings, coarse)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.get_pixel_geometry().width_m == pytest.approx(PIXEL_SIZE_M)
    assert probe_seq.width_px == PROBE_EXTENT_PX
    assert probe_seq.height_px == PROBE_EXTENT_PX
    # Photon-count invariance is asserted in test_probe.py against a smooth probe. This
    # fixture is white noise, whose power a resample legitimately smooths away: only
    # content the source grid resolves can survive being re-expressed on another one.


def test_from_file_builder_leaves_a_probe_without_a_pixel_size_alone() -> None:
    """Every registered probe format records no pixel size, so the common path must keep
    returning the array verbatim."""
    settings = _make_settings()
    source = _make_probe_seq(1, 1)
    bare = ProbeEnsemble(array=source.get_array(), opr_weights=None, pixel_geometry=None)
    builder = _make_from_file_builder(settings, bare)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    numpy.testing.assert_array_equal(probe_seq.get_array(), source.get_array())
    assert probe_seq.get_pixel_geometry().width_m == pytest.approx(PIXEL_SIZE_M)


class TestIncoherentModeDecaySettings:
    """The settings are looser than the api strategies, so the builder must absorb that.

    `IncoherentModeDecayRatio` permits 0.0 and the decay type is a free string, while
    `ProbeMomentPolynomialStrategy` rejects a non-positive ratio outright. A stored value
    the api would refuse has to leave the probe with a single occupied mode, not stop the
    build.
    """

    def _build_mode_powers(self, decay_type: str, decay_ratio: float) -> numpy.ndarray:
        settings = _make_settings()
        settings.num_incoherent_modes.set_value(4)
        settings.incoherent_mode_decay_type.set_value(decay_type)
        settings.incoherent_mode_decay_ratio.set_value(decay_ratio)

        probe_seq = DiskProbeBuilder(_make_rng(), settings).build(_StubProbeGeometryProvider())
        array = probe_seq.get_probe_no_opr().get_array()
        powers = numpy.array([numpy.square(numpy.abs(mode)).sum() for mode in array])
        return powers / powers.sum()

    def test_exponential_decay_reaches_the_modes(self) -> None:
        numpy.testing.assert_allclose(
            self._build_mode_powers('Exponential', 0.5),
            [8 / 15, 4 / 15, 2 / 15, 1 / 15],
            rtol=1e-9,
        )

    def test_polynomial_decay_reaches_the_modes(self) -> None:
        expected = numpy.array([1.0, 1 / 2, 1 / 3, 1 / 4])
        numpy.testing.assert_allclose(
            self._build_mode_powers('Polynomial', 0.5), expected / expected.sum(), rtol=1e-9
        )

    @pytest.mark.parametrize(
        ('decay_type', 'decay_ratio'),
        [('NotADecayType', 0.5), ('Exponential', 0.0)],
    )
    def test_a_value_the_api_would_reject_falls_back_to_one_occupied_mode(
        self, decay_type: str, decay_ratio: float
    ) -> None:
        powers = self._build_mode_powers(decay_type, decay_ratio)

        assert powers[0] == pytest.approx(1.0, rel=1e-9)
        numpy.testing.assert_allclose(powers[1:], 0.0, atol=1e-12)

    def test_the_photon_count_holds_whatever_the_decay_settings_say(self) -> None:
        for decay_type, decay_ratio in (('Exponential', 0.5), ('Exponential', 0.0)):
            settings = _make_settings()
            settings.num_incoherent_modes.set_value(4)
            settings.incoherent_mode_decay_type.set_value(decay_type)
            settings.incoherent_mode_decay_ratio.set_value(decay_ratio)

            probe_seq = DiskProbeBuilder(_make_rng(), settings).build(_StubProbeGeometryProvider())
            total = numpy.square(numpy.abs(probe_seq.get_array())).sum()

            assert total == pytest.approx(PROBE_PHOTON_COUNT, rel=1e-9)


class TestKBMirrorProbeBuilder:
    def _build(self, settings: ProbeSettings) -> ProbeEnsemble:
        builder = KBMirrorProbeBuilder(_make_rng(), settings, PluginChooser())
        return builder.build(_StubProbeGeometryProvider(pixel_size_m=KB_PIXEL_SIZE_M))

    def test_builds_a_probe_at_the_requested_photon_count(self) -> None:
        settings = _make_settings()
        settings.num_incoherent_modes.set_value(3)

        probe_seq = self._build(settings)

        assert probe_seq.num_incoherent_modes == 3
        assert probe_seq.num_coherent_modes == 1
        assert _total_intensity(probe_seq) == pytest.approx(PROBE_PHOTON_COUNT)

    def test_focus_narrows_as_the_numerical_aperture_grows(self) -> None:
        """A longer mirror subtends more angle, so it focuses tighter.

        This is the end-to-end check that the settings reach the optic rather than
        merely producing some probe.
        """

        def fwhm_px(acceptance_length_m: float) -> int:
            settings = _make_settings()
            settings.kb_horizontal_acceptance_length_m.set_value(acceptance_length_m)
            settings.kb_vertical_acceptance_length_m.set_value(acceptance_length_m)
            array = self._build(settings).get_probe_no_opr().get_array()
            cut = numpy.square(numpy.abs(array[0]))[PROBE_EXTENT_PX // 2, :]
            return int(numpy.count_nonzero(cut >= 0.5 * cut.max()))

        assert fwhm_px(0.2) < fwhm_px(0.05)

    def test_an_aperture_too_large_for_the_grid_is_refused(self) -> None:
        """The pupil window is lambda*z/dx_probe, so a coarse probe grid cannot hold a
        high-aperture optic; clipping it silently would quietly change the optic."""
        builder = KBMirrorProbeBuilder(_make_rng(), _make_settings(), PluginChooser())

        with pytest.raises(ValueError, match='does not fit the pupil window'):
            builder.build(_StubProbeGeometryProvider())

    def test_presets_are_empty_until_optics_are_registered(self) -> None:
        builder = KBMirrorProbeBuilder(_make_rng(), _make_settings(), PluginChooser())

        assert list(builder.labels_for_presets()) == []


class TestIncoherentModeStrategySettings:
    def _mode_powers(self, strategy: str, num_imodes: int = 3) -> numpy.ndarray:
        settings = _make_settings()
        settings.num_incoherent_modes.set_value(num_imodes)
        settings.incoherent_mode_strategy.set_value(strategy)
        settings.incoherent_mode_decay_type.set_value('Exponential')
        settings.incoherent_mode_decay_ratio.set_value(0.5)

        probe_seq = DiskProbeBuilder(_make_rng(), settings).build(_StubProbeGeometryProvider())
        array = probe_seq.get_probe_no_opr().get_array()
        powers = numpy.array([numpy.square(numpy.abs(mode)).sum() for mode in array])
        return powers / powers.sum()

    def test_every_selectable_name_resolves_to_its_own_strategy(self) -> None:
        """The offered names and the factories are one mapping, so they cannot drift.

        A name that is offered but unbuildable would silently fall back to the default,
        leaving a GUI option that quietly does something else.
        """
        settings = _make_settings()
        resolved = []

        for name in INCOHERENT_MODE_STRATEGY_NAMES:
            settings.incoherent_mode_strategy.set_value(name)
            builder = DiskProbeBuilder(_make_rng(), settings)
            resolved.append(type(builder._create_imode_strategy()))

        assert len(set(resolved)) == len(INCOHERENT_MODE_STRATEGY_NAMES)

    @pytest.mark.parametrize('strategy', INCOHERENT_MODE_STRATEGY_NAMES)
    def test_every_selectable_strategy_builds(self, strategy: str) -> None:
        powers = self._mode_powers(strategy)

        assert len(powers) == 3
        assert numpy.all(numpy.isfinite(powers))

    def test_the_decay_driven_strategies_follow_the_decay_settings(self) -> None:
        """Each reads the decay profile, so all land on the same weights."""
        expected = [4 / 7, 2 / 7, 1 / 7]

        numpy.testing.assert_allclose(self._mode_powers('MomentPolynomial'), expected, rtol=1e-9)
        numpy.testing.assert_allclose(self._mode_powers('RandomPhaseRamp'), expected, rtol=1e-9)
        numpy.testing.assert_allclose(self._mode_powers('Derivative'), expected, rtol=1e-9)

    def test_gaussian_schell_predicts_its_own_spectrum(self) -> None:
        """It ignores the decay profile, so its weights must differ from the others."""
        powers = self._mode_powers('GaussianSchell')

        assert not numpy.allclose(powers, [4 / 7, 2 / 7, 1 / 7], rtol=1e-3)

    def test_an_unrecognized_strategy_falls_back_to_the_first(self) -> None:
        """The settings accept any string while the api strategies do not, so a stored
        value they would refuse has to leave a usable probe."""
        numpy.testing.assert_allclose(
            self._mode_powers('NotAStrategy'),
            self._mode_powers(INCOHERENT_MODE_STRATEGY_NAMES[0]),
            rtol=1e-9,
        )

    def test_a_gaussian_schell_source_of_no_size_falls_back(self) -> None:
        """The settings permit a zero width where the model requires a positive one."""
        settings = _make_settings()
        settings.num_incoherent_modes.set_value(3)
        settings.incoherent_mode_strategy.set_value('GaussianSchell')
        settings.gaussian_schell_beam_width_m.set_value(0.0)

        probe_seq = DiskProbeBuilder(_make_rng(), settings).build(_StubProbeGeometryProvider())

        assert probe_seq.num_incoherent_modes == 3
        assert _total_intensity(probe_seq) == pytest.approx(PROBE_PHOTON_COUNT)


OTHER_SCAN_POSITIONS = NUM_SCAN_POINTS + 4
"""A probe-position count that is deliberately not this run's."""


def _opr_from_another_scan(num_cmodes: int = 3, num_imodes: int = 2) -> ProbeEnsemble:
    return _make_probe_seq(
        num_cmodes, num_imodes, with_opr_weights=True, num_positions=OTHER_SCAN_POSITIONS
    )


def test_from_file_builder_conforms_opr_weights_from_another_scan(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The motivating case: a solved OPR probe reused on a scan of a different size.

    Left alone the weights keep the old scan's row count, which pty-chi rejects outright
    and the per-position probe lookup turns into an IndexError. The default policy
    conforms them and says so.
    """
    settings = _make_settings()
    from_file = _opr_from_another_scan()
    builder = _make_from_file_builder(settings, from_file)

    with caplog.at_level('INFO'):
        probe_seq = builder.build(_StubProbeGeometryProvider())

    weights = probe_seq.get_opr_weights()
    assert weights.shape == (NUM_SCAN_POINTS, 3)
    assert probe_seq.num_opr_positions == NUM_SCAN_POINTS
    # The default is AVERAGE, so every position starts from the same row.
    numpy.testing.assert_allclose(
        weights,
        numpy.broadcast_to(from_file.get_opr_weights().mean(axis=0), weights.shape),
    )
    assert str(OTHER_SCAN_POSITIONS) in caplog.text
    assert 'AVERAGE' in caplog.text


def test_from_file_builder_leaves_matching_opr_weights_alone() -> None:
    """A warm start whose weights already fit must come through untouched."""
    settings = _make_settings()
    from_file = _make_probe_seq(3, 2, with_opr_weights=True)
    builder = _make_from_file_builder(settings, from_file)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    numpy.testing.assert_array_equal(probe_seq.get_opr_weights(), from_file.get_opr_weights())


@pytest.mark.parametrize(
    ('policy', 'expected_cmodes'),
    [
        (OPRWeightPolicy.AVERAGE, 3),
        (OPRWeightPolicy.REINITIALIZE, 3),
        (OPRWeightPolicy.COLLAPSE, 1),
        (OPRWeightPolicy.DISCARD, 1),
    ],
)
def test_from_file_builder_honors_the_policy_setting(
    policy: OPRWeightPolicy, expected_cmodes: int
) -> None:
    settings = _make_settings()
    settings.opr_weight_policy.set_value(policy.name)
    builder = _make_from_file_builder(settings, _opr_from_another_scan())

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.num_coherent_modes == expected_cmodes


def test_an_unknown_policy_name_still_builds() -> None:
    """The setting is a free string; a value with no matching policy must not stop the build."""
    settings = _make_settings()
    settings.opr_weight_policy.set_value('NotAPolicy')
    builder = _make_from_file_builder(settings, _opr_from_another_scan())

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.get_opr_weights().shape == (NUM_SCAN_POINTS, 3)


def test_an_override_applies_even_when_the_counts_already_agree() -> None:
    """A policy chosen for one ingest is a deliberate instruction, not a mismatch fix."""
    settings = _make_settings()
    from_file = _make_probe_seq(3, 2, with_opr_weights=True)
    builder = _make_from_file_builder(
        settings, from_file, opr_weight_policy=OPRWeightPolicy.REINITIALIZE
    )

    probe_seq = builder.build(_StubProbeGeometryProvider())

    weights = probe_seq.get_opr_weights()
    assert weights.shape == (NUM_SCAN_POINTS, 3)
    numpy.testing.assert_array_equal(weights[:, 0], 1.0)
    assert numpy.absolute(weights[:, 1:]).max() < 1.0e-5


def test_an_override_survives_copy() -> None:
    settings = _make_settings()
    builder = _make_from_file_builder(
        settings, _opr_from_another_scan(), opr_weight_policy=OPRWeightPolicy.DISCARD
    )

    probe_seq = builder.copy().build(_StubProbeGeometryProvider())

    assert probe_seq.num_coherent_modes == 1
    assert probe_seq.get_opr_weights_or_none() is None


def test_discarding_an_old_basis_composes_with_a_new_mode_count() -> None:
    """DISCARD runs before the expand-only pipeline, so a fresh basis can be requested."""
    settings = _make_settings()
    settings.opr_weight_policy.set_value(OPRWeightPolicy.DISCARD.name)
    builder = _make_from_file_builder(settings, _opr_from_another_scan(num_imodes=1))
    builder.num_coherent_modes.set_value(3)

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.num_coherent_modes == 3
    assert probe_seq.get_opr_weights().shape == (NUM_SCAN_POINTS, 3)


def test_from_memory_builder_conforms_opr_weights_from_another_scan() -> None:
    """Copying a probe between products reaches the same mismatch, through this builder."""
    settings = _make_settings()
    builder = FromMemoryProbeBuilder(_make_rng(), settings, _opr_from_another_scan())

    probe_seq = builder.build(_StubProbeGeometryProvider())

    assert probe_seq.get_opr_weights().shape == (NUM_SCAN_POINTS, 3)


def test_from_memory_builder_is_untouched_when_the_counts_agree() -> None:
    """The regression guard for reconstruction output.

    ProcessingTaskMonitor re-assigns the reconstructor's probe through this builder on
    every iteration, and those weights always match the run. Conforming must be a strict
    no-op there, or the reconstruction is averaged away one iteration at a time.
    """
    settings = _make_settings()
    settings.opr_weight_policy.set_value(OPRWeightPolicy.REINITIALIZE.name)
    in_memory = _make_probe_seq(3, 2, with_opr_weights=True)
    builder = FromMemoryProbeBuilder(_make_rng(), settings, in_memory)
    provider = _StubProbeGeometryProvider()

    for _ in range(3):
        probe_seq = builder.build(provider)
        numpy.testing.assert_array_equal(probe_seq.get_opr_weights(), in_memory.get_opr_weights())
        numpy.testing.assert_array_equal(probe_seq.get_array(), in_memory.get_array())


class TestFocusDistanceResolution:
    """Which of the two frames the focus is stated in wins, and when.

    A focusing builder's defocus and the product's focus-object distance are the
    same displacement with opposite signs. The product owns the geometry the
    reconstruction runs on, so it owns this too -- except where it declares no
    focusing optic, which cannot describe a focused probe.
    """

    def test_a_silent_product_leaves_the_builder_value_alone(self) -> None:
        # Zero means no focusing optic, which is every far-field configuration
        # written before the product carried this at all.
        assert resolve_defocus_distance_m(_StubProbeGeometryProvider(), 8.0e-4) == 8.0e-4

    def test_the_product_wins_when_it_declares_a_focus(self) -> None:
        provider = _StubProbeGeometryProvider(focus_object_distance_m=-8.0e-4)

        assert resolve_defocus_distance_m(provider, 0.0) == pytest.approx(8.0e-4)

    def test_the_sign_flips_between_the_two_frames(self) -> None:
        """A focus upstream of the object is a probe sampled downstream of the focus."""
        upstream = _StubProbeGeometryProvider(focus_object_distance_m=-8.0e-4)
        downstream = _StubProbeGeometryProvider(focus_object_distance_m=+8.0e-4)

        assert resolve_defocus_distance_m(upstream, 0.0) > 0.0
        assert resolve_defocus_distance_m(downstream, 0.0) < 0.0

    def test_agreement_is_quiet(self, caplog: pytest.LogCaptureFixture) -> None:
        provider = _StubProbeGeometryProvider(focus_object_distance_m=-8.0e-4)

        with caplog.at_level(logging.WARNING, logger='ptychodus.model.product.probe.builder'):
            resolve_defocus_distance_m(provider, 8.0e-4)

        assert not caplog.records

    def test_disagreement_warns_and_the_product_still_wins(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        provider = _StubProbeGeometryProvider(focus_object_distance_m=-8.0e-4)

        with caplog.at_level(logging.WARNING, logger='ptychodus.model.product.probe.builder'):
            resolved = resolve_defocus_distance_m(provider, 9.0e-4)

        assert resolved == pytest.approx(8.0e-4)
        assert len(caplog.records) == 1


def _every_builder(settings: ProbeSettings) -> list[ProbeEnsembleBuilder]:
    """One of each builder the factory offers, all reading the same settings."""
    registry = SettingsRegistry()
    task_manager = InlineTaskManager()
    dataset = AssembledDiffractionDataset(
        DiffractionSettings(registry),
        DetectorSettings(registry),
        task_manager,
        DiffractionTaskMonitor(task_manager),
    )
    factory = ProbeBuilderFactory(
        _make_rng(), settings, PluginChooser(), PluginChooser(), PluginChooser(), PluginChooser()
    )
    return [factory.create(name, dataset=dataset) for name in factory]


class TestPropagationDistanceIsSeparateFromDefocus:
    """Defocus and propagation are two settings that never stand in for each other.

    Defocus is the curvature of the beam a builder models; propagation moves the
    plane of a field already formed. Sharing one key once let editing a disk probe's
    propagation move a zone-plate probe's focus, so every builder reads each key on
    its own, whichever of the two it offers.
    """

    def test_every_builder_offers_at_least_one_of_the_two(self) -> None:
        builders = _every_builder(_make_settings())

        assert any('defocus_distance_m' in builder.parameters() for builder in builders)
        assert any('propagation_distance_m' in builder.parameters() for builder in builders)

    def test_setting_the_propagation_never_moves_a_defocus(self) -> None:
        settings = _make_settings()
        settings.propagation_distance_m.set_value(5.0e-5)

        for builder in _every_builder(settings):
            parameters = builder.parameters()

            if 'defocus_distance_m' in parameters:
                assert parameters['defocus_distance_m'].get_value() == 0.0, builder.get_name()

            if 'propagation_distance_m' in parameters:
                assert parameters['propagation_distance_m'].get_value() == pytest.approx(5.0e-5), (
                    builder.get_name()
                )

    def test_setting_the_defocus_never_moves_a_propagation(self) -> None:
        settings = _make_settings()
        settings.defocus_distance_m.set_value(5.0e-5)

        for builder in _every_builder(settings):
            parameters = builder.parameters()

            if 'propagation_distance_m' in parameters:
                assert parameters['propagation_distance_m'].get_value() == 0.0, builder.get_name()

            if 'defocus_distance_m' in parameters:
                assert parameters['defocus_distance_m'].get_value() == pytest.approx(5.0e-5), (
                    builder.get_name()
                )

    def test_the_focusing_builders_ignore_the_propagation_key(self) -> None:
        settings = _make_settings()
        settings.propagation_distance_m.set_value(5.0e-5)

        builder = KBMirrorProbeBuilder(_make_rng(), settings, PluginChooser())

        assert builder.defocus_distance_m.get_value() == 0.0
        assert 'propagation_distance_m' not in builder.parameters()


_ParametricProbeBuilder = DiskProbeBuilder | RectangularProbeBuilder | SuperGaussianProbeBuilder
# Past the 1 mm the disk needs to stay unaliased on the stub's 100 nm grid, and short
# of the 2.5 mm where its Fresnel number drops below one.
_PARAMETRIC_DEFOCUS_M = 2.0e-3


def _make_parametric_builder(name: str, settings: ProbeSettings) -> _ParametricProbeBuilder:
    builder: _ParametricProbeBuilder

    match name:
        case 'disk':
            builder = DiskProbeBuilder(_make_rng(), settings)
        case 'rectangular':
            builder = RectangularProbeBuilder(_make_rng(), settings)
        case 'super_gaussian':
            builder = SuperGaussianProbeBuilder(_make_rng(), settings)
            # The 400 um default would fill the grid edge to edge; at 0.6 um the 1 %
            # tail still stays unaliased at the test defocus.
            builder.fwhm_m.set_value(0.6e-6)
        case _:
            raise ValueError(name)

    return builder


def _flat_amplitude(builder: _ParametricProbeBuilder) -> Probe:
    """The builder's amplitude as its generator alone produces it."""
    geometry = _StubProbeGeometryProvider().get_probe_geometry()

    if isinstance(builder, DiskProbeBuilder):
        return generate_disk_probe(geometry, radius_m=builder.diameter_m.get_value() / 2.0)
    elif isinstance(builder, RectangularProbeBuilder):
        return generate_rectangular_probe(
            geometry, width_m=builder.width_m.get_value(), height_m=builder.height_m.get_value()
        )
    else:
        return generate_super_gaussian_probe(
            geometry,
            annular_radius_m=builder.annular_radius_m.get_value(),
            fwhm_m=builder.fwhm_m.get_value(),
            order_parameter=builder.order_parameter.get_value(),
        )


_PARAMETRIC_BUILDER_NAMES = ['disk', 'rectangular', 'super_gaussian']


class TestParametricProbeDefocus:
    """Disk, rectangle and super-Gaussian take a defocus phase, then propagate."""

    @pytest.mark.parametrize('name', _PARAMETRIC_BUILDER_NAMES)
    def test_zero_defocus_and_propagation_is_the_bare_amplitude(self, name: str) -> None:
        builder = _make_parametric_builder(name, _make_settings())
        expected = rescale_probe_intensity(_flat_amplitude(builder), PROBE_PHOTON_COUNT)

        probe = builder.build(_StubProbeGeometryProvider())

        numpy.testing.assert_array_equal(probe.get_array()[0], expected.get_array())

    @pytest.mark.parametrize('name', ['disk', 'rectangular'])
    def test_skipping_the_zero_propagation_changes_only_round_off(self, name: str) -> None:
        """Disk and rectangle once ran a zero-distance FFT round trip; now they skip it."""
        builder = _make_parametric_builder(name, _make_settings())
        round_tripped = propagate_probe(
            _flat_amplitude(builder),
            photon_wavelength_m=_StubProbeGeometryProvider().photon_wavelength_m,
            propagation_distance_m=0.0,
        )
        expected = rescale_probe_intensity(round_tripped, PROBE_PHOTON_COUNT)

        probe = builder.build(_StubProbeGeometryProvider())

        numpy.testing.assert_allclose(
            probe.get_array()[0], expected.get_array(), rtol=0.0, atol=1.0e-9
        )

    def test_the_super_gaussian_propagates(self) -> None:
        settings = _make_settings()
        settings.propagation_distance_m.set_value(5.0e-5)
        builder = _make_parametric_builder('super_gaussian', settings)
        provider = _StubProbeGeometryProvider()
        propagated = propagate_probe(
            _flat_amplitude(builder),
            photon_wavelength_m=provider.photon_wavelength_m,
            propagation_distance_m=5.0e-5,
        )
        expected = rescale_probe_intensity(propagated, PROBE_PHOTON_COUNT)

        probe = builder.build(provider)

        numpy.testing.assert_allclose(probe.get_array()[0], expected.get_array(), atol=1e-9)

    @pytest.mark.parametrize('name', _PARAMETRIC_BUILDER_NAMES)
    def test_the_builder_defocus_curves_the_probe(self, name: str) -> None:
        flat = _make_parametric_builder(name, _make_settings()).build(_StubProbeGeometryProvider())
        builder = _make_parametric_builder(name, _make_settings())
        builder.defocus_distance_m.set_value(_PARAMETRIC_DEFOCUS_M)

        curved = builder.build(_StubProbeGeometryProvider())

        assert not numpy.allclose(curved.get_array(), flat.get_array())
        numpy.testing.assert_allclose(numpy.abs(curved.get_array()), numpy.abs(flat.get_array()))

    @pytest.mark.parametrize('name', _PARAMETRIC_BUILDER_NAMES)
    def test_a_declared_focus_stands_in_for_the_builder_defocus(self, name: str) -> None:
        builder = _make_parametric_builder(name, _make_settings())
        builder.defocus_distance_m.set_value(_PARAMETRIC_DEFOCUS_M)
        from_builder = builder.build(_StubProbeGeometryProvider())

        builder.defocus_distance_m.set_value(0.0)
        from_product = builder.build(
            _StubProbeGeometryProvider(focus_object_distance_m=-_PARAMETRIC_DEFOCUS_M)
        )

        numpy.testing.assert_allclose(from_product.get_array(), from_builder.get_array())

    @pytest.mark.parametrize('name', _PARAMETRIC_BUILDER_NAMES)
    def test_the_declared_focus_wins_a_disagreement(
        self, name: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        expected_builder = _make_parametric_builder(name, _make_settings())
        expected_builder.defocus_distance_m.set_value(_PARAMETRIC_DEFOCUS_M)
        expected = expected_builder.build(_StubProbeGeometryProvider())
        builder = _make_parametric_builder(name, _make_settings())
        builder.defocus_distance_m.set_value(1.5e-3)

        with caplog.at_level(logging.WARNING, logger='ptychodus.model.product.probe.builder'):
            probe = builder.build(
                _StubProbeGeometryProvider(focus_object_distance_m=-_PARAMETRIC_DEFOCUS_M)
            )

        numpy.testing.assert_allclose(probe.get_array(), expected.get_array())
        assert 'disagrees' in caplog.text


# ---------------------------------------------------------------------------
# Size estimates and the average-pattern options
# ---------------------------------------------------------------------------

_DQ = 1.0 / (PROBE_EXTENT_PX * PIXEL_SIZE_M)
_ESTIMATE_DEFOCUS_M = 2.0e-3


def _stub_dataset(mean_pattern: numpy.ndarray) -> AssembledDiffractionDataset:
    """A dataset whose assembled data has exactly *mean_pattern* as its mean."""
    data = AssembledDiffractionData(
        indexes=numpy.arange(2),
        patterns=numpy.stack([mean_pattern, mean_pattern]),
        # The detector pitch the stub geometry implies: dx = lambda z / (N p).
        pixel_geometry=PixelGeometry(
            width_m=1.0e-10 * 1.0 / (PROBE_EXTENT_PX * PIXEL_SIZE_M),
            height_m=1.0e-10 * 1.0 / (PROBE_EXTENT_PX * PIXEL_SIZE_M),
        ),
        bad_pixels=numpy.zeros(mean_pattern.shape, dtype=bool),
    )
    dataset = MagicMock(spec=AssembledDiffractionDataset)
    dataset.get_assembled_data.return_value = data
    return cast(AssembledDiffractionDataset, dataset)


def _frequency_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    q = (numpy.arange(PROBE_EXTENT_PX) - PROBE_EXTENT_PX // 2) * _DQ
    return q.reshape(1, -1), q.reshape(-1, 1)


def _make_estimating_builder(
    name: str, dataset: AssembledDiffractionDataset | None
) -> _ParametricProbeBuilder:
    builder: _ParametricProbeBuilder

    match name:
        case 'disk':
            builder = DiskProbeBuilder(_make_rng(), _make_settings(), dataset)
        case 'rectangular':
            builder = RectangularProbeBuilder(_make_rng(), _make_settings(), dataset)
        case 'super_gaussian':
            builder = SuperGaussianProbeBuilder(_make_rng(), _make_settings(), dataset)
        case _:
            raise ValueError(name)

    builder.defocus_distance_m.set_value(_ESTIMATE_DEFOCUS_M)
    return builder


class TestSizeEstimateFromDiffraction:
    """The parametric builders set their size from the measured bright field, once."""

    def test_a_disk_takes_the_diameter_of_its_shadow(self) -> None:
        qx, qy = _frequency_grid()
        mean_pattern = 100.0 * (numpy.hypot(qx, qy) <= 8.0 * _DQ)
        builder = _make_estimating_builder('disk', _stub_dataset(mean_pattern))

        assert builder.estimate_size_from_diffraction(_StubProbeGeometryProvider())

        expected_m = 2.0 * 1.0e-10 * _ESTIMATE_DEFOCUS_M * 8.0 * _DQ
        assert isinstance(builder, DiskProbeBuilder)
        assert builder.diameter_m.get_value() == pytest.approx(expected_m, rel=0.05)

    def test_a_rectangle_takes_both_sides_of_its_shadow(self) -> None:
        qx, qy = _frequency_grid()
        mean_pattern = 100.0 * ((numpy.abs(qx) <= 8.0 * _DQ) & (numpy.abs(qy) <= 4.0 * _DQ))
        builder = _make_estimating_builder('rectangular', _stub_dataset(mean_pattern))

        assert builder.estimate_size_from_diffraction(_StubProbeGeometryProvider())

        # The aperture spans 2 * half + 1 pixels of the frequency grid.
        scale_m = 1.0e-10 * _ESTIMATE_DEFOCUS_M * _DQ
        assert isinstance(builder, RectangularProbeBuilder)
        assert builder.width_m.get_value() == pytest.approx(17.0 * scale_m, rel=0.03)
        assert builder.height_m.get_value() == pytest.approx(9.0 * scale_m, rel=0.03)

    def test_a_super_gaussian_takes_the_fwhm_of_a_gaussian_beam(self) -> None:
        qx, qy = _frequency_grid()
        sigma_q = 3.0 * _DQ
        mean_pattern = 100.0 * numpy.exp(-(qx**2 + qy**2) / (2.0 * sigma_q**2))
        builder = _make_estimating_builder('super_gaussian', _stub_dataset(mean_pattern))

        assert builder.estimate_size_from_diffraction(_StubProbeGeometryProvider())

        sigma_m = math.hypot(
            1.0 / (4.0 * math.pi * sigma_q), 1.0e-10 * _ESTIMATE_DEFOCUS_M * sigma_q
        )
        assert isinstance(builder, SuperGaussianProbeBuilder)
        assert builder.fwhm_m.get_value() == pytest.approx(
            2.0 * math.sqrt(2.0 * math.log(2.0)) * sigma_m, rel=0.03
        )

    @pytest.mark.parametrize('name', _PARAMETRIC_BUILDER_NAMES)
    def test_without_a_dataset_nothing_is_estimated(self, name: str) -> None:
        builder = _make_estimating_builder(name, None)
        before = {key: value.get_value() for key, value in builder.parameters().items()}

        assert not builder.can_estimate_size_from_diffraction()
        assert not builder.estimate_size_from_diffraction(_StubProbeGeometryProvider())
        assert {key: value.get_value() for key, value in builder.parameters().items()} == before

    @pytest.mark.parametrize('name', _PARAMETRIC_BUILDER_NAMES)
    def test_a_copy_keeps_the_dataset(self, name: str) -> None:
        dataset = _stub_dataset(numpy.ones((PROBE_EXTENT_PX, PROBE_EXTENT_PX)))
        builder = _make_estimating_builder(name, dataset)

        assert builder.copy().can_estimate_size_from_diffraction()

    def test_the_factory_hands_the_dataset_to_the_parametric_builders(self) -> None:
        dataset = _stub_dataset(numpy.ones((PROBE_EXTENT_PX, PROBE_EXTENT_PX)))
        factory = ProbeBuilderFactory(
            _make_rng(),
            _make_settings(),
            PluginChooser(),
            PluginChooser(),
            PluginChooser(),
            PluginChooser(),
        )

        for name in _PARAMETRIC_BUILDER_NAMES:
            assert factory.create(name, dataset=dataset).can_estimate_size_from_diffraction()
            assert not factory.create(name).can_estimate_size_from_diffraction()

        assert factory.create_default().get_name() == 'disk'


class TestAveragePatternOptions:
    """The average-pattern builder's defocus, propagation and bright-field threshold."""

    def _mean_pattern(self) -> numpy.ndarray:
        qx, qy = _frequency_grid()
        return 100.0 * (numpy.hypot(qx, qy) <= 8.0 * _DQ) + 2.0

    def _expected(self, **kwargs: Any) -> numpy.ndarray:
        provider = _StubProbeGeometryProvider()
        data = _stub_dataset(self._mean_pattern()).get_assembled_data()
        probe = generate_average_pattern_probe(
            provider.get_probe_geometry(),
            data.get_mean_pattern(),
            detector_pixel_geometry=data.get_pixel_geometry(),
            bad_pixels=data.get_bad_pixels(),
            photon_wavelength_m=provider.photon_wavelength_m,
            detector_distance_m=provider.detector_distance_m,
            **kwargs,
        )
        return rescale_probe_intensity(probe, PROBE_PHOTON_COUNT).get_array()

    def _builder(self) -> AveragePatternProbeBuilder:
        return AveragePatternProbeBuilder(
            _make_rng(), _make_settings(), _stub_dataset(self._mean_pattern())
        )

    def test_defocus_and_propagation_add(self) -> None:
        builder = self._builder()
        builder.defocus_distance_m.set_value(2.0e-6)
        builder.propagation_distance_m.set_value(1.0e-6)

        built = builder.build(_StubProbeGeometryProvider()).get_probe_no_opr().get_array()

        numpy.testing.assert_allclose(built, self._expected(defocus_distance_m=3.0e-6), atol=1e-9)

    def test_a_declared_focus_stands_in_for_the_builder_defocus(self) -> None:
        builder = self._builder()
        provider = _StubProbeGeometryProvider(focus_object_distance_m=-2.0e-6)

        built = builder.build(provider).get_probe_no_opr().get_array()

        numpy.testing.assert_allclose(built, self._expected(defocus_distance_m=2.0e-6), atol=1e-9)

    def test_the_bright_field_threshold_is_off_by_default_and_passes_through(self) -> None:
        builder = self._builder()
        provider = _StubProbeGeometryProvider()

        numpy.testing.assert_allclose(
            builder.build(provider).get_probe_no_opr().get_array(), self._expected(), atol=1e-9
        )

        builder.bright_field_threshold.set_value(True)

        numpy.testing.assert_allclose(
            builder.build(provider).get_probe_no_opr().get_array(),
            self._expected(bright_field_mad_threshold=4.5),
            atol=1e-9,
        )
