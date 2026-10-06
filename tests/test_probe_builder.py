"""Regression tests for the probe conditioning pipeline (incoherent modes -> OPR modes).

Two invariants carry the weight here.

First, conditioning is expand-only and therefore idempotent. The generators in
ptychodus.api.simulate.probe are not safe to re-apply: generate_incoherent_probe_modes
re-orthogonalizes and renormalizes every mode to the decay profile, and
generate_coherent_probe_modes fills its output with fresh Gaussian noise, keeps
only coherent mode zero of its input, and regenerates the OPR weights from
scratch. The guards in ProbeSequenceBuilder._condition_probe are what stop a
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

import numpy
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.probe import (
    OPRWeightPolicy,
    ProbeFileReader,
    ProbeGeometry,
    ProbeGeometryProvider,
    ProbeSequence,
)
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.probe.builder import (
    FromFileProbeBuilder,
    FromMemoryProbeBuilder,
    ProbeSequenceBuilder,
)
from ptychodus.model.product.probe import INCOHERENT_MODE_STRATEGY_NAMES
from ptychodus.model.product.probe.disk import DiskProbeBuilder
from ptychodus.model.product.probe.kb_mirror import KBMirrorProbeBuilder
from ptychodus.model.product.probe.settings import ProbeSettings

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
    ) -> None:
        self._probe_photon_count = probe_photon_count
        self._pixel_size_m = pixel_size_m

    @property
    def detector_distance_m(self) -> float:
        return 1.0

    @property
    def far_field(self) -> bool:
        return True

    @property
    def probe_photon_count(self) -> float:
        return self._probe_photon_count

    @property
    def probe_wavelength_m(self) -> float:
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
) -> ProbeSequence:
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

    return ProbeSequence(
        array=array,
        opr_weights=opr_weights,
        pixel_geometry=PixelGeometry(width_m=PIXEL_SIZE_M, height_m=PIXEL_SIZE_M),
    )


class _StubProbeFileReader(ProbeFileReader):
    def __init__(self, probe_seq: ProbeSequence) -> None:
        self._probe_seq = probe_seq

    def read(self, file_path: Path) -> ProbeSequence:
        return self._probe_seq


def _make_from_file_builder(
    settings: ProbeSettings,
    probe_seq: ProbeSequence,
    *,
    opr_weight_policy: OPRWeightPolicy | None = None,
) -> FromFileProbeBuilder:
    return FromFileProbeBuilder(
        _make_rng(),
        settings,
        _StubProbeFileReader(probe_seq),
        opr_weight_policy=opr_weight_policy,
    )


def _total_intensity(probe_seq: ProbeSequence) -> float:
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
    builder: ProbeSequenceBuilder

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
    coarse = ProbeSequence(
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
    bare = ProbeSequence(array=source.get_array(), opr_weights=None, pixel_geometry=None)
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
    def _build(self, settings: ProbeSettings) -> ProbeSequence:
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
        """Both read the decay profile, so both land on the same weights."""
        expected = [4 / 7, 2 / 7, 1 / 7]

        numpy.testing.assert_allclose(self._mode_powers('MomentPolynomial'), expected, rtol=1e-9)
        numpy.testing.assert_allclose(self._mode_powers('RandomPhaseRamp'), expected, rtol=1e-9)

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


def _opr_from_another_scan(num_cmodes: int = 3, num_imodes: int = 2) -> ProbeSequence:
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
    assert len(probe_seq) == NUM_SCAN_POINTS
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
