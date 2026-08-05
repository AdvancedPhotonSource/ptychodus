"""Regression tests for the probe conditioning pipeline (incoherent modes -> OPR modes).

Two invariants carry the weight here.

First, conditioning is expand-only and therefore idempotent. The generators in
ptychodus.api.probe_gen are not safe to re-apply: generate_incoherent_probe_modes
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
    ProbeFileReader,
    ProbeGeometry,
    ProbeGeometryProvider,
    ProbeSequence,
)
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.probe.builder import (
    FromFileProbeBuilder,
    FromMemoryProbeBuilder,
    ProbeSequenceBuilder,
)
from ptychodus.model.product.probe.disk import DiskProbeBuilder
from ptychodus.model.product.probe.settings import ProbeSettings

NUM_SCAN_POINTS = 7
PROBE_PHOTON_COUNT = 1.0e6
# Fine enough that the default 1 um disk covers several pixels; at a coarser
# pixel size the generated probe is empty and rescale_probe_intensity bails.
PIXEL_SIZE_M = 1.0e-7
PROBE_EXTENT_PX = 64


def _make_settings() -> ProbeSettings:
    return ProbeSettings(SettingsRegistry())


def _make_rng() -> numpy.random.Generator:
    return numpy.random.default_rng(42)


class _StubProbeGeometryProvider(ProbeGeometryProvider):
    """A ready geometry provider, so the builders never hit the not-yet-bound guard."""

    def __init__(self, *, probe_photon_count: float = PROBE_PHOTON_COUNT) -> None:
        self._probe_photon_count = probe_photon_count

    @property
    def detector_distance_m(self) -> float:
        return 1.0

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
            pixel_width_m=PIXEL_SIZE_M,
            pixel_height_m=PIXEL_SIZE_M,
        )


def _make_probe_seq(
    num_cmodes: int, num_imodes: int, *, with_opr_weights: bool = False
) -> ProbeSequence:
    """A deterministic, non-degenerate probe of the requested mode structure."""
    rng = numpy.random.default_rng(7)
    shape = (num_cmodes, num_imodes, 8, 8)
    array = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(complex)

    opr_weights = None

    if with_opr_weights:
        opr_weights = rng.normal(size=(NUM_SCAN_POINTS, num_cmodes))
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
    settings: ProbeSettings, probe_seq: ProbeSequence
) -> FromFileProbeBuilder:
    return FromFileProbeBuilder(_make_rng(), settings, _StubProbeFileReader(probe_seq))


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


@pytest.mark.parametrize('builder_name', ['disk', 'from_file'])
def test_copy_preserves_mode_parameters(builder_name: str) -> None:
    """copy() iterates parameters() generically and now also has to carry the rng
    hoisted into the base, so the copy must still build."""
    settings = _make_settings()
    builder: ProbeSequenceBuilder

    if builder_name == 'disk':
        builder = DiskProbeBuilder(_make_rng(), settings)
    else:
        builder = _make_from_file_builder(settings, _make_probe_seq(1, 1))

    builder.num_incoherent_modes.set_value(3)
    builder.num_coherent_modes.set_value(2)

    duplicate = builder.copy()

    assert duplicate.num_incoherent_modes.get_value() == 3
    assert duplicate.num_coherent_modes.get_value() == 2

    probe_seq = duplicate.build(_StubProbeGeometryProvider())
    assert probe_seq.get_array().shape[:2] == (2, 3)
