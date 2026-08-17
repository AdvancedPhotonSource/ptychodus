"""Regression tests for ProbeRepositoryItem._rebuild.

The critical invariant: when the geometry provider has not yet bound to a
dataset (its ProbeGeometry has zero-valued pixel dimensions), _rebuild must
NOT invoke the builder — otherwise builders like FZP that divide by
`geometry.width_m` crash with ZeroDivisionError. See CLAUDE fly001.ini bug
report.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.observer import Observable
from ptychodus.api.probe import ProbeGeometry, ProbeGeometryProvider, ProbeSequence
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.probe.builder import ProbeSequenceBuilder
from ptychodus.model.product.probe.item import ProbeRepositoryItem
from ptychodus.model.product.probe.settings import ProbeSettings


def _make_rng() -> numpy.random.Generator:
    return numpy.random.default_rng(42)


class _RecordingBuilder(ProbeSequenceBuilder):
    """Minimal builder that records build() invocations without touching numpy math."""

    def __init__(self, settings: ProbeSettings, probe_seq: ProbeSequence) -> None:
        super().__init__(_make_rng(), settings, 'recording')
        self._settings = settings
        self._probe_seq = probe_seq
        self.build_calls: list[ProbeGeometryProvider] = []

    def copy(self) -> _RecordingBuilder:
        return _RecordingBuilder(self._settings, self._probe_seq)

    def _build_raw(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        self.build_calls.append(geometry_provider)
        return self._probe_seq

    def build(self, geometry_provider: ProbeGeometryProvider) -> ProbeSequence:
        # These tests exercise _rebuild's geometry guard, not the conditioning
        # pipeline, so bypass it and hand back the canned sequence by identity.
        return self._build_raw(geometry_provider)


def _make_provider(pixel_width_m: float, pixel_height_m: float) -> MagicMock:
    provider = MagicMock(spec=ProbeGeometryProvider)
    provider.get_probe_geometry.return_value = ProbeGeometry(
        width_px=64,
        height_px=64,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
    )
    return provider


def _make_probe_seq(pixel_size_m: float) -> ProbeSequence:
    array = numpy.zeros((1, 4, 4), dtype=numpy.complex64)
    return ProbeSequence(
        array=array,
        opr_weights=None,
        pixel_geometry=PixelGeometry(width_m=pixel_size_m, height_m=pixel_size_m),
    )


def test_rebuild_skips_when_geometry_not_ready() -> None:
    """Pre-dataset startup: pixel dimensions are zero. Builder must NOT be called;
    the initial null ProbeSequence stays in place; no exception escapes."""
    registry = SettingsRegistry()
    settings = ProbeSettings(registry)
    provider = _make_provider(pixel_width_m=0.0, pixel_height_m=0.0)
    canned = _make_probe_seq(pixel_size_m=1e-6)
    builder = _RecordingBuilder(settings, canned)

    item = ProbeRepositoryItem(_make_rng(), provider, settings, builder)

    assert builder.build_calls == []
    # The null sentinel from ProbeRepositoryItem.__init__ has size 0; the canned
    # replacement would be shape (1, 4, 4). Same-size check would let a silent
    # overwrite slip past.
    assert item.get_probes().get_array().size == 0


def test_rebuild_fires_when_geometry_becomes_ready() -> None:
    """Once the provider reports a valid pixel geometry and the item is nudged
    (e.g. via set_builder from a settings change), the builder runs and its
    ProbeSequence replaces the null sentinel."""
    registry = SettingsRegistry()
    settings = ProbeSettings(registry)
    provider = _make_provider(pixel_width_m=0.0, pixel_height_m=0.0)
    canned = _make_probe_seq(pixel_size_m=2e-6)
    builder = _RecordingBuilder(settings, canned)

    item = ProbeRepositoryItem(_make_rng(), provider, settings, builder)
    assert builder.build_calls == []

    # Provider becomes ready (dataset would bind in production).
    provider.get_probe_geometry.return_value = ProbeGeometry(
        width_px=64,
        height_px=64,
        pixel_width_m=2e-6,
        pixel_height_m=2e-6,
    )
    # A settings change would normally re-fire _rebuild via the observer chain;
    # set_builder is the shortest public path that triggers a rebuild.
    replacement = _RecordingBuilder(settings, canned)
    item.set_builder(replacement)

    assert len(replacement.build_calls) == 1
    assert item.get_probes().get_array() is canned.get_array()


def test_rebuild_skips_when_only_one_dimension_is_zero() -> None:
    """The guard uses PixelGeometry.is_valid, which requires BOTH dims positive.
    An asymmetric zero (e.g. width provided, height missing) still blocks the
    rebuild — matches PixelGeometry.is_valid semantics."""
    registry = SettingsRegistry()
    settings = ProbeSettings(registry)
    provider = _make_provider(pixel_width_m=1e-6, pixel_height_m=0.0)
    canned = _make_probe_seq(pixel_size_m=1e-6)
    builder = _RecordingBuilder(settings, canned)

    item = ProbeRepositoryItem(_make_rng(), provider, settings, builder)

    assert builder.build_calls == []
    assert item.get_probes().get_array().size == 0


class _ObservableProbeProvider(ProbeGeometryProvider, Observable):
    """Test double: an Observable + ProbeGeometryProvider. Only get_probe_geometry
    is exercised by ProbeRepositoryItem's rebuild guard; the other abstract
    properties are stubbed with sensible defaults. set_geometry() mutates and
    fires notify_observers, mimicking what ProductGeometry.set_detector_extent
    does in production."""

    def __init__(self, geometry: ProbeGeometry) -> None:
        Observable.__init__(self)
        self._geometry = geometry

    def set_geometry(self, geometry: ProbeGeometry) -> None:
        self._geometry = geometry
        self.notify_observers()

    @property
    def detector_distance_m(self) -> float:
        return 1.0

    @property
    def probe_photon_count(self) -> float:
        return 1.0

    @property
    def probe_wavelength_m(self) -> float:
        return 1e-10

    @property
    def probe_power_W(self) -> float:  # noqa: N802
        return 1.0

    @property
    def num_scan_points(self) -> int:
        return 1

    def get_detector_pixel_geometry(self) -> PixelGeometry:
        return PixelGeometry(width_m=1e-6, height_m=1e-6)

    def get_probe_geometry(self) -> ProbeGeometry:
        return self._geometry


def test_rebuild_fires_on_geometry_observer_notification() -> None:
    """When the geometry provider is Observable, ProbeRepositoryItem should
    register itself and re-run _rebuild each time notify_observers fires
    (matches the ProductGeometry.set_detector_extent path in production).
    """
    registry = SettingsRegistry()
    settings = ProbeSettings(registry)
    provider = _ObservableProbeProvider(
        ProbeGeometry(width_px=64, height_px=64, pixel_width_m=0.0, pixel_height_m=0.0),
    )
    canned = _make_probe_seq(pixel_size_m=1e-6)
    builder = _RecordingBuilder(settings, canned)

    item = ProbeRepositoryItem(_make_rng(), provider, settings, builder)
    assert builder.build_calls == []  # guard blocks initial rebuild

    provider.set_geometry(
        ProbeGeometry(width_px=64, height_px=64, pixel_width_m=1e-6, pixel_height_m=1e-6),
    )

    assert len(builder.build_calls) == 1
    assert item.get_probes().get_array() is canned.get_array()
