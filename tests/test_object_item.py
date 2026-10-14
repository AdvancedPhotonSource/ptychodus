"""Regression tests for ObjectRepositoryItem's rebuild guard.

ObjectRepositoryItem must not build an Object while the geometry provider
still reports zero-valued pixel dimensions, and must rebuild once the provider
notifies that real dimensions have arrived. See CLAUDE fly001.ini bug report.
"""

from __future__ import annotations

from collections.abc import Sequence
import logging

import numpy
import pytest

from ptychodus.api.geometry import GeometryNotDefinedError, PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.observer import Observable
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.object.builder import ObjectBuilder
from ptychodus.model.product.object.item import ObjectRepositoryItem
from ptychodus.model.product.object.settings import ObjectSettings


class _ObservableObjectProvider(ObjectGeometryProvider, Observable):
    """Test double: an Observable + ObjectGeometryProvider. set_geometry()
    mutates the returned geometry and fires notify_observers, mimicking what
    ProductGeometryProvider.set_detector_extent does in production."""

    def __init__(self, geometry: ObjectGeometry) -> None:
        Observable.__init__(self)
        self._geometry = geometry

    def set_geometry(self, geometry: ObjectGeometry) -> None:
        self._geometry = geometry
        self.notify_observers()

    @property
    def photon_wavelength_m(self) -> float:
        return 1.0e-10

    @property
    def object_plane_propagation_distance_m(self) -> float:
        return 1.0

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return ()

    def get_object_geometry(self) -> ObjectGeometry:
        if self._geometry.pixel_width_m <= 0.0 or self._geometry.pixel_height_m <= 0.0:
            raise GeometryNotDefinedError('not bound')

        return self._geometry


class _RecordingObjectBuilder(ObjectBuilder):
    """Minimal builder that records build() calls and returns a canned Object."""

    def __init__(self, settings: ObjectSettings, canned: Object) -> None:
        super().__init__(settings, 'recording')
        self._settings = settings
        self._canned = canned
        self.build_calls: list[ObjectGeometryProvider] = []

    def copy(self) -> _RecordingObjectBuilder:
        return _RecordingObjectBuilder(self._settings, self._canned)

    def _build_raw(self, geometry_provider: ObjectGeometryProvider) -> Object:
        # See the note in tests/test_probe_item.py: real builders read the geometry
        # first, so an undetermined one raises before anything is recorded.
        geometry_provider.get_object_geometry()
        self.build_calls.append(geometry_provider)
        return self._canned

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        # These tests exercise rebuild's geometry guard, not the conditioning
        # pipeline, so bypass it and hand back the canned object by identity.
        return self._build_raw(geometry_provider)


def _make_object_geometry(pixel_width_m: float, pixel_height_m: float) -> ObjectGeometry:
    return ObjectGeometry(
        width_px=8,
        height_px=8,
        pixel_width_m=pixel_width_m,
        pixel_height_m=pixel_height_m,
        center_x_m=0.0,
        center_y_m=0.0,
    )


def test_rebuild_fires_on_geometry_observer_notification() -> None:
    """When the geometry provider is Observable, ObjectRepositoryItem should
    register itself and re-run rebuild each time notify_observers fires
    (matches the ProductGeometryProvider.set_detector_extent path in production).
    Also verifies that an undetermined geometry blocks the initial rebuild when the
    provider reports zero-valued pixel dimensions.
    """
    registry = SettingsRegistry()
    settings = ObjectSettings(registry)
    provider = _ObservableObjectProvider(_make_object_geometry(0.0, 0.0))
    canned = Object(
        array=numpy.zeros((1, 4, 4), dtype=complex),
        pixel_geometry=PixelGeometry(width_m=1e-6, height_m=1e-6),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
    )
    builder = _RecordingObjectBuilder(settings, canned)

    item = ObjectRepositoryItem(provider, settings, builder)
    assert builder.build_calls == []  # guard blocks initial rebuild

    provider.set_geometry(_make_object_geometry(1e-6, 1e-6))

    assert len(builder.build_calls) == 1
    assert item.get_object().get_array() is canned.get_array()


def _make_item_with_canned_pixel_size(
    pixel_m: float, provider_pixel_m: float
) -> tuple[ObjectRepositoryItem, Object]:
    registry = SettingsRegistry()
    settings = ObjectSettings(registry)
    provider = _ObservableObjectProvider(_make_object_geometry(provider_pixel_m, provider_pixel_m))
    canned = Object(
        array=numpy.zeros((1, 4, 4), dtype=complex),
        pixel_geometry=PixelGeometry(width_m=pixel_m, height_m=pixel_m),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
    )
    item = ObjectRepositoryItem(provider, settings, _RecordingObjectBuilder(settings, canned))
    return item, canned


def test_rebuild_warns_when_the_object_is_sampled_differently_from_the_product(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An already-conditioned object -- a product read back from file, or reconstruction
    output -- keeps the sampling it was saved with. Rebinding the product to another
    dataset moves the run's pixel size out from under it, which nothing else reveals."""
    with caplog.at_level(logging.WARNING, logger='ptychodus.model.product.object.item'):
        item, canned = _make_item_with_canned_pixel_size(2.0e-6, 1.0e-6)

    assert 'wrong scale' in caplog.text
    # Reported, not corrected: the same path carries reconstruction output.
    assert item.get_object().get_array() is canned.get_array()


def test_rebuild_is_silent_when_the_samplings_agree(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger='ptychodus.model.product.object.item'):
        _make_item_with_canned_pixel_size(1.0e-6, 1.0e-6)

    assert caplog.text == ''
