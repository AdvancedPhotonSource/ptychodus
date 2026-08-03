"""Regression tests for the try_get_object helper.

The critical invariant: Object.get_pixel_geometry() raises ValueError on the
null sentinel that ObjectRepositoryItem holds before a dataset binds; the
controller layer must not let that ValueError escape into a Qt signal handler
(would abort the process). See CLAUDE fly001.ini bug report.
"""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import MagicMock

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry, ObjectGeometryProvider
from ptychodus.api.observer import Observable
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.settings import SettingsRegistry
from ptychodus.controller.object.tree_model import try_get_object
from ptychodus.model.product.object.builder import ObjectBuilder
from ptychodus.model.product.object.item import ObjectRepositoryItem
from ptychodus.model.product.object.settings import ObjectSettings


def test_try_get_object_returns_none_on_null_sentinel() -> None:
    """The null sentinel Object(array=None, pixel_geometry=None, center=None)
    is what ObjectRepositoryItem holds until _rebuild produces a real Object;
    try_get_object must catch the resulting ValueError and return None."""
    sentinel = Object(array=None, pixel_geometry=None, center=None)
    item = MagicMock(spec=ObjectRepositoryItem)
    item.get_object.return_value = sentinel

    assert try_get_object(item) is None


def test_try_get_object_returns_object_when_ready() -> None:
    """With a real pixel_geometry present, try_get_object returns the Object."""
    array = numpy.zeros((1, 4, 4), dtype=complex)
    ready = Object(
        array=array,
        pixel_geometry=PixelGeometry(width_m=1e-6, height_m=1e-6),
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
    )
    item = MagicMock(spec=ObjectRepositoryItem)
    item.get_object.return_value = ready

    result = try_get_object(item)
    assert result is ready


class _ObservableObjectProvider(ObjectGeometryProvider, Observable):
    """Test double: an Observable + ObjectGeometryProvider. set_geometry()
    mutates the returned geometry and fires notify_observers, mimicking what
    ProductGeometry.set_detector_extent does in production."""

    def __init__(self, geometry: ObjectGeometry) -> None:
        Observable.__init__(self)
        self._geometry = geometry

    def set_geometry(self, geometry: ObjectGeometry) -> None:
        self._geometry = geometry
        self.notify_observers()

    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return ()

    def get_object_geometry(self) -> ObjectGeometry:
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

    def build(
        self,
        geometry_provider: ObjectGeometryProvider,
        layer_spacing_m: Sequence[float],
    ) -> Object:
        self.build_calls.append(geometry_provider)
        return self._canned


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
    (matches the ProductGeometry.set_detector_extent path in production).
    Also verifies the is_valid guard blocks the initial rebuild when the
    provider reports zero-valued pixel dimensions.
    """
    registry = SettingsRegistry()
    settings = ObjectSettings(registry)
    provider = _ObservableObjectProvider(_make_object_geometry(0.0, 0.0))
    canned = Object(
        array=numpy.zeros((1, 4, 4), dtype=complex),
        pixel_geometry=PixelGeometry(width_m=1e-6, height_m=1e-6),
        center=ObjectCenter(coordinate_x_m=0.0, coordinate_y_m=0.0),
    )
    builder = _RecordingObjectBuilder(settings, canned)

    item = ObjectRepositoryItem(provider, settings, builder)
    assert builder.build_calls == []  # guard blocks initial rebuild

    provider.set_geometry(_make_object_geometry(1e-6, 1e-6))

    assert len(builder.build_calls) == 1
    assert item.get_object().get_array() is canned.get_array()
