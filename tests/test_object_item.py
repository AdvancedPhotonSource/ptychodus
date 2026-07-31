"""Regression tests for the try_get_object helper.

The critical invariant: Object.get_pixel_geometry() raises ValueError on the
null sentinel that ObjectRepositoryItem holds before a dataset binds; the
controller layer must not let that ValueError escape into a Qt signal handler
(would abort the process). See CLAUDE fly001.ini bug report.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import Object, ObjectCenter
from ptychodus.controller.object.tree_model import try_get_object
from ptychodus.model.product.object.item import ObjectRepositoryItem


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
