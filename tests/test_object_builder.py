"""Regression tests for the object conditioning pipeline (extra padding -> layers).

Two invariants carry the weight here.

First, the two operations sit on opposite sides of a split. generate_layers is
conditioning: it applies to generated and file-loaded objects alike, guarded so
it never destroys layers the input already has, because it truncates when asked
for fewer than it is given. pad_object is generation-only: it is strictly
additive and leaves no trace in the array, so there is no way to detect an
already-padded object and skip it. Applying it to a file-loaded object would grow
that object on every load/reconstruct/save/load round trip, unbounded -- and the
padding defaults to 1, so it would happen to users who never touched the setting.

Second, FromMemoryObjectBuilder must never condition. It holds an object that is
already conditioned -- reconstruction output, which ProcessingTaskMonitor
re-assigns to the output product item on every reconstructor iteration, and
products loaded from HDF5/NPZ. Padding there would grow the array once per
iteration, and generate_layers would collapse a converged multislice result back
to whatever the item's layer spacing happens to say.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy
import pytest

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.object import (
    Object,
    ObjectCenter,
    ObjectFileReader,
    ObjectGeometry,
    ObjectGeometryProvider,
)
from ptychodus.api.probe_positions import ProbePosition
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.object.builder import (
    FromFileObjectBuilder,
    FromMemoryObjectBuilder,
    ObjectBuilder,
)
from ptychodus.model.product.object.item import ObjectRepositoryItem
from ptychodus.model.product.object.random import RandomObjectBuilder
from ptychodus.model.product.object.settings import ObjectSettings

EXTENT_PX = 8
PIXEL_SIZE_M = 1.0e-6
LAYER_SPACING_M = 1.0e-6


def _make_settings() -> ObjectSettings:
    return ObjectSettings(SettingsRegistry())


def _make_rng() -> numpy.random.Generator:
    return numpy.random.default_rng(42)


class _StubObjectGeometryProvider(ObjectGeometryProvider):
    def get_probe_positions(self) -> Sequence[ProbePosition]:
        return ()

    def get_object_geometry(self) -> ObjectGeometry:
        return ObjectGeometry(
            width_px=EXTENT_PX,
            height_px=EXTENT_PX,
            pixel_width_m=PIXEL_SIZE_M,
            pixel_height_m=PIXEL_SIZE_M,
            center_x_m=0.0,
            center_y_m=0.0,
        )


def _make_object(num_layers: int) -> Object:
    """A deterministic, non-degenerate object with the requested layer count."""
    rng = numpy.random.default_rng(7)
    shape = (num_layers, EXTENT_PX, EXTENT_PX)
    # Keep the amplitude away from zero so the phase unwrapping in
    # generate_layers is well conditioned.
    array = (1.0 + 0.1 * rng.normal(size=shape)) * numpy.exp(1j * 0.1 * rng.normal(size=shape))
    return Object(
        array=array.astype(complex),
        pixel_geometry=PixelGeometry(width_m=PIXEL_SIZE_M, height_m=PIXEL_SIZE_M),
        center=ObjectCenter(x_m=0.0, y_m=0.0),
        layer_spacing_m=[LAYER_SPACING_M] * (num_layers - 1),
    )


class _StubObjectFileReader(ObjectFileReader):
    def __init__(self, object_: Object) -> None:
        self._object = object_

    def read(self, file_path: Path) -> Object:
        return self._object


def _make_from_file_builder(settings: ObjectSettings, object_: Object) -> FromFileObjectBuilder:
    return FromFileObjectBuilder(settings, _StubObjectFileReader(object_))


def test_generator_pads_canvas() -> None:
    """The padding moved from the shared pipeline into _build_raw; the generative
    path must still honor it."""
    settings = _make_settings()
    builder = RandomObjectBuilder(_make_rng(), settings)
    builder.extra_padding_x.set_value(3)
    builder.extra_padding_y.set_value(2)

    object_ = builder.build(_StubObjectGeometryProvider(), [])

    assert object_.width_px == EXTENT_PX + 6
    assert object_.height_px == EXTENT_PX + 4


def test_generator_generates_layers() -> None:
    settings = _make_settings()
    builder = RandomObjectBuilder(_make_rng(), settings)

    object_ = builder.build(_StubObjectGeometryProvider(), [LAYER_SPACING_M] * 2)

    assert object_.num_layers == 3
    assert list(object_.layer_spacing_m) == [LAYER_SPACING_M] * 2


def test_from_file_builder_generates_layers() -> None:
    """FromFileObjectBuilder.build() used to return the reader's output verbatim,
    so the layer spacing was silently ignored for every file-loaded object."""
    settings = _make_settings()
    builder = _make_from_file_builder(settings, _make_object(1))

    object_ = builder.build(_StubObjectGeometryProvider(), [LAYER_SPACING_M] * 2)

    assert object_.num_layers == 3
    assert list(object_.layer_spacing_m) == [LAYER_SPACING_M] * 2


def test_from_file_builder_keeps_existing_layers(caplog: pytest.LogCaptureFixture) -> None:
    """The destructive case: generate_layers truncates when asked for fewer layers
    than it is given, so the default empty spacing would collapse a converged
    four-layer warm start to one layer."""
    settings = _make_settings()
    from_file = _make_object(4)
    builder = _make_from_file_builder(settings, from_file)

    with caplog.at_level('INFO'):
        object_ = builder.build(_StubObjectGeometryProvider(), [])

    assert object_.num_layers == 4
    assert 'keeping them rather than re-slicing to 1' in caplog.text


def test_from_file_builder_does_not_pad() -> None:
    """Padding is generation-only. It is strictly additive and undetectable after
    the fact, so applying it here would grow a warm-start object by twice the
    padding on every load/save round trip -- with the default padding of 1, for
    users who never touched the setting."""
    settings = _make_settings()
    from_file = _make_object(1)
    builder = _make_from_file_builder(settings, from_file)

    assert builder.extra_padding_x.get_value() == 1
    assert builder.extra_padding_y.get_value() == 1

    object_ = builder.build(_StubObjectGeometryProvider(), [])

    assert object_.width_px == from_file.width_px
    assert object_.height_px == from_file.height_px


def test_from_file_conditioning_is_idempotent() -> None:
    """Several rebuild paths -- a geometry-provider notification, a
    builder-parameter edit -- can re-run build() on an already-conditioned
    object."""
    settings = _make_settings()
    provider = _StubObjectGeometryProvider()
    layer_spacing_m = [LAYER_SPACING_M] * 2

    conditioned = _make_from_file_builder(settings, _make_object(1)).build(
        provider, layer_spacing_m
    )
    reconditioned = _make_from_file_builder(settings, conditioned).build(provider, layer_spacing_m)

    assert numpy.array_equal(reconditioned.get_array(), conditioned.get_array())
    assert list(reconditioned.layer_spacing_m) == list(conditioned.layer_spacing_m)


def test_from_memory_builder_ignores_conditioning() -> None:
    """Guards reconstruction output: the from-memory builder must return its
    object verbatim no matter what the conditioning parameters say."""
    settings = _make_settings()
    raw = _make_object(3)
    builder = FromMemoryObjectBuilder(settings, raw)
    builder.extra_padding_x.set_value(5)
    builder.extra_padding_y.set_value(5)

    object_ = builder.build(_StubObjectGeometryProvider(), [])

    assert numpy.array_equal(object_.get_array(), raw.get_array())
    assert list(object_.layer_spacing_m) == list(raw.layer_spacing_m)


def test_repeated_from_memory_builds_are_idempotent() -> None:
    """The reconstruct loop rebuilds the output item's object once per iteration.
    pad_object is strictly additive, so conditioning here would grow the array
    without bound."""
    settings = _make_settings()
    settings.extra_padding_x.set_value(4)
    settings.extra_padding_y.set_value(4)

    provider = _StubObjectGeometryProvider()
    expected = _make_object(3)
    object_ = expected

    for _ in range(3):
        builder = FromMemoryObjectBuilder(settings, object_)
        object_ = builder.build(provider, [])

    assert object_.get_array().shape == expected.get_array().shape
    assert numpy.array_equal(object_.get_array(), expected.get_array())
    assert list(object_.layer_spacing_m) == list(expected.layer_spacing_m)


def test_item_adopts_layer_spacing_actually_produced() -> None:
    """ObjectRepositoryItem.rebuild overwrites its own layer_spacing_m parameter
    from whatever the builder returned, so the parameter tracks the layers the
    object really has rather than the ones that were asked for."""
    settings = _make_settings()
    builder = _make_from_file_builder(settings, _make_object(4))

    item = ObjectRepositoryItem(_StubObjectGeometryProvider(), settings, builder)

    assert item.get_object().num_layers == 4
    assert list(item.layer_spacing_m.get_value()) == [LAYER_SPACING_M] * 3


@pytest.mark.parametrize('builder_name', ['random', 'from_file'])
def test_copy_preserves_padding_parameters(builder_name: str) -> None:
    """copy() iterates parameters() generically, so the padding rides along
    without any per-subclass change."""
    settings = _make_settings()
    builder: ObjectBuilder

    if builder_name == 'random':
        builder = RandomObjectBuilder(_make_rng(), settings)
    else:
        builder = _make_from_file_builder(settings, _make_object(1))

    builder.extra_padding_x.set_value(2)
    builder.extra_padding_y.set_value(3)

    duplicate = builder.copy()

    assert duplicate.extra_padding_x.get_value() == 2
    assert duplicate.extra_padding_y.get_value() == 3
