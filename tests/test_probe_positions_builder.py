"""Regression tests for the probe-position conditioning pipeline (trim -> affine -> jitter).

Two invariants carry the weight here.

First, trimming discards points by acquisition order but must leave the surviving
points' scan indexes untouched. prepare_reconstruct_input
joins diffraction patterns to positions by index and refuses to extrapolate beyond
the position-index anchors, so renumbering the survivors would silently pair every
pattern with the wrong position.

Second, FromMemoryProbePositionsBuilder must never condition. It holds positions
that are already conditioned -- reconstruction output, which ProcessingTaskMonitor
re-assigns to the output product item on every reconstructor iteration, and
products loaded from HDF5/NPZ. Re-applying the trim or the affine transform there
would corrupt position-corrected output a little more on every iteration.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pytest

from ptychodus.api.probe_positions import (
    ProbePosition,
    ProbePositionFileReader,
    ProbePositionSequence,
)
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.probe_positions.builder import (
    FromFileProbePositionsBuilder,
    FromMemoryProbePositionsBuilder,
)
from ptychodus.model.product.probe_positions.cartesian import (
    CartesianProbePositionsBuilder,
    CartesianProbePositionsVariant,
)
from ptychodus.model.product.probe_positions.settings import ProbePositionsSettings
from ptychodus.model.product.probe_positions.streaming import StreamingScanBuilder


def _make_settings() -> ProbePositionsSettings:
    return ProbePositionsSettings(SettingsRegistry())


def _make_rng() -> numpy.random.Generator:
    return numpy.random.default_rng(42)


def _make_line(num_points: int) -> ProbePositionSequence:
    """A horizontal line of positions whose index equals its ordinal."""
    return ProbePositionSequence(
        [
            ProbePosition(index=idx, coordinate_x_m=float(idx), coordinate_y_m=0.0)
            for idx in range(num_points)
        ]
    )


class _StubPositionFileReader(ProbePositionFileReader):
    def __init__(self, positions: ProbePositionSequence) -> None:
        self._positions = positions

    def read(self, file_path: Path) -> ProbePositionSequence:
        return self._positions


def _make_cartesian_line_builder(
    settings: ProbePositionsSettings, num_points: int
) -> CartesianProbePositionsBuilder:
    """A single-row raster, so acquisition order is unambiguous."""
    builder = CartesianProbePositionsBuilder(
        CartesianProbePositionsVariant.RECTANGULAR_RASTER, _make_rng(), settings
    )
    builder.num_points_x.set_value(num_points)
    builder.num_points_y.set_value(1)
    return builder


def test_slice_returns_subsequence() -> None:
    """ProbePositionSequence.__getitem__ used to raise TypeError on any slice
    with an implicit bound, because it built a range() from the raw slice
    attributes. The trim needs slicing to work."""
    seq = _make_line(5)

    assert [p.index for p in seq[:2]] == [0, 1]
    assert [p.index for p in seq[1:3]] == [1, 2]
    assert [p.index for p in seq[-1:]] == [4]
    assert len(seq[5:5]) == 0
    assert [p.coordinate_x_m for p in seq[1:3]] == [1.0, 2.0]


def test_generator_builder_trims_by_acquisition_order() -> None:
    settings = _make_settings()
    builder = _make_cartesian_line_builder(settings, 5)
    builder.num_discard_at_start.set_value(1)
    builder.num_discard_at_end.set_value(1)

    assert len(builder.build()) == 3


def test_trim_preserves_original_indexes() -> None:
    """The load-bearing invariant for prepare_reconstruct_input's index anchoring:
    surviving points keep the indexes they were acquired with."""
    settings = _make_settings()
    builder = _make_cartesian_line_builder(settings, 5)
    builder.num_discard_at_start.set_value(1)
    builder.num_discard_at_end.set_value(1)

    assert [p.index for p in builder.build()] == [1, 2, 3]


def test_over_trim_yields_empty_sequence_and_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings = _make_settings()
    builder = _make_cartesian_line_builder(settings, 5)
    builder.num_discard_at_start.set_value(3)
    builder.num_discard_at_end.set_value(3)

    with caplog.at_level('WARNING'):
        positions = builder.build()

    assert len(positions) == 0
    assert 'leaves nothing of 5' in caplog.text


def test_from_file_builder_applies_affine_and_trim() -> None:
    """FromFileProbePositionsBuilder.build() used to return the reader's output
    verbatim, so the affine transform and jitter were silently ignored for every
    file-loaded scan even though the GUI offered the transform editor."""
    settings = _make_settings()
    reader = _StubPositionFileReader(_make_line(5))
    builder = FromFileProbePositionsBuilder(_make_rng(), settings, reader)
    builder.affine00.set_value(-1.0)
    builder.num_discard_at_start.set_value(1)
    builder.num_discard_at_end.set_value(1)

    positions = builder.build()

    assert [p.index for p in positions] == [1, 2, 3]
    assert [p.coordinate_x_m for p in positions] == [-1.0, -2.0, -3.0]


def test_from_memory_builder_ignores_conditioning() -> None:
    """Guards reconstruction output: the from-memory builder must return its
    positions verbatim no matter what the conditioning parameters say."""
    settings = _make_settings()
    raw = _make_line(5)
    builder = FromMemoryProbePositionsBuilder(_make_rng(), settings, raw)
    builder.affine00.set_value(-1.0)
    builder.num_discard_at_start.set_value(2)
    builder.num_discard_at_end.set_value(2)
    builder.jitter_radius_m.set_value(1e-6)

    positions = builder.build()

    assert [p.index for p in positions] == [0, 1, 2, 3, 4]
    assert [p.coordinate_x_m for p in positions] == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_repeated_from_memory_builds_are_idempotent() -> None:
    """The reconstruct loop rebuilds the output item's positions once per
    iteration; conditioning must not accumulate across those rebuilds."""
    settings = _make_settings()
    settings.num_discard_at_start.set_value(1)
    settings.affine00.set_value(-1.0)

    positions = _make_line(5)

    for _ in range(3):
        builder = FromMemoryProbePositionsBuilder(_make_rng(), settings, positions)
        positions = builder.build()

    assert [p.index for p in positions] == [0, 1, 2, 3, 4]
    assert [p.coordinate_x_m for p in positions] == [0.0, 1.0, 2.0, 3.0, 4.0]


@pytest.mark.parametrize('builder_name', ['cartesian', 'from_file'])
def test_copy_preserves_trim_parameters(builder_name: str) -> None:
    """copy() iterates parameters() generically, so the new counts should ride
    along without any per-subclass change."""
    settings = _make_settings()

    if builder_name == 'cartesian':
        builder = _make_cartesian_line_builder(settings, 5)
    else:
        reader = _StubPositionFileReader(_make_line(5))
        builder = FromFileProbePositionsBuilder(_make_rng(), settings, reader)  # type: ignore[assignment]

    builder.num_discard_at_start.set_value(2)
    builder.num_discard_at_end.set_value(3)

    duplicate = builder.copy()

    assert duplicate.num_discard_at_start.get_value() == 2
    assert duplicate.num_discard_at_end.get_value() == 3


def test_streaming_builder_is_instantiable_and_copyable() -> None:
    """StreamingScanBuilder never implemented the abstract copy(), so it could
    not be instantiated at all."""
    settings = _make_settings()
    builder = StreamingScanBuilder(_make_rng(), settings, _make_line(5))
    builder.num_discard_at_start.set_value(1)

    duplicate = builder.copy()

    assert duplicate.num_discard_at_start.get_value() == 1
    assert [p.index for p in builder.build()] == [1, 2, 3, 4]
