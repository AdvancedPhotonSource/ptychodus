"""The product API's parallax defocus action, run inline on the parallax fixtures.

Behaviors verified:
  - a resolved parallax sets the product's focus position to the negated defocus
  - an unresolved one, at focus, leaves the focus position alone
  - near-field data is refused without making an estimate
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import pytest

from ptychodus.api.parallax import ParallaxEstimate
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import ProductMetadata
from ptychodus.model.product import ProductAPI
from ptychodus.model.task_manager import BackgroundTask, ForegroundTask, TaskManager

from parallax_scan_fixtures import (
    METADATA,
    OBJECT_GEOMETRY,
    PROBE_GEOMETRY,
    make_positions,
    simulate_scan,
)


class _InlineTaskManager(TaskManager):
    """Runs a background task and the foreground task it returns before returning."""

    def put_background_task(self, task: BackgroundTask) -> None:
        foreground_task = task()

        if foreground_task is not None:
            foreground_task()

    def put_foreground_task(self, task: ForegroundTask) -> None:
        task()


def _estimate(
    defocus_m: float, metadata: ProductMetadata = METADATA
) -> tuple[MagicMock, list[tuple[ParallaxEstimate | None, bool]]]:
    item = MagicMock()
    item.get_dataset.return_value.get_assembled_data.return_value = simulate_scan(defocus_m)
    item.get_metadata_item.return_value.get_metadata.return_value = metadata
    item.get_probe_positions_item.return_value.get_probe_positions.return_value = (
        ProbePositionSequence(make_positions())
    )
    item.get_geometry.return_value.get_object_geometry.return_value = OBJECT_GEOMETRY
    item.get_geometry.return_value.get_probe_geometry.return_value = PROBE_GEOMETRY
    repository = MagicMock()
    repository.__getitem__.return_value = item
    api = ProductAPI(
        MagicMock(),
        repository,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        _InlineTaskManager(),
        MagicMock(),
    )
    results: list[tuple[ParallaxEstimate | None, bool]] = []

    # Bins of two pixels: the fixture's bright field is only eight pixels in radius.
    api.estimate_defocus_from_parallax(
        0, bin_size_px=2, on_finished=lambda estimate, applied: results.append((estimate, applied))
    )
    return item, results


@pytest.mark.parametrize('defocus_m', [2.0e-3, -2.0e-3])
def test_a_resolved_parallax_sets_the_focus_position(defocus_m: float) -> None:
    item, results = _estimate(defocus_m)

    [(estimate, applied)] = results
    assert applied
    assert estimate is not None
    assert estimate.defocus_distance_m == pytest.approx(defocus_m, rel=0.06)
    focus = item.get_metadata_item.return_value.focus_object_distance_m
    focus.set_value.assert_called_once_with(-estimate.defocus_distance_m)


def test_an_unresolved_parallax_leaves_the_focus_position_alone() -> None:
    item, results = _estimate(0.0)

    [(estimate, applied)] = results
    assert estimate is not None
    assert not applied
    item.get_metadata_item.return_value.focus_object_distance_m.set_value.assert_not_called()


def test_near_field_data_is_refused() -> None:
    item, results = _estimate(2.0e-3, replace(METADATA, far_field=False))

    assert results == [(None, False)]
    item.get_metadata_item.return_value.focus_object_distance_m.set_value.assert_not_called()
