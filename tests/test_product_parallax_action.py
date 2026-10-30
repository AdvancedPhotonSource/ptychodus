"""The product API's parallax defocus action, run inline on the parallax fixtures.

Behaviors verified:
  - a resolved parallax sets the product's focus position to the negated defocus
  - an unresolved one, at focus, leaves the focus position alone
  - near-field data is refused without making an estimate
  - replacing a stated focus reports a disagreement in sign or beyond ParallaxRtol
  - the settings-driven estimate runs on insert only where EstimateDefocusFromParallax
    is set and the product states no focus of its own
  - WorkflowAPI.estimate_defocus blocks, so a probe generated next sees the focus
  - inserting a product reaches the settings-driven estimate at all
"""

from __future__ import annotations

import logging
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from ptychodus.api.parallax import ParallaxEstimate
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import ProductMetadata
from ptychodus.model.settings import SettingsRegistry
from ptychodus.model.product import ProductAPI
from ptychodus.model.product.settings import ProductSettings
from ptychodus.model.workflow import ConcreteProductWorkflowAPI
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


def _api(
    defocus_m: float,
    metadata: ProductMetadata = METADATA,
    *,
    settings: ProductSettings | None = None,
) -> tuple[ProductAPI, MagicMock]:
    """A ProductAPI over one stub product holding a simulated scan at *defocus_m*."""
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
        _settings() if settings is None else settings,
        repository,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        _InlineTaskManager(),
        MagicMock(),
    )
    return api, item


def _settings() -> ProductSettings:
    """Product settings with the bin size the fixture needs.

    Bins of two pixels: the fixture's bright field is only eight pixels in radius.
    """
    settings = ProductSettings(SettingsRegistry())
    settings.parallax_bin_size_px.set_value(2)
    return settings


def _estimate(
    defocus_m: float, metadata: ProductMetadata = METADATA
) -> tuple[MagicMock, list[tuple[ParallaxEstimate | None, bool]]]:
    api, item = _api(defocus_m, metadata)
    results: list[tuple[ParallaxEstimate | None, bool]] = []

    api.estimate_defocus_from_parallax(
        0, on_finished=lambda estimate, applied: results.append((estimate, applied))
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


# ---------------------------------------------------------------------------
# Replacing a stated focus
# ---------------------------------------------------------------------------


def _stated(defocus_m: float) -> ProductMetadata:
    """Metadata stating a focus, in the opposite frame from the defocus."""
    return replace(METADATA, focus_object_distance_m=-defocus_m)


def test_a_stated_focus_is_replaced_without_comment_when_it_agrees(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger='ptychodus.model.product.api'):
        item, [(estimate, applied)] = _estimate(2.0e-3, _stated(2.0e-3))

    assert applied and estimate is not None
    item.get_metadata_item.return_value.focus_object_distance_m.set_value.assert_called_once_with(
        -estimate.defocus_distance_m
    )
    assert caplog.records == []


@pytest.mark.parametrize(
    ('stated_m', 'reason'), [(-2.0e-3, 'the opposite sign'), (0.2e-3, 'ten times too small')]
)
def test_a_stated_focus_that_disagrees_is_replaced_with_a_warning(
    caplog: pytest.LogCaptureFixture, stated_m: float, reason: str
) -> None:
    with caplog.at_level(logging.WARNING, logger='ptychodus.model.product.api'):
        item, [(estimate, applied)] = _estimate(2.0e-3, _stated(stated_m))

    assert applied and estimate is not None, reason
    item.get_metadata_item.return_value.focus_object_distance_m.set_value.assert_called_once_with(
        -estimate.defocus_distance_m
    )
    assert any('against the' in record.getMessage() for record in caplog.records)


# ---------------------------------------------------------------------------
# The settings-driven estimate
# ---------------------------------------------------------------------------


def _maybe_estimate(
    *,
    enabled: bool,
    metadata: ProductMetadata = METADATA,
    has_dataset: bool = True,
) -> MagicMock:
    """Run _maybe_estimate_defocus and report whether it reached the measurement."""
    settings = _settings()
    settings.estimate_defocus_from_parallax.set_value(enabled)
    api, item = _api(2.0e-3, metadata, settings=settings)

    if not has_dataset:
        item.get_dataset.return_value = None

    with patch.object(ProductAPI, 'estimate_defocus_from_parallax') as estimate:
        api._maybe_estimate_defocus(0, block=True)

    return estimate


def test_the_setting_measures_a_product_that_states_no_focus() -> None:
    estimate = _maybe_estimate(enabled=True)

    estimate.assert_called_once_with(0, block=True)


@pytest.mark.parametrize(
    ('kwargs', 'reason'),
    [
        ({'enabled': False}, 'the setting is off'),
        ({'enabled': True, 'metadata': _stated(5.0e-3)}, 'the product states a focus'),
        ({'enabled': True, 'metadata': replace(METADATA, far_field=False)}, 'near field'),
        ({'enabled': True, 'has_dataset': False}, 'no data is bound'),
    ],
)
def test_the_setting_measures_nothing_where_it_should_not(kwargs: dict, reason: str) -> None:
    assert not _maybe_estimate(**kwargs).called, reason


def test_the_workflow_api_blocks_so_a_probe_built_next_sees_the_focus() -> None:
    product_api = MagicMock()
    workflow_api = ConcreteProductWorkflowAPI(
        product_api,
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        7,
    )

    workflow_api.estimate_defocus()

    product_api.estimate_defocus_from_parallax.assert_called_once_with(7, block=True)


def test_inserting_a_product_reaches_the_settings_driven_estimate() -> None:
    """Guards the wiring rather than the decision: without this call the setting is inert."""
    api, _item = _api(2.0e-3)

    with patch.object(ProductAPI, '_maybe_estimate_defocus') as maybe:
        index = api.insert_new_product('test')

    maybe.assert_called_once_with(index, block=True)
