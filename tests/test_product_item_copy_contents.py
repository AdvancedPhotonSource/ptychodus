"""Tests for ProductRepositoryItem.copy_contents_from ordering.

Guards against the regression where the stub's probe/object subgroups were
assigned before the dataset was bound: their _rebuild() saw an invalid pixel
geometry (detector_extent still None) and silently no-op'd, leaving the
finalized product with empty probe/object arrays.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ptychodus.model.product.item import ProductRepositoryItem, ProductState


def _make_source_and_stub() -> tuple[ProductRepositoryItem, ProductRepositoryItem, MagicMock]:
    parent = MagicMock()

    # Bypass __init__ so we don't have to satisfy every dependency of ProductGeometry.
    stub = ProductRepositoryItem.__new__(ProductRepositoryItem)
    stub._parent = parent
    stub._metadata_item = MagicMock()
    stub._probe_positions_item = MagicMock()
    stub._probe_item = MagicMock()
    stub._object_item = MagicMock()
    stub._geometry = MagicMock()
    stub._losses = []
    stub._dataset = None
    stub._state = ProductState.PENDING

    source = ProductRepositoryItem.__new__(ProductRepositoryItem)
    source._parent = MagicMock()
    source._metadata_item = MagicMock()
    source._probe_positions_item = MagicMock()
    source._probe_item = MagicMock()
    source._object_item = MagicMock()
    source._geometry = MagicMock()
    source._losses = ['loss-value']
    source._dataset = MagicMock()

    return source, stub, parent


def test_copy_contents_from_binds_dataset_before_assigning_probe_and_object() -> None:
    source, stub, _parent = _make_source_and_stub()

    manager = MagicMock()
    manager.attach_mock(stub._geometry.set_detector_extent, 'set_detector_extent')
    manager.attach_mock(stub._probe_item.assign_item, 'probe_assign')
    manager.attach_mock(stub._object_item.assign_item, 'object_assign')
    manager.attach_mock(stub._probe_positions_item.assign_item, 'positions_assign')

    stub.copy_contents_from(source)

    call_names = [call[0] for call in manager.mock_calls]
    # set_detector_extent must precede both probe and object assign_item.
    assert call_names.index('set_detector_extent') < call_names.index('probe_assign')
    assert call_names.index('set_detector_extent') < call_names.index('object_assign')


def test_copy_contents_from_copies_all_state() -> None:
    source, stub, parent = _make_source_and_stub()

    stub.copy_contents_from(source)

    stub._metadata_item.assign.assert_called_once_with(
        source._metadata_item.get_metadata.return_value
    )
    stub._probe_positions_item.assign_item.assert_called_once_with(source._probe_positions_item)
    stub._probe_item.assign_item.assert_called_once_with(source._probe_item)
    stub._object_item.assign_item.assert_called_once_with(source._object_item)
    assert stub._losses == source._losses
    assert stub._dataset is source._dataset
    parent.handle_losses_changed.assert_called_once_with(stub)
