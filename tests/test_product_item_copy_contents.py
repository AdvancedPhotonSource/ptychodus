"""Tests for ProductRepositoryItem.copy_contents_from ordering.

Guards against the regression where the stub's probe/object subgroups were
assigned before the dataset was bound: their _rebuild() saw an invalid pixel
geometry (detector_extent still None) and silently no-op'd, leaving the
finalized product with empty probe/object arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock

from ptychodus.api.product import LossValue
from ptychodus.model.product.item import ProductRepositoryItem, ProductState


@dataclass(frozen=True)
class _Mocks:
    """The mocks standing in for an item's collaborators.

    Held separately from the item because reading them back off the item -- e.g.
    ``item._probe_item.assign_item`` -- resolves against the real declared types,
    which are bound methods rather than mocks.
    """

    parent: MagicMock
    metadata_item: MagicMock
    probe_positions_item: MagicMock
    probe_item: MagicMock
    object_item: MagicMock
    geometry: MagicMock


def _make_item(
    *, losses: list[LossValue], dataset: MagicMock | None
) -> tuple[ProductRepositoryItem, _Mocks]:
    mocks = _Mocks(
        parent=MagicMock(),
        metadata_item=MagicMock(),
        probe_positions_item=MagicMock(),
        probe_item=MagicMock(),
        object_item=MagicMock(),
        geometry=MagicMock(),
    )

    # Bypass __init__ so we don't have to satisfy every dependency of ProductGeometry.
    item = ProductRepositoryItem.__new__(ProductRepositoryItem)
    item._parent = mocks.parent
    item._metadata_item = mocks.metadata_item
    item._probe_positions_item = mocks.probe_positions_item
    item._probe_item = mocks.probe_item
    item._object_item = mocks.object_item
    item._geometry = mocks.geometry
    item._losses = losses
    item._dataset = dataset
    item._state = ProductState.PENDING

    return item, mocks


def _make_source_and_stub() -> tuple[ProductRepositoryItem, ProductRepositoryItem, _Mocks, _Mocks]:
    source, source_mocks = _make_item(losses=[LossValue(epoch=1, value=0.5)], dataset=MagicMock())
    stub, stub_mocks = _make_item(losses=[], dataset=None)
    return source, stub, source_mocks, stub_mocks


def test_copy_contents_from_binds_dataset_before_assigning_probe_and_object() -> None:
    source, stub, _source_mocks, stub_mocks = _make_source_and_stub()

    manager = MagicMock()
    manager.attach_mock(stub_mocks.geometry.set_detector_extent, 'set_detector_extent')
    manager.attach_mock(stub_mocks.probe_item.assign_item, 'probe_assign')
    manager.attach_mock(stub_mocks.object_item.assign_item, 'object_assign')
    manager.attach_mock(stub_mocks.probe_positions_item.assign_item, 'positions_assign')

    stub.copy_contents_from(source)

    call_names = [call[0] for call in manager.mock_calls]
    # set_detector_extent must precede both probe and object assign_item.
    assert call_names.index('set_detector_extent') < call_names.index('probe_assign')
    assert call_names.index('set_detector_extent') < call_names.index('object_assign')


def test_copy_contents_from_copies_all_state() -> None:
    source, stub, source_mocks, stub_mocks = _make_source_and_stub()

    stub.copy_contents_from(source)

    stub_mocks.metadata_item.assign.assert_called_once_with(
        source_mocks.metadata_item.get_metadata.return_value
    )
    stub_mocks.probe_positions_item.assign_item.assert_called_once_with(
        source._probe_positions_item
    )
    stub_mocks.probe_item.assign_item.assert_called_once_with(source._probe_item)
    stub_mocks.object_item.assign_item.assert_called_once_with(source._object_item)
    assert stub._losses == source._losses
    assert stub._dataset is source._dataset
    stub_mocks.parent.handle_losses_changed.assert_called_once_with(stub)
