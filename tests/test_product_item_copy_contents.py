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


def _make_metadata_item(name: str) -> MagicMock:
    """A metadata mock whose name parameter stores what is written to it.

    ``assign`` overwrites the name from the incoming metadata, the way the real
    MetadataRepositoryItem does, so the test can tell whose name survives the copy.
    """
    metadata_item = MagicMock()
    holder = [name]

    def get_value() -> str:
        return holder[0]

    def set_value(value: str) -> None:
        holder[0] = value

    metadata_item.name.get_value.side_effect = get_value
    metadata_item.name.set_value.side_effect = set_value
    metadata_item.get_metadata.return_value.name = name
    metadata_item.assign.side_effect = lambda metadata: set_value(metadata.name)
    return metadata_item


def _make_item(
    *, losses: list[LossValue], dataset: MagicMock | None, name: str = 'item'
) -> tuple[ProductRepositoryItem, _Mocks]:
    mocks = _Mocks(
        parent=MagicMock(),
        metadata_item=_make_metadata_item(name),
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
    # The stub holds the name the caller asked for; the source was built while the stub
    # already occupied it, so it carries the collision-avoidance counter.
    source, source_mocks = _make_item(
        losses=[LossValue(epoch=1, value=0.5)], dataset=MagicMock(), name='scan-1'
    )
    stub, stub_mocks = _make_item(losses=[], dataset=None, name='scan')
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


def test_copy_contents_from_keeps_the_stub_name() -> None:
    source, stub, _source_mocks, _stub_mocks = _make_source_and_stub()

    stub.copy_contents_from(source)

    # The source's 'scan-1' exists only because the stub was holding 'scan' while the
    # source was built; letting it win would grow the name on every queued insert.
    assert stub.get_name() == 'scan'
