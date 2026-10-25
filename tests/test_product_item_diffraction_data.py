"""Diffraction data reaches a product's builders through its geometry provider.

Behaviors verified:
  - the provider answers get_assembled_data from the bound dataset, and binding one
    reports whether it changed without notifying observers
  - binding other data, unbinding, or reloading the bound dataset asks the probe and
    object items to rebuild whatever reads the data; rebinding the same dataset does not
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ptychodus.api.observer import Observable, Observer
from ptychodus.model.product.geometry import ProductGeometryProvider
from ptychodus.model.product.item import ProductRepositoryItem, ProductState, _BoundDatasetObserver


class _CountingObserver(Observer):
    def __init__(self) -> None:
        self.count = 0

    def _update(self, observable: Observable) -> None:
        self.count += 1


def test_the_provider_answers_from_the_bound_dataset_without_notifying() -> None:
    geometry = ProductGeometryProvider(MagicMock(), MagicMock())
    observer = _CountingObserver()
    geometry.add_observer(observer)
    dataset = MagicMock()

    assert geometry.get_assembled_data() is None
    assert geometry.set_diffraction_dataset(dataset)
    assert geometry.get_assembled_data() is dataset.get_assembled_data.return_value
    assert not geometry.set_diffraction_dataset(dataset)
    assert geometry.set_diffraction_dataset(None)
    assert geometry.get_assembled_data() is None
    assert observer.count == 0


def _make_item() -> ProductRepositoryItem:
    # Bypass __init__: only the dataset binding and its two sub-item calls are exercised.
    item = ProductRepositoryItem.__new__(ProductRepositoryItem)
    item._parent = MagicMock()
    item._probe_item = MagicMock()
    item._object_item = MagicMock()
    item._geometry = ProductGeometryProvider(MagicMock(), MagicMock())
    item._dataset = None
    item._dataset_observer = None
    item._state = ProductState.READY
    item._auto_estimate_probe_photon_count = lambda: None  # type: ignore[method-assign]
    item._estimate_mean_amplitude_transmission = lambda: None  # type: ignore[method-assign]
    return item


def _rebuild_requests(item: ProductRepositoryItem) -> int:
    probe_calls = item._probe_item.handle_diffraction_data_changed.call_count  # type: ignore[attr-defined]
    object_calls = item._object_item.handle_diffraction_data_changed.call_count  # type: ignore[attr-defined]
    assert probe_calls == object_calls
    return probe_calls


def test_binding_other_data_rebuilds_what_reads_it() -> None:
    item = _make_item()
    first, second = MagicMock(), MagicMock()

    item.bind_dataset(first)
    assert _rebuild_requests(item) == 1
    assert item._geometry.get_assembled_data() is first.get_assembled_data.return_value

    item.bind_dataset(first)
    assert _rebuild_requests(item) == 1

    item.bind_dataset(second)
    assert _rebuild_requests(item) == 2
    assert item._geometry.get_assembled_data() is second.get_assembled_data.return_value

    item.unbind_dataset()
    assert _rebuild_requests(item) == 3
    assert item._geometry.get_assembled_data() is None


def test_a_reload_rebuilds_what_reads_the_data() -> None:
    item = _make_item()
    item.bind_dataset(MagicMock())
    before = _rebuild_requests(item)

    _BoundDatasetObserver(item).handle_dataset_reloaded()

    assert _rebuild_requests(item) == before + 1
