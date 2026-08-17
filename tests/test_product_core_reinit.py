"""Tests for the settings-reinit wiring in ProductCore.

The bug this guards against: on `-s settings.ini` startup, ProductCore._update
must pass the just-inserted diffraction dataset to insert_product_from_settings,
otherwise the settings-driven product ends up with no associated dataset and
reconstruction is refused with a "no associated diffraction dataset" warning.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from ptychodus.api.observer import Observable
from ptychodus.model.product.core import ProductCore


def _make_core(
    *, repo_datasets: list, product_api: MagicMock, reinit_observable: Observable
) -> ProductCore:
    """Build a ProductCore instance with only the fields _update reads set."""
    core = ProductCore.__new__(ProductCore)
    diffraction_api = MagicMock()
    diffraction_api.get_repository.return_value = repo_datasets
    core._diffraction_api = diffraction_api
    core.product_api = product_api
    core._reinit_observable = reinit_observable
    reinit_observable.add_observer(core)
    return core


def test_update_passes_last_dataset_and_block_false_when_repo_nonempty() -> None:
    reinit = Observable()
    product_api = MagicMock()
    dataset_a = MagicMock(name='dataset_a')
    dataset_b = MagicMock(name='dataset_b')
    _make_core(
        repo_datasets=[dataset_a, dataset_b],
        product_api=product_api,
        reinit_observable=reinit,
    )

    reinit.notify_observers()

    product_api.insert_product_from_settings.assert_called_once_with(dataset=dataset_b, block=False)


def test_update_passes_dataset_none_when_repo_empty() -> None:
    reinit = Observable()
    product_api = MagicMock()
    _make_core(repo_datasets=[], product_api=product_api, reinit_observable=reinit)

    reinit.notify_observers()

    product_api.insert_product_from_settings.assert_called_once_with(dataset=None, block=False)


def test_update_ignores_other_observables() -> None:
    reinit = Observable()
    other = Observable()
    product_api = MagicMock()
    core = _make_core(
        repo_datasets=[MagicMock()], product_api=product_api, reinit_observable=reinit
    )

    core._update(other)

    product_api.insert_product_from_settings.assert_not_called()
