"""Diffraction data reaches a product's builders through its geometry provider.

Behaviors verified:
  - the provider answers get_assembled_data from the bound dataset, and binding one
    reports whether it changed without notifying observers
  - binding other data, unbinding, or reloading the bound dataset asks the probe and
    object items to rebuild whatever reads the data; rebinding the same dataset does not
  - the moments of a finished load arriving re-estimate the beam and rebuild the same way
  - the beam estimate sets the transmission from the moments, fills the photon count only
    while it is unset unless asked to replace it, and waits while the moments are pending
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy
import pytest

from ptychodus.api.accumulate import DiffractionMoments
from ptychodus.api.observer import Observable, Observer
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.settings import SettingsRegistry
from ptychodus.model.product.geometry import ProductGeometryProvider
from ptychodus.model.product.item import ProductRepositoryItem, ProductState, _BoundDatasetObserver
from ptychodus.model.product.metadata import MetadataRepositoryItem
from ptychodus.model.product.settings import ProductSettings


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
    item.estimates = 0  # type: ignore[attr-defined]

    def count_estimate(**_: object) -> None:
        item.estimates += 1  # type: ignore[attr-defined]

    item._estimate_incident_beam = count_estimate  # type: ignore[method-assign]
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


def test_the_moments_arriving_reestimate_and_rebuild() -> None:
    item = _make_item()
    item.bind_dataset(MagicMock())
    before = _rebuild_requests(item)
    estimates = item.estimates  # type: ignore[attr-defined]

    _BoundDatasetObserver(item).handle_moments_changed()

    assert _rebuild_requests(item) == before + 1
    assert item.estimates == estimates + 1  # type: ignore[attr-defined]


def _moments(totals: list[float]) -> DiffractionMoments:
    frame = numpy.zeros((4, 4))
    rows = numpy.zeros(len(totals))
    return DiffractionMoments(
        num_patterns=len(totals),
        minimum_frame=frame,
        mean_frame=frame,
        maximum_frame=frame,
        bad_pixels=numpy.zeros((4, 4), dtype=bool),
        indexes=numpy.arange(len(totals)),
        total_counts=numpy.asarray(totals, dtype=numpy.float64),
        center_of_mass_x_px=rows,
        center_of_mass_y_px=rows.copy(),
    )


def _estimating_item(
    moments: DiffractionMoments | None,
    *,
    probe_photon_count: float = 0.0,
    i0: list[float] | None = None,
    i0_photons_per_count: float = 0.0,
) -> ProductRepositoryItem:
    name_factory = MagicMock()
    name_factory.create_unique_name.side_effect = lambda name: name
    metadata_item = MetadataRepositoryItem(
        ProductSettings(SettingsRegistry()),
        name_factory,
        probe_photon_count=probe_photon_count,
        i0_photons_per_count=i0_photons_per_count,
    )
    positions_item = MagicMock()
    positions_item.get_probe_positions.return_value = ProbePositionSequence(
        [
            ProbePosition(index, 0.0, 0.0, None if i0 is None else i0[index])
            for index in range(0 if moments is None else moments.num_patterns)
        ]
    )
    item = ProductRepositoryItem.__new__(ProductRepositoryItem)
    item._metadata_item = metadata_item
    item._probe_positions_item = positions_item
    item._geometry = ProductGeometryProvider(metadata_item, positions_item)
    item._dataset = MagicMock()
    item._dataset.get_diffraction_moments.return_value = moments
    return item


def test_an_unset_photon_count_and_the_transmission_come_from_the_moments() -> None:
    item = _estimating_item(_moments([64.0, 16.0]))

    item._estimate_incident_beam()

    incident = numpy.quantile([64.0, 16.0], 0.99)
    assert item._metadata_item.probe_photon_count.get_value() == pytest.approx(incident)
    assert item._geometry.mean_amplitude_transmission == pytest.approx(numpy.sqrt(40.0 / incident))


def test_a_set_photon_count_survives_unless_replaced() -> None:
    item = _estimating_item(_moments([64.0, 16.0]), probe_photon_count=1.0e6)

    item._estimate_incident_beam()
    assert item._metadata_item.probe_photon_count.get_value() == 1.0e6
    assert item._geometry.mean_amplitude_transmission < 1.0

    item.estimate_probe_photon_count()
    assert item._metadata_item.probe_photon_count.get_value() == pytest.approx(
        numpy.quantile([64.0, 16.0], 0.99)
    )


def test_calibrated_i0_on_the_positions_sets_the_photon_count() -> None:
    item = _estimating_item(_moments([64.0, 32.0]), i0=[16.0, 16.0], i0_photons_per_count=4.0)

    item._estimate_incident_beam()

    assert item._metadata_item.probe_photon_count.get_value() == pytest.approx(64.0)
    assert item._geometry.mean_amplitude_transmission == pytest.approx(numpy.sqrt(0.75))


def test_nothing_changes_while_the_moments_are_pending() -> None:
    item = _estimating_item(None)

    item._estimate_incident_beam()

    assert item._metadata_item.probe_photon_count.get_value() == 0.0
    assert item._geometry.mean_amplitude_transmission == 1.0
