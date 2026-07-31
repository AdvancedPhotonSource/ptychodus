"""Unit tests for DiffractionDatasetRepository."""

from __future__ import annotations

from unittest.mock import MagicMock

from ptychodus.model.diffraction.repository import (
    DiffractionDatasetRepository,
    DiffractionDatasetRepositoryObserver,
)


def _make_dataset(name: str) -> MagicMock:
    dataset = MagicMock()
    dataset.get_name.return_value = name
    return dataset


class _RecordingObserver(DiffractionDatasetRepositoryObserver):
    def __init__(self) -> None:
        self.inserted: list[tuple[int, str]] = []
        self.removed: list[tuple[int, str]] = []

    def handle_dataset_inserted(self, index, dataset) -> None:  # noqa: ANN001
        self.inserted.append((index, dataset.get_name()))

    def handle_dataset_removed(self, index, dataset) -> None:  # noqa: ANN001
        self.removed.append((index, dataset.get_name()))


def test_empty_repository_is_empty() -> None:
    repo = DiffractionDatasetRepository()
    assert len(repo) == 0


def test_insert_dataset_appends_and_returns_index() -> None:
    repo = DiffractionDatasetRepository()
    a = _make_dataset('a')
    b = _make_dataset('b')

    assert repo.insert_dataset(a) == 0
    assert repo.insert_dataset(b) == 1
    assert len(repo) == 2
    assert repo[0] is a
    assert repo[1] is b


def test_insert_notifies_observers() -> None:
    repo = DiffractionDatasetRepository()
    observer = _RecordingObserver()
    repo.add_observer(observer)

    a = _make_dataset('a')
    b = _make_dataset('b')
    repo.insert_dataset(a)
    repo.insert_dataset(b)

    assert observer.inserted == [(0, 'a'), (1, 'b')]


def test_remove_dataset_pops_and_calls_clear() -> None:
    repo = DiffractionDatasetRepository()
    a = _make_dataset('a')
    b = _make_dataset('b')
    repo.insert_dataset(a)
    repo.insert_dataset(b)

    repo.remove_dataset(0)

    assert len(repo) == 1
    assert repo[0] is b
    a.clear.assert_called_once()


def test_remove_notifies_observers() -> None:
    repo = DiffractionDatasetRepository()
    observer = _RecordingObserver()
    repo.add_observer(observer)

    a = _make_dataset('a')
    repo.insert_dataset(a)
    repo.remove_dataset(0)

    assert observer.removed == [(0, 'a')]


def test_remove_out_of_range_is_noop() -> None:
    repo = DiffractionDatasetRepository()
    repo.insert_dataset(_make_dataset('a'))
    repo.remove_dataset(5)
    assert len(repo) == 1


def test_clear_removes_all_and_fires_per_remove_events() -> None:
    repo = DiffractionDatasetRepository()
    observer = _RecordingObserver()
    repo.add_observer(observer)

    repo.insert_dataset(_make_dataset('a'))
    repo.insert_dataset(_make_dataset('b'))
    repo.insert_dataset(_make_dataset('c'))

    repo.clear()

    assert len(repo) == 0
    # Removed bottom-up so the caller sees stable indexes during iteration.
    assert observer.removed == [(2, 'c'), (1, 'b'), (0, 'a')]


def test_create_unique_name_returns_input_when_free() -> None:
    repo = DiffractionDatasetRepository()
    assert repo.create_unique_name('foo') == 'foo'


def test_create_unique_name_suffixes_collisions() -> None:
    repo = DiffractionDatasetRepository()
    repo.insert_dataset(_make_dataset('foo'))
    repo.insert_dataset(_make_dataset('foo-1'))
    assert repo.create_unique_name('foo') == 'foo-2'


def test_create_unique_name_maps_empty_to_unnamed() -> None:
    repo = DiffractionDatasetRepository()
    assert repo.create_unique_name('') == 'Unnamed'


def test_remove_observer_stops_notifications() -> None:
    repo = DiffractionDatasetRepository()
    observer = _RecordingObserver()
    repo.add_observer(observer)
    repo.remove_observer(observer)

    repo.insert_dataset(_make_dataset('a'))
    assert observer.inserted == []


def test_getitem_slice_returns_sequence() -> None:
    repo = DiffractionDatasetRepository()
    a = _make_dataset('a')
    b = _make_dataset('b')
    repo.insert_dataset(a)
    repo.insert_dataset(b)

    tail = repo[1:]
    assert list(tail) == [b]


def test_create_dataset_without_factory_raises() -> None:
    repo = DiffractionDatasetRepository()
    import pytest

    with pytest.raises(RuntimeError):
        repo.create_dataset('foo')


def test_create_dataset_with_factory_uses_unique_name() -> None:
    called_with: list[str] = []

    def _factory(name: str) -> MagicMock:
        called_with.append(name)
        return _make_dataset(name)

    repo = DiffractionDatasetRepository(factory=_factory)
    repo.insert_dataset(_make_dataset('foo'))
    repo.create_dataset('foo')
    assert called_with == ['foo-1']
