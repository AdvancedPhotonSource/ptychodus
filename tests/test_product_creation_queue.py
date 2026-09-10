"""Unit tests for the async product-creation queue in ProductAPI._insert_via_queue."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from ptychodus.model.product.api import ProductAPI
from ptychodus.model.product.monitor import ProductTaskMonitor
from ptychodus.model.product.item import ProductState


class _StubTaskManager:
    """Minimal TaskManager stand-in: records enqueued tasks and exposes the standard
    is_stopping / WAIT_TIME_S attributes ProductAPI reads.

    Foreground tasks are recorded rather than run: ProductTaskMonitor posts observer
    notifications through this on every enter/exit of the queued finalize."""

    is_stopping = False
    WAIT_TIME_S = 0.01

    def __init__(self) -> None:
        self.background_tasks: list = []
        self.foreground_tasks: list = []

    def put_background_task(self, task) -> None:  # noqa: ANN001
        self.background_tasks.append(task)

    def put_foreground_task(self, task) -> None:  # noqa: ANN001
        self.foreground_tasks.append(task)


def _make_api(
    task_manager: _StubTaskManager,
    stub_item: MagicMock,
    real_item: MagicMock,
    monitor: ProductTaskMonitor | None = None,
) -> tuple[ProductAPI, MagicMock, MagicMock]:
    repository = MagicMock()
    inserted: list = []

    def insert_product(item):  # noqa: ANN001
        inserted.append(item)
        return len(inserted) - 1

    repository.insert_product.side_effect = insert_product

    item_factory = MagicMock()
    item_factory.create_pending_stub.return_value = stub_item
    item_factory.create_from_values.return_value = real_item
    item_factory.create_from_product.return_value = real_item
    item_factory.create_from_settings.return_value = real_item

    api = ProductAPI(
        settings=MagicMock(),
        repository=repository,
        item_factory=item_factory,
        file_reader_chooser=MagicMock(),
        file_writer_chooser=MagicMock(),
        task_manager=task_manager,  # type: ignore[arg-type]
        # A real monitor, not a mock: _insert_via_queue uses it as a context manager
        # and reads is_stopping, so the queue path depends on its actual behavior.
        task_monitor=monitor or ProductTaskMonitor(task_manager),  # type: ignore[arg-type]
    )
    return api, repository, item_factory


def _dataset(
    *,
    in_progress: bool,
    error: BaseException | None = None,
    event_set: bool | None = None,
) -> MagicMock:
    dataset = MagicMock()
    dataset.is_load_in_progress.return_value = in_progress
    dataset.get_last_load_error.return_value = error
    event = threading.Event()
    # By default, event is set iff the load is not in progress. Individual tests
    # can pass event_set=True to simulate \"load just finished with an error\".
    should_set = (not in_progress) if event_set is None else event_set
    if should_set:
        event.set()
    dataset.get_last_load_finished_event.return_value = event
    return dataset


def test_sync_path_when_dataset_is_none() -> None:
    tm = _StubTaskManager()
    stub, real = MagicMock(), MagicMock()
    api, repo, factory = _make_api(tm, stub, real)

    index = api.insert_new_product(dataset=None, block=False)

    assert index == 0
    factory.create_from_values.assert_called_once()
    factory.create_pending_stub.assert_not_called()
    repo.insert_product.assert_called_once_with(real)
    assert tm.background_tasks == []


def test_sync_path_when_dataset_already_loaded() -> None:
    tm = _StubTaskManager()
    stub, real = MagicMock(), MagicMock()
    api, repo, factory = _make_api(tm, stub, real)

    dataset = _dataset(in_progress=False)
    index = api.insert_new_product(dataset=dataset, block=False)

    assert index == 0
    factory.create_from_values.assert_called_once()
    factory.create_pending_stub.assert_not_called()
    assert tm.background_tasks == []


def test_enqueues_stub_when_mid_load_and_not_blocking() -> None:
    tm = _StubTaskManager()
    stub, real = MagicMock(), MagicMock()
    api, repo, factory = _make_api(tm, stub, real)

    dataset = _dataset(in_progress=True)
    index = api.insert_new_product(name='p', dataset=dataset, block=False)

    assert index == 0
    factory.create_pending_stub.assert_called_once_with(name='p')
    repo.insert_product.assert_called_once_with(stub)
    assert len(tm.background_tasks) == 1
    factory.create_from_values.assert_not_called()


def test_background_task_finalizes_stub_on_success() -> None:
    tm = _StubTaskManager()
    stub, real = MagicMock(), MagicMock()
    api, repo, factory = _make_api(tm, stub, real)

    dataset = _dataset(in_progress=True)
    api.insert_new_product(dataset=dataset, block=False)

    background_task = tm.background_tasks[0]
    foreground_task = background_task()
    foreground_task()

    factory.create_from_values.assert_called_once()
    stub.copy_contents_from.assert_called_once_with(real)
    stub.set_state.assert_called_once_with(ProductState.READY)


def test_background_task_marks_stub_failed_on_load_error() -> None:
    tm = _StubTaskManager()
    stub, real = MagicMock(), MagicMock()
    api, repo, factory = _make_api(tm, stub, real)

    err = RuntimeError('boom')
    dataset = _dataset(in_progress=True, error=err)
    api.insert_new_product(dataset=dataset, block=False)

    background_task = tm.background_tasks[0]
    foreground_task = background_task()
    foreground_task()

    factory.create_from_values.assert_not_called()
    stub.set_state.assert_called_once_with(ProductState.FAILED)


def test_blocking_call_raises_on_load_error() -> None:
    tm = _StubTaskManager()
    stub, real = MagicMock(), MagicMock()
    api, _repo, _factory = _make_api(tm, stub, real)

    err = RuntimeError('boom')
    # Simulate load that has just finished (event set) with an error stored.
    dataset = _dataset(in_progress=True, error=err, event_set=True)

    with pytest.raises(RuntimeError):
        api.insert_new_product(dataset=dataset, block=True)


def test_blocking_call_returns_index_after_wait() -> None:
    tm = _StubTaskManager()
    stub, real = MagicMock(), MagicMock()
    api, repo, factory = _make_api(tm, stub, real)

    # in_progress=True but event already set (immediate wake), no error.
    dataset = _dataset(in_progress=True, event_set=True)

    index = api.insert_new_product(dataset=dataset, block=True)

    assert index == 0
    factory.create_from_values.assert_called_once()
    repo.insert_product.assert_called_once_with(real)
    assert tm.background_tasks == []


def test_monitor_is_busy_from_enqueue_until_finalize() -> None:
    tm = _StubTaskManager()
    monitor = ProductTaskMonitor(tm)  # type: ignore[arg-type]
    stub, real = MagicMock(), MagicMock()
    api, _repo, _factory = _make_api(tm, stub, real, monitor=monitor)

    assert not monitor.is_processing

    api.insert_new_product(dataset=_dataset(in_progress=True), block=False)

    # Busy from the moment the stub is queued: that is the window Stop must reach.
    assert monitor.is_processing
    assert (monitor.get_progress(), monitor.get_progress_goal()) == (0, 1)

    tm.background_tasks[0]()()

    assert not monitor.is_processing


def test_a_burst_of_queued_products_nests() -> None:
    tm = _StubTaskManager()
    monitor = ProductTaskMonitor(tm)  # type: ignore[arg-type]
    api, _repo, _factory = _make_api(tm, MagicMock(), MagicMock(), monitor=monitor)

    dataset = _dataset(in_progress=True)
    api.insert_new_product(dataset=dataset, block=False)
    api.insert_new_product(dataset=dataset, block=False)

    assert (monitor.get_progress(), monitor.get_progress_goal()) == (0, 2)

    tm.background_tasks[0]()()

    # Still busy: the burst is not drained until every queued product finalizes.
    assert monitor.is_processing
    assert (monitor.get_progress(), monitor.get_progress_goal()) == (1, 2)

    tm.background_tasks[1]()()

    assert not monitor.is_processing


def test_stop_cancels_a_queued_product() -> None:
    tm = _StubTaskManager()
    monitor = ProductTaskMonitor(tm)  # type: ignore[arg-type]
    stub = MagicMock()
    api, _repo, factory = _make_api(tm, stub, MagicMock(), monitor=monitor)

    api.insert_new_product(dataset=_dataset(in_progress=True), block=False)
    monitor.stop_processing()
    tm.background_tasks[0]()()

    stub.set_state.assert_called_with(ProductState.FAILED)
    factory.create_from_values.assert_not_called()
    assert not monitor.is_processing
