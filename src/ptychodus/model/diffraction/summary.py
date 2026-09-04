from __future__ import annotations

import logging

from ptychodus.api.assemble import DiffractionSummary, summarize_dataset
from ptychodus.api.diffraction import BadPixels, DiffractionDataset

from ..task_manager import TaskManager
from ..task_monitor import TaskProgressMonitor
from .api import DiffractionAPI
from .settings import DetectorSettings

__all__ = [
    'DiffractionSummaryService',
    'DiffractionSummaryTaskMonitor',
    'SummarizeBackgroundTask',
]

logger = logging.getLogger(__name__)


class DiffractionSummaryTaskMonitor(TaskProgressMonitor):
    """TaskProgressMonitor for the summarize_dataset background compute.

    Marker subclass so :class:`~ptychodus.controller.task_status.TaskStatusController`
    and the wizard's summary controller bind to the right monitor without
    depending on the underlying :class:`TaskManager`.
    """


class SummarizeBackgroundTask:
    """Background task adapting :func:`summarize_dataset` to the task manager.

    Everything the summarize needs is snapshotted by the caller before this task
    is queued: a settings change while the compute runs cannot change the
    bad-pixels mask under it, and the source dataset reference is held for the
    lifetime of the task. Only the winning result is published — a cancelled or
    failed run leaves the service's previous summary untouched.
    """

    def __init__(
        self,
        service: DiffractionSummaryService,
        source: DiffractionDataset,
        bad_pixels: BadPixels | None,
    ) -> None:
        self._service = service
        self._source = source
        self._bad_pixels = bad_pixels

    def __call__(self) -> None:
        monitor = self._service.task_monitor
        with monitor:
            summary = summarize_dataset(
                self._source,
                bad_pixels=self._bad_pixels,
                on_progress=monitor.update_progress,
                should_stop=lambda: monitor.is_stopping,
            )
            if monitor.is_stopping:
                # Preserve any prior summary; a cancelled run publishes nothing.
                return
            self._service._publish(summary)


class DiffractionSummaryService:
    """Background compute service wrapping :func:`summarize_dataset`.

    Owns the :class:`DiffractionSummaryTaskMonitor` that controllers observe;
    holds the last completed :class:`DiffractionSummary` for observers to read
    once the monitor transitions to ``is_processing = False``.

    ``compute`` resolves the source dataset and bad-pixels mask on the caller
    (foreground) thread so file-read errors surface synchronously, then
    dispatches the summarize itself as a background task. ``_task_manager`` is
    private: controllers reach this service via ``DiffractionCore.summary_service``
    and never see the task manager.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        api: DiffractionAPI,
        detector_settings: DetectorSettings,
    ) -> None:
        self._task_manager = task_manager
        self._api = api
        self._detector_settings = detector_settings
        self.task_monitor = DiffractionSummaryTaskMonitor(task_manager)
        self._last_summary: DiffractionSummary | None = None
        self._last_run_id = 0

    def get_last_summary(self) -> DiffractionSummary | None:
        return self._last_summary

    def get_last_run_id(self) -> int:
        """Monotonic counter incremented every time a summary is published.

        Observers compare this against their own last-seen id to detect a fresh
        result without an identity comparison on the summary itself.
        """
        return self._last_run_id

    def stop(self) -> None:
        self.task_monitor.stop_processing()

    def compute(self, dataset_index: int) -> None:
        """Kick off a background summarize for the dataset at ``dataset_index``.

        Raises ``ValueError`` synchronously when the index is out of range or
        the dataset has no arrays; the bad-pixels load may raise ``FileNotFoundError``
        or ``RuntimeError`` from :meth:`DiffractionAPI.load_bad_pixels`. Callers
        should surface these to the user directly.
        """
        repository = self._api.get_repository()

        if dataset_index < 0 or dataset_index >= len(repository):
            raise ValueError(f'No pending dataset at index {dataset_index}.')

        source = repository[dataset_index].get_source()

        if len(source) == 0:
            raise ValueError('Pending dataset has no arrays to summarize.')

        if self._detector_settings.bad_pixels_enabled.get_value():
            bad_pixels: BadPixels | None = self._api.load_bad_pixels(
                self._detector_settings.bad_pixels_file_path.get_value(),
                self._detector_settings.bad_pixels_file_type.get_value(),
            )
        else:
            bad_pixels = None

        self._task_manager.put_background_task(SummarizeBackgroundTask(self, source, bad_pixels))

    def _publish(self, summary: DiffractionSummary) -> None:
        """Called on the background thread inside ``SummarizeBackgroundTask`` to
        publish a winning summary. The monitor's ``__exit__`` fires immediately
        after (on the same thread) and its foreground-queued notification is
        what wakes observers, so the write is visible to them by the time they
        read ``get_last_summary()``.
        """
        self._last_summary = summary
        self._last_run_id += 1
