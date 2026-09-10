from __future__ import annotations

import logging

from ptychodus.api.assemble import (
    DiffractionSummary,
    DiffractionTotalCounts,
    compute_dataset_total_counts,
    summarize_dataset,
)
from ptychodus.api.diffraction import BadPixels, CropRegion, DiffractionDataset
from ptychodus.api.preprocess.diffraction import DiffractionPrepPipeline, DiffractionPrepPlan

from ..task_manager import TaskManager
from ..task_monitor import TaskProgressMonitor
from .api import DiffractionAPI
from .prep_pipeline import PrepPipelineBuilder
from .settings import DetectorSettings, DiffractionSettings

__all__ = [
    'DiffractionSummaryService',
    'DiffractionSummaryTaskMonitor',
    'SummarizeBackgroundTask',
    'TotalCountsBackgroundTask',
]

_TotalCountsKey = tuple[int, DiffractionPrepPlan, bool, str, str]

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


class TotalCountsBackgroundTask:
    """Background task adapting :func:`compute_dataset_total_counts` to the task manager.

    Snapshotting matches :class:`SummarizeBackgroundTask`: the prep plan, the
    bad-pixels mask, and the cache key are all resolved by the caller before the
    task is queued, so a settings edit mid-run cannot change what is being
    measured or mislabel the result it publishes.
    """

    def __init__(
        self,
        service: DiffractionSummaryService,
        source: DiffractionDataset,
        pipeline: DiffractionPrepPipeline,
        bad_pixels: BadPixels | None,
        read_region: CropRegion | None,
        cache_key: _TotalCountsKey,
    ) -> None:
        self._service = service
        self._source = source
        self._pipeline = pipeline
        self._bad_pixels = bad_pixels
        self._read_region = read_region
        self._cache_key = cache_key

    def __call__(self) -> None:
        monitor = self._service.task_monitor
        with monitor:
            total_counts = compute_dataset_total_counts(
                self._source,
                self._pipeline,
                bad_pixels=self._bad_pixels,
                read_region=self._read_region,
                on_progress=monitor.update_progress,
                should_stop=lambda: monitor.is_stopping,
            )
            if monitor.is_stopping:
                # A partial pass must not be cached against the full-run key.
                return
            self._service._publish_total_counts(total_counts, self._cache_key)


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

    ``compute_total_counts`` is the second, cheaper pass. The summary's own
    ``total_counts`` are summed over the raw full detector, but the total-counts
    filter compares its bounds against post-crop, post-pipeline counts, so bounds
    derived from the summary can be an order of magnitude too high and empty every
    array. This pass measures the counts the filter actually sees, reading only the
    crop rectangle from disk. Its result is cached against the prep plan that
    produced it, so re-requesting it with unchanged processing settings is free.
    """

    def __init__(
        self,
        task_manager: TaskManager,
        api: DiffractionAPI,
        detector_settings: DetectorSettings,
        diffraction_settings: DiffractionSettings,
    ) -> None:
        self._task_manager = task_manager
        self._api = api
        self._detector_settings = detector_settings
        # Stateless; reads live settings on each get_plan call.
        self._pipeline_builder = PrepPipelineBuilder(diffraction_settings)
        self.task_monitor = DiffractionSummaryTaskMonitor(task_manager)
        self._last_summary: DiffractionSummary | None = None
        self._last_run_id = 0
        self._last_total_counts: DiffractionTotalCounts | None = None
        self._last_total_counts_key: _TotalCountsKey | None = None
        self._last_total_counts_run_id = 0

    def get_last_summary(self) -> DiffractionSummary | None:
        return self._last_summary

    def get_last_run_id(self) -> int:
        """Monotonic counter incremented every time a summary is published.

        Observers compare this against their own last-seen id to detect a fresh
        result without an identity comparison on the summary itself.
        """
        return self._last_run_id

    def get_last_total_counts(self) -> DiffractionTotalCounts | None:
        """Post-crop, post-pipeline per-pattern counts from the last completed pass."""
        return self._last_total_counts

    def get_last_total_counts_run_id(self) -> int:
        """Monotonic counter incremented every time total counts are published."""
        return self._last_total_counts_run_id

    def is_total_counts_stale(self, dataset_index: int) -> bool:
        """True when no cached counts match the current dataset and processing settings.

        Cheap enough to call from a settings observer: it resolves the prep plan
        from live settings but touches neither the repository's file handles nor
        the bad-pixels file.
        """
        if self._last_total_counts is None:
            return True

        try:
            key = self._build_total_counts_key(dataset_index)
        except ValueError:
            # An incomplete plan (e.g. a bin size that does not divide the crop)
            # cannot match anything that was measured.
            return True

        return key != self._last_total_counts_key

    def _build_total_counts_key(self, dataset_index: int) -> _TotalCountsKey:
        """Identify a counts pass by everything that changes the numbers it yields.

        Raises ``ValueError`` when the live settings do not describe a usable prep
        plan. The bad-pixels mask enters the key by its settings rather than its
        contents so the key stays free of file I/O.
        """
        repository = self._api.get_repository()

        if dataset_index < 0 or dataset_index >= len(repository):
            raise ValueError(f'No pending dataset at index {dataset_index}.')

        detector_extent = repository[dataset_index].get_source().get_metadata().detector_extent
        plan = self._pipeline_builder.get_plan(detector_extent)

        return (
            dataset_index,
            plan,
            self._detector_settings.bad_pixels_enabled.get_value(),
            str(self._detector_settings.bad_pixels_file_path.get_value()),
            self._detector_settings.bad_pixels_file_type.get_value(),
        )

    def compute_total_counts(self, dataset_index: int, *, force: bool = False) -> bool:
        """Kick off a background counts pass for the dataset at ``dataset_index``.

        Returns whether a pass was dispatched; ``False`` means the cached counts
        already match the current settings and nothing needed to run. Pass
        ``force`` to re-measure regardless. Raises the same errors as
        :meth:`compute`, synchronously, on the calling thread.
        """
        key = self._build_total_counts_key(dataset_index)

        if not force and self._last_total_counts is not None and key == self._last_total_counts_key:
            return False

        repository = self._api.get_repository()
        source = repository[dataset_index].get_source()

        if len(source) == 0:
            raise ValueError('Pending dataset has no arrays to measure.')

        bad_pixels = self._load_bad_pixels()
        plan = key[1]
        self._task_manager.put_background_task(
            TotalCountsBackgroundTask(
                self, source, plan.pipeline, bad_pixels, plan.read_region, key
            )
        )
        return True

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

        bad_pixels = self._load_bad_pixels()
        self._task_manager.put_background_task(SummarizeBackgroundTask(self, source, bad_pixels))

    def _load_bad_pixels(self) -> BadPixels | None:
        """Load the configured bad-pixels mask, or None when the override is off."""
        if not self._detector_settings.bad_pixels_enabled.get_value():
            return None

        return self._api.load_bad_pixels(
            self._detector_settings.bad_pixels_file_path.get_value(),
            self._detector_settings.bad_pixels_file_type.get_value(),
        )

    def _publish(self, summary: DiffractionSummary) -> None:
        """Called on the background thread inside ``SummarizeBackgroundTask`` to
        publish a winning summary. The monitor's ``__exit__`` fires immediately
        after (on the same thread) and its foreground-queued notification is
        what wakes observers, so the write is visible to them by the time they
        read ``get_last_summary()``.
        """
        self._last_summary = summary
        self._last_run_id += 1

    def _publish_total_counts(
        self, total_counts: DiffractionTotalCounts, cache_key: _TotalCountsKey
    ) -> None:
        """Publish a winning counts pass and the key it was measured under.

        Same threading contract as :meth:`_publish`: called on the background
        thread, made visible to observers by the monitor's foreground-queued
        notification.
        """
        self._last_total_counts = total_counts
        self._last_total_counts_key = cache_key
        self._last_total_counts_run_id += 1
