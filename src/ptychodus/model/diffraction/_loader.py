from __future__ import annotations
from collections.abc import Callable
import logging
import threading

from ptychodus.api.assemble import AssembledDiffractionData, assemble_dataset
from ptychodus.api.diffraction import BadPixels, CropRegion, DiffractionDataset
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.preprocess.diffraction import DiffractionPrepPipeline

from ..task_monitor import TaskProgressMonitor

logger = logging.getLogger(__name__)


class LoadDiffractionDataset:
    """Background task adapting :func:`assemble_dataset` to the task manager.

    Everything the assembly needs is snapshotted by the caller before this task
    is queued, so a settings change while the load runs cannot leave some arrays
    processed differently from others. The task itself only bridges the
    :class:`TaskProgressMonitor` protocol to the callback parameters and records
    the first per-array failure for :meth:`get_error`.
    """

    def __init__(
        self,
        source: DiffractionDataset,
        out: AssembledDiffractionData,
        pipeline: DiffractionPrepPipeline | None,
        *,
        bad_pixels: BadPixels,
        raw_pixel_geometry: PixelGeometry,
        total_counts_lower_bound: int | None,
        total_counts_upper_bound: int | None,
        read_region: CropRegion | None,
        on_array_assembled: Callable[[int, str, AssembledDiffractionData], None],
        task_monitor: TaskProgressMonitor,
    ) -> None:
        super().__init__()
        self._source = source
        self._out = out
        self._pipeline = pipeline
        self._bad_pixels = bad_pixels
        self._raw_pixel_geometry = raw_pixel_geometry
        self._total_counts_lower_bound = total_counts_lower_bound
        self._total_counts_upper_bound = total_counts_upper_bound
        self._read_region = read_region
        self._on_array_assembled = on_array_assembled
        self._task_monitor = task_monitor
        self._finished_event = threading.Event()
        self._error: BaseException | None = None

    def get_finished_event(self) -> threading.Event:
        return self._finished_event

    def get_error(self) -> BaseException | None:
        """First per-array exception observed during __call__, or None on clean load."""
        return self._error

    def _handle_array_error(self, array_index: int, label: str, error: Exception) -> None:
        if self._error is None:
            self._error = error

    def __call__(self) -> None:
        try:
            with self._task_monitor as monitor:
                assemble_dataset(
                    self._source,
                    self._pipeline,
                    bad_pixels=self._bad_pixels,
                    raw_pixel_geometry=self._raw_pixel_geometry,
                    out=self._out,
                    total_counts_lower_bound=self._total_counts_lower_bound,
                    total_counts_upper_bound=self._total_counts_upper_bound,
                    read_region=self._read_region,
                    on_array_assembled=self._on_array_assembled,
                    on_progress=monitor.update_progress,
                    on_array_error=self._handle_array_error,
                    should_stop=lambda: monitor.is_stopping,
                )
        except BaseException as ex:
            if self._error is None:
                self._error = ex
            raise
        finally:
            self._finished_event.set()
