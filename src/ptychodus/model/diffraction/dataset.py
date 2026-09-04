from __future__ import annotations
from abc import ABC, abstractmethod
from bisect import bisect
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import IO, overload
import logging
import tempfile
import threading

import numpy

from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.diffraction import (
    BadPixels,
    CropRegion,
    DiffractionArray,
    DiffractionDataset,
    DiffractionDatasetLayoutNode,
    DiffractionIndexes,
    DiffractionMetadata,
    DiffractionPattern,
    DiffractionPatterns,
    SimpleDiffractionDataset,
)
from ptychodus.api.assemble import (
    AssembledDiffractionData,
    allocate_assembled_data,
    compute_array_offsets,
    compute_assembled_patterns_shape,
    preprocess_array,
)
from ptychodus.api.io import load_diffraction_data, save_diffraction_data
from ptychodus.api.preprocess.diffraction import DiffractionPrepPlan

from ..task_manager import TaskManager
from ._loader import LoadDiffractionDataset
from .monitor import DiffractionTaskMonitor
from .prep_pipeline import PrepPipelineBuilder
from .settings import DetectorSettings, DiffractionSettings

logger = logging.getLogger(__name__)


class DiffractionDatasetState(Enum):
    """Load state of an AssembledDiffractionDataset.

    Mirrors ProductState so the three repository items expose the same vocabulary.
    Derived from the loader handles rather than stored, so there is no extra
    invariant to keep in sync.
    """

    READY = 'ready'
    PENDING = 'pending'
    FAILED = 'failed'


class DiffractionDatasetObserver(ABC):
    @abstractmethod
    def handle_array_inserted(self, index: int) -> None:
        pass

    @abstractmethod
    def handle_array_changed(self, index: int) -> None:
        pass

    @abstractmethod
    def handle_dataset_reloaded(self) -> None:
        pass


class AssembledDiffractionArray(DiffractionArray):
    def __init__(
        self,
        array_index: int,
        label: str,
        data: AssembledDiffractionData,
    ) -> None:
        super().__init__()
        self._array_index = array_index
        self._label = label
        self._data = data
        self._total_counts = data.get_total_counts()
        self._mean_pattern = data.get_mean_pattern()

    @classmethod
    def create_null(cls) -> AssembledDiffractionArray:
        data = AssembledDiffractionData.create_null()
        return cls(0, 'null', data)

    @property
    def array_index(self) -> int:
        return self._array_index

    def get_label(self) -> str:
        return self._label

    def get_indexes(self) -> DiffractionIndexes:
        return self._data.get_indexes()

    def get_patterns(self, *, read_region: CropRegion | None = None) -> DiffractionPatterns:
        patterns = self._data.get_patterns()
        if read_region is None:
            return patterns
        return read_region.apply_to(patterns)

    def get_pattern(self, index: int) -> DiffractionPattern:
        return self._data.get_pattern(index)

    def get_total_counts(self, index: int) -> int:
        return self._total_counts[index]

    def get_mean_total_counts(self) -> float:
        return numpy.mean(self._total_counts).item()

    def get_max_total_counts(self) -> int:
        return self._total_counts.max().item()

    def get_mean_pattern(self) -> DiffractionPattern:
        return self._mean_pattern


class AssembledDiffractionDataset(DiffractionDataset):
    def __init__(
        self,
        settings: DiffractionSettings,
        detector_settings: DetectorSettings,
        task_manager: TaskManager,
        task_monitor: DiffractionTaskMonitor,
        *,
        name: str = 'default',
    ) -> None:
        super().__init__()
        self._name = name
        self._settings = settings
        self._pipeline_builder = PrepPipelineBuilder(settings)
        self._detector_settings = detector_settings
        self._task_manager = task_manager
        self._task_monitor = task_monitor
        self._observer_list: list[DiffractionDatasetObserver] = []

        self._dataset = SimpleDiffractionDataset.create_null()
        self._data = AssembledDiffractionData.create_null()
        self._array_list: list[AssembledDiffractionArray] = list()
        self._array_counter = 0
        # Retained after dispatch so callers can detect \"still loading\" and read
        # any error the loader stored (see is_load_in_progress / get_last_load_error).
        self._last_array_loader: LoadDiffractionDataset | None = None
        self._scratch_tempfile: IO[bytes] | None = None
        # Source dataset queued by reload() and consumed by load_all_arrays().
        # self._dataset drops the array list, so this is the only handle on the
        # lazy arrays; cleared once dispatched, since the task then owns it.
        self._source: DiffractionDataset | None = None

        # Raw (pre-processing) bad-pixel mask; starts as an empty (0, 0) placeholder
        # and is always overwritten by reload() or by load_all_arrays() before any
        # array is processed, so the shape here only matters when nothing is loaded.
        self._bad_pixels = self._create_default_bad_pixels()

    def _create_default_bad_pixels(self) -> BadPixels:
        extent = self._dataset.get_metadata().detector_extent
        return numpy.zeros((extent.height_px, extent.width_px), dtype=numpy.bool_)

    def get_name(self) -> str:
        return self._name

    def set_name(self, name: str) -> None:
        """Set the dataset's display name. Callers must ensure uniqueness themselves
        (typically by routing the candidate through DiffractionDatasetRepository.create_unique_name).
        """
        self._name = name

    def set_bad_pixels(self, bad_pixels: BadPixels) -> None:
        if bad_pixels.ndim != 2:
            raise ValueError(f'Bad pixels array must be 2D, got {bad_pixels.ndim}D.')

        extent = self._dataset.get_metadata().detector_extent

        if bad_pixels.shape != extent.get_shape():
            raise ValueError(
                f'Bad pixels shape {bad_pixels.shape} does not match '
                f'loaded detector extent {extent.get_shape()}.'
            )

        self._bad_pixels = bad_pixels

    def reset_bad_pixels(self) -> None:
        self._bad_pixels = self._create_default_bad_pixels()

    def add_observer(self, observer: DiffractionDatasetObserver) -> None:
        if observer not in self._observer_list:
            self._observer_list.append(observer)

    def remove_observer(self, observer: DiffractionDatasetObserver) -> None:
        try:
            self._observer_list.remove(observer)
        except ValueError:
            pass

    def get_metadata(self) -> DiffractionMetadata:
        return self._dataset.get_metadata()

    def get_layout(self) -> DiffractionDatasetLayoutNode:
        return self._dataset.get_layout()

    def get_raw_pixel_geometry(self) -> PixelGeometry:
        """Resolve the raw (pre-processing) detector pixel geometry for this dataset.

        Priority: metadata > global DetectorSettings fallback. Users who need to
        change the value edit DetectorSettings from the wizard's metadata page
        before the dataset is loaded.
        """
        metadata_geometry = self._dataset.get_metadata().detector_pixel_geometry
        if metadata_geometry is not None:
            return metadata_geometry

        return PixelGeometry(
            width_m=self._detector_settings.pixel_width_m.get_value(),
            height_m=self._detector_settings.pixel_height_m.get_value(),
        )

    def get_processed_pixel_geometry(self) -> PixelGeometry:
        """Return the pixel geometry that matches the assembled patterns.

        Delegates to the AssembledDiffractionData snapshot, which is immutable
        after construction and authoritative for both load-from-raw
        (allocate_assembled_data folds the pipeline through raw geometry at
        buffer-allocation time) and import-from-exported (the processed value is
        read back from the file's DETECTOR_PIXEL_WIDTH/HEIGHT attributes).
        Consulting live pipeline settings here would double-apply binning on
        re-import.
        """
        return self._data.get_pixel_geometry()

    def get_processed_image_extent(self) -> ImageExtent:
        """Return the image extent that matches the assembled patterns.

        Derived from the stored pattern shape rather than metadata + live pipeline,
        for the same reason as get_processed_pixel_geometry.
        """
        return self._data.get_image_extent()

    def get_bad_pixels(self) -> BadPixels:
        return self._bad_pixels

    def get_assembled_data(self) -> AssembledDiffractionData:
        return self._data

    def get_source(self) -> DiffractionDataset:
        """Return the raw source dataset queued for load, or the currently-loaded one.

        `reload()` stores the source on the dataset until `load_all_arrays()`
        dispatches; after dispatch the arrays live on `_dataset`. Callers that
        want to summarize the pre-processing patterns (for example, the wizard's
        Summary panel) can iterate this without waiting for the load pipeline.
        """
        return self._source if self._source is not None else self._dataset

    def get_nbytes(self) -> int:
        return self._data.nbytes

    def is_load_in_progress(self) -> bool:
        """True while a load task is queued but has not finished.

        Note: this does not track per-array append_array() streams, which are
        used only by the pvapy streaming path; products are created there only
        after the stream stops, so the streaming case doesn't rely on this.
        """
        if self._source is not None:
            return True

        loader = self._last_array_loader
        return loader is not None and not loader.get_finished_event().is_set()

    def get_state(self) -> DiffractionDatasetState:
        if self.is_load_in_progress():
            return DiffractionDatasetState.PENDING

        if self.get_last_load_error() is not None:
            return DiffractionDatasetState.FAILED

        return DiffractionDatasetState.READY

    def is_pending(self) -> bool:
        return self.get_state() is DiffractionDatasetState.PENDING

    def is_failed(self) -> bool:
        return self.get_state() is DiffractionDatasetState.FAILED

    def get_last_load_error(self) -> BaseException | None:
        """First exception raised by the most recent load run, or None."""
        loader = self._last_array_loader
        return loader.get_error() if loader is not None else None

    def get_last_load_finished_event(self) -> threading.Event | None:
        """The finished_event of the most recent load task, if any."""
        loader = self._last_array_loader
        return loader.get_finished_event() if loader is not None else None

    def get_mean_pattern(self) -> DiffractionPattern | None:
        if not self._array_list:
            return None
        weights = numpy.array(
            [array.get_num_patterns() for array in self._array_list], dtype=numpy.float64
        )
        means = numpy.stack([array.get_mean_pattern() for array in self._array_list])
        return numpy.average(means, axis=0, weights=weights)

    @overload
    def __getitem__(self, index: int) -> AssembledDiffractionArray: ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[AssembledDiffractionArray]: ...

    def __getitem__(
        self, index: int | slice
    ) -> AssembledDiffractionArray | Sequence[AssembledDiffractionArray]:
        return self._array_list[index]

    def __len__(self) -> int:
        return len(self._array_list)

    def _get_total_counts_bounds(self) -> tuple[int | None, int | None]:
        """Snapshot the enabled total-counts bounds from settings."""
        lower = (
            self._settings.total_counts_lower_bound.get_value()
            if self._settings.total_counts_lower_bound_enabled.get_value()
            else None
        )
        upper = (
            self._settings.total_counts_upper_bound.get_value()
            if self._settings.total_counts_upper_bound_enabled.get_value()
            else None
        )
        return lower, upper

    def _resolve_prep_plan(
        self,
        detector_extent: ImageExtent,
        *,
        process_patterns: bool,
        log_level: int = logging.DEBUG,
    ) -> DiffractionPrepPlan | None:
        """Resolve the prep plan and report the region patterns will be read from.

        A mis-centered or off-detector crop silently starves the reconstruction of
        signal, so the resolved region is always reported. load_all_arrays raises
        log_level to INFO because it resolves the plan once per dataset; the
        streaming path in _append_array resolves it once per incoming array and
        leaves the DEBUG default, so a live feed does not emit a line per frame.
        """
        if not process_patterns:
            return None

        plan = self._pipeline_builder.get_plan(detector_extent)
        read_region = plan.read_region
        region_text = (
            'the full frame'
            if read_region is None
            else f'x={read_region.x_range} y={read_region.y_range}'
        )
        logger.log(
            log_level,
            f'Detector is {detector_extent.width_px}x{detector_extent.height_px};'
            f' reading {region_text}.',
        )
        return plan

    def _append_array(self, array: DiffractionArray, *, process_patterns: bool) -> None:
        """Preprocess one array and scatter it into the current buffer.

        Unlike load_all_arrays this reads settings at call time, because the
        streaming path wants each appended frame to reflect the live
        configuration rather than a snapshot taken when the dataset was opened.
        """
        array_index = self._array_counter
        self._array_counter += 1

        metadata = self._dataset.get_metadata()
        plan = self._resolve_prep_plan(metadata.detector_extent, process_patterns=process_patterns)
        read_region = plan.read_region if plan is not None else None
        pipeline = plan.pipeline if plan is not None else None
        raw_bad_pixels = (
            self._bad_pixels if read_region is None else read_region.apply_to(self._bad_pixels)
        )
        processed_bad_pixels = (
            raw_bad_pixels if pipeline is None else pipeline.apply_to_mask(raw_bad_pixels)
        )
        lower, upper = self._get_total_counts_bounds()
        label = array.get_label()

        try:
            data = preprocess_array(
                array,
                pipeline,
                raw_bad_pixels=raw_bad_pixels,
                processed_bad_pixels=processed_bad_pixels,
                raw_pixel_geometry=self.get_raw_pixel_geometry(),
                total_counts_lower_bound=lower,
                total_counts_upper_bound=upper,
                read_region=read_region,
            )
        except FileNotFoundError:
            logger.warning(f'File not found for "{label}"!')
            return

        offset = compute_array_offsets(metadata)[array_index]
        self._on_array_assembled(array_index, label, self._data.assemble(data, offset))

    def append_array(self, array: DiffractionArray, *, process_patterns: bool = True) -> None:
        self._task_manager.put_background_task(
            lambda: self._append_array(array, process_patterns=process_patterns)
        )

    def _insert_array(self, array: AssembledDiffractionArray) -> None:
        pos = bisect(self._array_list, array.array_index, key=lambda x: x.array_index)
        self._array_list.insert(pos, array)

        for observer in self._observer_list:
            observer.handle_array_inserted(pos)

    def _on_array_assembled(
        self, array_index: int, label: str, view: AssembledDiffractionData
    ) -> None:
        """Publish one assembled array. Called from an assembly worker thread.

        The list insert is bounced to the foreground queue so the bisect stays
        single-threaded; arrays may complete out of order and are sorted there by
        their monotonic array_index.
        """
        assembled_array = AssembledDiffractionArray(
            array_index=array_index,
            label=label,
            data=view,
        )
        self._task_manager.put_foreground_task(lambda: self._insert_array(assembled_array))

    def clear(self) -> None:
        self._dataset = SimpleDiffractionDataset.create_null()
        self._data = AssembledDiffractionData.create_null()
        self._array_list.clear()
        self._array_counter = 0
        self._source = None
        self._last_array_loader = None
        self._bad_pixels = self._create_default_bad_pixels()

        if self._scratch_tempfile is not None:
            self._scratch_tempfile.close()
            self._scratch_tempfile = None

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def reload(self, dataset: DiffractionDataset) -> None:
        self.clear()
        metadata = dataset.get_metadata()
        self._dataset = SimpleDiffractionDataset(metadata, dataset.get_layout(), [])
        self._bad_pixels = dataset.get_bad_pixels()
        # self._dataset drops the array list, so the source is the only handle on the
        # lazy arrays that load_all_arrays will read.
        self._source = dataset

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def load_all_arrays(self, *, process_patterns: bool, block: bool) -> None:
        source = self._source

        if source is None:
            logger.warning('No dataset queued for loading; call reload() first.')
            return

        metadata = self._dataset.get_metadata()
        plan = self._resolve_prep_plan(
            metadata.detector_extent,
            process_patterns=process_patterns,
            log_level=logging.INFO,
        )
        read_region = plan.read_region if plan is not None else None
        pipeline = plan.pipeline if plan is not None else None
        raw_pixel_geometry = self.get_raw_pixel_geometry()
        lower, upper = self._get_total_counts_bounds()

        patterns_shape = compute_assembled_patterns_shape(
            source, pipeline, bad_pixels=self._bad_pixels, read_region=read_region
        )
        patterns: DiffractionPatterns | None = None

        if self._settings.memmap_enabled.get_value():
            scratch_dir = self._settings.scratch_directory.get_value()
            scratch_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
            # Held on self so the file persists on disk for as long as the memmap is live;
            # released in clear().
            self._scratch_tempfile = tempfile.NamedTemporaryFile(dir=scratch_dir, suffix='.npy')
            logger.info(f'Scratch data file {self._scratch_tempfile.name} is {patterns_shape}')
            patterns = numpy.memmap(
                self._scratch_tempfile, dtype=metadata.pattern_dtype, shape=patterns_shape
            )
            patterns[:] = 0
        else:
            logger.info(f'Scratch memory is {patterns_shape}')

        self._data = allocate_assembled_data(
            source,
            pipeline,
            bad_pixels=self._bad_pixels,
            raw_pixel_geometry=raw_pixel_geometry,
            patterns=patterns,
            read_region=read_region,
        )
        self._array_counter = len(source)

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

        # load all arrays in background
        loader = LoadDiffractionDataset(
            source,
            self._data,
            pipeline,
            bad_pixels=self._bad_pixels,
            raw_pixel_geometry=raw_pixel_geometry,
            total_counts_lower_bound=lower,
            total_counts_upper_bound=upper,
            read_region=read_region,
            on_array_assembled=self._on_array_assembled,
            task_monitor=self._task_monitor,
        )
        finished_event = loader.get_finished_event()
        self._last_array_loader = loader
        self._task_manager.put_background_task(loader)
        self._source = None

        if block:
            while not self._task_manager.is_stopping:
                if finished_event.wait(timeout=TaskManager.WAIT_TIME_S):
                    break

    def _generate_dataset_for_assembled_data(self, file_path: Path | None = None) -> None:
        num_patterns, detector_height, detector_width = self._data.get_patterns_shape()
        metadata = DiffractionMetadata(
            num_patterns_per_array=[num_patterns],
            pattern_dtype=self._data.get_patterns_dtype(),
            detector_extent=ImageExtent(detector_width, detector_height),
            # Deliberately no detector_pixel_geometry: self._data already holds the
            # processed value and downstream consumers read it via
            # get_processed_pixel_geometry(). The raw detector geometry is not
            # recoverable from an exported file (only the processed value is
            # persisted), so leave get_raw_pixel_geometry() to fall through to
            # DetectorSettings for the rare consumer that needs it.
            file_path=file_path,
        )
        contents_tree = DiffractionDatasetLayoutNode.create_root()
        array = AssembledDiffractionArray(
            array_index=0,
            label='In-Memory' if file_path is None else file_path.stem,
            data=self._data,
        )
        self._bad_pixels = self._data.get_bad_pixels()
        self._dataset = SimpleDiffractionDataset(metadata, contents_tree, [], self._bad_pixels)
        self._array_list = [array]
        self._array_counter = 1

        for observer in self._observer_list:
            observer.handle_dataset_reloaded()

    def set_assembled_patterns(self, data: AssembledDiffractionData) -> None:
        self.clear()
        self._data = data
        self._generate_dataset_for_assembled_data(file_path=None)

    def import_assembled_patterns(self, file_path: Path, *, mmap_file: Path | None = None) -> None:
        """Import an assembled dataset, optionally staging patterns into a memory map.

        The caller owns ``mmap_file``; see :func:`load_diffraction_data`.
        """
        if file_path.is_file():
            self.clear()
            logger.info(f'Importing assembled dataset from "{file_path}"')
            self._data = load_diffraction_data(file_path, mmap_file=mmap_file)
            self._generate_dataset_for_assembled_data(file_path=file_path)
        else:
            logger.warning(f'Refusing to read invalid file path {file_path}')

    def export_assembled_patterns(self, file_path: Path, compression: str = 'lzf') -> None:
        logger.info(f'Exporting assembled dataset to "{file_path}"')
        save_diffraction_data(file_path, self._data, compression=compression)
