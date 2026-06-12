from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
import concurrent.futures
import logging
import threading

from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    SimpleDiffractionArray,
)
from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.io import AssembledDiffractionData

from ..task_manager import BackgroundTask, ForegroundTask, ForegroundTaskManager
from ..task_monitor import TaskProgressMonitor
from .processor import DiffractionPatternProcessor

logger = logging.getLogger(__name__)


class ArrayAssembler(ABC):
    @abstractmethod
    def create_array_loader(
        self, array: DiffractionArray, *, process_patterns: bool
    ) -> BackgroundTask:
        pass

    @abstractmethod
    def assemble_array(
        self,
        array_index: int,
        label: str,
        data: AssembledDiffractionData,
    ) -> None:
        pass


class LoadArray:
    def __init__(
        self,
        array_index: int,
        array: DiffractionArray,
        pixel_geometry: PixelGeometry,
        bad_pixels: BadPixels,
        processor: DiffractionPatternProcessor | None,
        assembler: ArrayAssembler,
    ) -> None:
        super().__init__()
        self._array_index = array_index
        self._array = array
        self._pixel_geometry = pixel_geometry
        self._bad_pixels = bad_pixels
        self._processor = processor
        self._assembler = assembler

    def __call__(self) -> ForegroundTask | None:
        label = self._array.get_label()

        try:
            loaded_array = SimpleDiffractionArray(
                label,
                self._array.get_indexes(),
                self._array.get_patterns(),
            )
        except FileNotFoundError:
            logger.warning(f'File not found for "{label}"!')
        else:
            processed_array = (
                loaded_array if self._processor is None else self._processor(loaded_array)
            )
            data = AssembledDiffractionData(
                indexes=processed_array.get_indexes(),
                patterns=processed_array.get_patterns(),
                pixel_geometry=self._pixel_geometry,
                bad_pixels=self._bad_pixels,
            )
            self._assembler.assemble_array(
                self._array_index,
                label,
                data,
            )

        return None


class LoadAllArrays:
    def __init__(
        self,
        array_seq: Sequence[DiffractionArray],
        assembler: ArrayAssembler,
        foreground_task_manager: ForegroundTaskManager,
        task_monitor: TaskProgressMonitor,
    ) -> None:
        super().__init__()
        self._array_seq = array_seq
        self._assembler = assembler
        self._foreground_task_manager = foreground_task_manager
        self._task_monitor = task_monitor
        self._process_patterns = False
        self._finished_event = threading.Event()

    def enable_pattern_processing(self) -> None:
        self._process_patterns = True

    def get_finished_event(self) -> threading.Event:
        return self._finished_event

    def __call__(self) -> None:
        try:
            with self._task_monitor as monitor:
                total = len(self._array_seq)
                monitor.update_progress(0, total)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_list = [
                        executor.submit(
                            self._assembler.create_array_loader(
                                array,
                                process_patterns=self._process_patterns,
                            )
                        )
                        for array in self._array_seq
                    ]

                    completed = 0
                    cancelled = False

                    for future in concurrent.futures.as_completed(future_list):
                        try:
                            task = future.result()
                        except concurrent.futures.CancelledError:
                            pass
                        except Exception as ex:
                            logger.warning(ex)
                        else:
                            if task is not None:
                                self._foreground_task_manager.put_foreground_task(task)

                        completed += 1
                        monitor.update_progress(completed, total)

                        if not cancelled and monitor.is_stopping:
                            num_cancelled = sum(1 for f in future_list if f.cancel())
                            logger.info(
                                f'Diffraction load stop requested; cancelled '
                                f'{num_cancelled} pending arrays. In-flight reads will finish.'
                            )
                            cancelled = True
        finally:
            self._finished_event.set()
