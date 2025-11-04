from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import concurrent.futures
import logging
import threading

import numpy

from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionIndexes,
    DiffractionPatterns,
    SimpleDiffractionArray,
)
from ptychodus.api.units import BYTES_PER_MEGABYTE

from ..task_manager import BackgroundTask, ForegroundTask, ForegroundTaskManager
from .processor import DiffractionPatternProcessor

logger = logging.getLogger('.'.join(__name__.split('.')[:-1]))


@dataclass(frozen=True)
class AssembledDiffractionData:
    indexes: DiffractionIndexes
    patterns: DiffractionPatterns
    pattern_counts: DiffractionPatterns

    def __post_init__(self) -> None:
        if self.indexes.ndim != 1:
            raise ValueError(
                'Unexpected number of dimensions for indexes!'
                f' (actual={self.indexes.ndim} expected=1)'
            )

        if self.patterns.ndim != 3:
            raise ValueError(
                'Unexpected number of dimensions for patterns!'
                f' (actual={self.patterns.ndim} expected=3)'
            )

        if self.pattern_counts.ndim != 1:
            raise ValueError(
                'Unexpected number of dimensions for pattern counts!'
                f' (actual={self.pattern_counts.ndim} expected=1)'
            )

        if self.indexes.shape[0] != self.patterns.shape[0]:
            raise ValueError('Number of indexes does not match number of patterns!')

        if self.pattern_counts.shape[0] != self.patterns.shape[0]:
            raise ValueError('Number of patterns does not match number of pattern counts!')

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=int),
            patterns=numpy.zeros((1, 1, 1), dtype=int),
            pattern_counts=numpy.zeros(1, dtype=int),
        )

    @classmethod
    def create_pattern_counts(
        cls, indexes: DiffractionIndexes, patterns: DiffractionPatterns, bad_pixels: BadPixels
    ) -> AssembledDiffractionData:
        good_pixels = numpy.logical_not(bad_pixels)
        return cls(
            indexes=indexes,
            patterns=patterns,
            pattern_counts=numpy.sum(patterns[:, good_pixels], axis=-1),
        )

    def get_assembled_indexes(self) -> DiffractionIndexes:
        return self.indexes[self.indexes >= 0]

    def get_assembled_patterns(self) -> DiffractionPatterns:
        return self.patterns[self.indexes >= 0]

    def get_assembled_pattern_counts(self) -> DiffractionPatterns:
        return self.pattern_counts[self.indexes >= 0]

    def get_pattern_counts_lut(self) -> Mapping[int, int]:
        return dict(zip(self.get_assembled_indexes(), self.get_assembled_pattern_counts()))

    def get_info_text(self, label: str) -> str:
        number, height, width = self.patterns.shape
        dtype = str(self.patterns.dtype)
        size_MB = self.patterns.nbytes / BYTES_PER_MEGABYTE  # noqa: N806
        return f'{label}: {number} x {width}W x {height}H {dtype} [{size_MB:.2f}MB]'


class ArrayAssembler(ABC):
    @abstractmethod
    def _create_array_loader(
        self, array: DiffractionArray, *, process_patterns: bool
    ) -> BackgroundTask:
        pass

    @abstractmethod
    def _assemble_array(
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
        bad_pixels: BadPixels,
        processor: DiffractionPatternProcessor | None,
        assembler: ArrayAssembler,
    ) -> None:
        super().__init__()
        self._array_index = array_index
        self._array = array
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
            data = AssembledDiffractionData.create_pattern_counts(
                indexes=processed_array.get_indexes(),
                patterns=processed_array.get_patterns(),
                bad_pixels=self._bad_pixels,
            )
            self._assembler._assemble_array(
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
        *,
        process_patterns: bool,
    ) -> None:
        super().__init__()
        self._array_seq = array_seq
        self._assembler = assembler
        self._foreground_task_manager = foreground_task_manager
        self._process_patterns = process_patterns
        self._finished_event = threading.Event()

    def get_finished_event(self) -> threading.Event:
        return self._finished_event

    def __call__(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_list = [
                executor.submit(
                    lambda loader_task: loader_task(),
                    self._assembler._create_array_loader(
                        array, process_patterns=self._process_patterns
                    ),
                )
                for array in self._array_seq
            ]

            for future in concurrent.futures.as_completed(future_list):
                try:
                    task = future.result()
                except Exception as ex:
                    logger.warning(ex)
                else:
                    if task is not None:
                        self._foreground_task_manager.put_foreground_task(task)

        self._finished_event.set()
