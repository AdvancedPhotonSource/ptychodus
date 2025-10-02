from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import concurrent.futures
import logging

import numpy

from ptychodus.api.diffraction import (
    BadPixels,
    DiffractionArray,
    DiffractionIndexes,
    DiffractionPatterns,
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

    def get_info_text(self, label: str) -> str:
        number, height, width = self.patterns.shape
        dtype = str(self.patterns.dtype)
        size_MB = self.patterns.nbytes / BYTES_PER_MEGABYTE  # noqa: N806
        return f'{label}: {number} x {width}W x {height}H {dtype} [{size_MB:.2f}MB]'


class ArrayAssembler(ABC):
    @abstractmethod
    def _create_array_loader(self, array: DiffractionArray) -> BackgroundTask:
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
        processor: DiffractionPatternProcessor,
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
            processed_array = self._processor(self._array)
        except FileNotFoundError:
            logger.warning(f'File not found for "{label}"!')
        else:
            patterns = processed_array.get_patterns()
            data = AssembledDiffractionData.create_pattern_counts(
                indexes=processed_array.get_indexes(),
                patterns=patterns,
                bad_pixels=self._bad_pixels,
            )
            self._assembler._assemble_array(
                self._array_index,
                processed_array.get_label(),
                data,
            )

        return None


class LoadAllArrays:
    def __init__(
        self,
        array_seq: Sequence[DiffractionArray],
        processor: DiffractionPatternProcessor,
        processed_bad_pixels: BadPixels,
        assembler: ArrayAssembler,
        foreground_task_manager: ForegroundTaskManager,
    ) -> None:
        super().__init__()
        self._array_seq = array_seq
        self._processor = processor
        self._processed_bad_pixels = processed_bad_pixels
        self._assembler = assembler
        self._foreground_task_manager = foreground_task_manager

    def _load(self, task: BackgroundTask) -> ForegroundTask | None:
        return task()

    def __call__(self) -> None:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_list = [
                executor.submit(self._load, self._assembler._create_array_loader(array))
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
