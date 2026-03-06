"""Reconstructor interfaces, I/O data containers, and assembled diffraction data management."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import auto, Enum
from pathlib import Path
import logging

import numpy

from .common import BYTES_PER_MEGABYTE
from .diffraction import (
    BadPixels,
    DiffractionIndexes,
    DiffractionPattern,
    DiffractionPatternCounts,
    DiffractionPatternDType,
    DiffractionPatterns,
)
from .geometry import PixelGeometry
from .probe_positions import ProbePositionSequence, ProbePosition
from .product import LossValue, Product

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReconstructInput:
    """All data required to start a reconstruction: patterns, bad-pixel mask, and initial product."""

    diffraction_patterns: DiffractionPatterns
    bad_pixels: BadPixels
    product: Product


@dataclass(frozen=True)
class ReconstructOutput:
    """Reconstruction result yielded at each iteration: updated product, progress, and status code."""

    product: Product
    progress: int = 0
    status: int = 0


class Reconstructor(ABC):
    """Abstract interface for iterative ptychography reconstruction algorithms."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_progress_goal(self) -> int:
        pass

    @abstractmethod
    def reconstruct(self, parameters: ReconstructInput) -> Iterator[ReconstructOutput]:
        pass


@dataclass(frozen=True)
class TrainOutput:
    """Training result yielded at each step: loss curves, progress, and status code."""

    training_loss: Sequence[LossValue] = field(default_factory=list)
    validation_loss: Sequence[LossValue] = field(default_factory=list)
    progress: int = 0
    status: int = 0


class TrainableReconstructor(Reconstructor):
    """Reconstructor that also supports ML model loading and training."""

    @abstractmethod
    def is_model_loaded(self) -> bool:
        pass

    @abstractmethod
    def get_model_file_filter(self) -> str:
        pass

    @abstractmethod
    def load_model_from_file(self, file_path: Path) -> None:
        pass

    @abstractmethod
    def get_training_data_file_filter(self) -> str:
        pass

    @abstractmethod
    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        pass

    @abstractmethod
    def train(self, input_path: Path, output_path: Path) -> Iterator[TrainOutput]:
        pass


class NullReconstructor(TrainableReconstructor):
    """No-op TrainableReconstructor used as a placeholder when no algorithm is available."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_progress_goal(self) -> int:
        return 0

    def reconstruct(self, parameters: ReconstructInput) -> Iterator[ReconstructOutput]:
        yield from ()

    def is_model_loaded(self) -> bool:
        return False

    def get_model_file_filter(self) -> str:
        return str()

    def load_model_from_file(self, file_path: Path) -> None:
        pass

    def get_training_data_file_filter(self) -> str:
        return str()

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        pass

    def train(self, input_path: Path, output_path: Path) -> Iterator[TrainOutput]:
        yield from ()


class ReconstructorLibrary(Iterable[Reconstructor], ABC):
    """Iterable collection of Reconstructor instances provided by a plugin library."""

    def __init__(self, logger_name: str) -> None:
        super().__init__()
        self._logger = logging.getLogger(logger_name)

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def log_levels(self) -> Iterable[str]:
        return ('CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG')

    def get_logger(self) -> logging.Logger:
        return self._logger

    def get_log_level(self) -> str:
        level = self._logger.getEffectiveLevel()
        return logging.getLevelName(level)

    def set_log_level(self, name: str) -> None:
        name_before = self.get_log_level()

        try:
            self._logger.setLevel(name)
        except ValueError:
            self._logger.error(f'Bad log level "{name}".')

        name_after = self.get_log_level()
        self._logger.info(f'Changed {self.name} logging level {name_before} -> {name_after}')


class PositionIndexFilter(Enum):
    """Filter scan points by scan index."""

    ALL = auto()
    ODD = auto()
    EVEN = auto()

    def __call__(self, index: int) -> bool:
        """Return True to include the scan point, False to exclude it."""
        if self is PositionIndexFilter.ODD:
            return index & 1 != 0
        elif self is PositionIndexFilter.EVEN:
            return index & 1 == 0

        return True


class AssembledDiffractionData:
    """In-memory store for a complete set of indexed diffraction patterns and their bad-pixel mask."""

    def __init__(
        self,
        indexes: DiffractionIndexes,
        patterns: DiffractionPatterns,
        pixel_geometry: PixelGeometry,
        bad_pixels: BadPixels,
    ) -> None:
        self._indexes = indexes
        self._patterns = patterns
        self._pixel_geometry = pixel_geometry
        self._bad_pixels = bad_pixels

        if indexes.ndim != 1:
            raise ValueError(
                f'Unexpected number of dimensions for indexes! (actual={indexes.ndim} expected=1)'
            )

        if patterns.ndim != 3:
            raise ValueError(
                f'Unexpected number of dimensions for patterns! (actual={patterns.ndim} expected=3)'
            )

        if bad_pixels.ndim != 2:
            raise ValueError(
                f'Unexpected number of dimensions for bad pixels! (actual={bad_pixels.ndim} expected=2)'
            )

        if indexes.shape[0] != patterns.shape[0]:
            raise ValueError('Number of indexes does not match number of patterns!')

        if patterns.shape[1:] != bad_pixels.shape:
            raise ValueError(
                'Patterns shape does not match bad pixels shape!'
                f'(actual={patterns.shape[1:]} expected={bad_pixels.shape})'
            )

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=int),
            patterns=numpy.zeros((1, 1, 1), dtype=int),
            pixel_geometry=PixelGeometry(0, 0),
            bad_pixels=numpy.zeros((1, 1), dtype=bool),
        )

    def get_patterns_shape(self) -> tuple[int, int, int]:
        return self._patterns.shape

    def get_patterns_dtype(self) -> DiffractionPatternDType:
        return self._patterns.dtype

    def get_pattern(self, index: int) -> DiffractionPattern:
        return self._patterns[index]

    def get_pixel_geometry(self) -> PixelGeometry:
        return self._pixel_geometry

    def get_bad_pixels(self) -> BadPixels:
        # FIXME verify assembled diffraction pattern pixel geometry is set correctly
        return self._bad_pixels

    def assemble(self, data: AssembledDiffractionData, offset: int) -> AssembledDiffractionData:
        assembled_indexes = slice(offset, offset + len(data._indexes))

        self._indexes[assembled_indexes] = data._indexes
        indexes_view = self._indexes[assembled_indexes]
        indexes_view.flags.writeable = False

        self._patterns[assembled_indexes, :, :] = data._patterns
        patterns_view = self._patterns[assembled_indexes, :, :]
        patterns_view.flags.writeable = False

        return AssembledDiffractionData(
            indexes=indexes_view,
            patterns=patterns_view,
            pixel_geometry=self._pixel_geometry,
            bad_pixels=data._bad_pixels,
        )

    def get_indexes(self) -> DiffractionIndexes:
        return self._indexes[self._indexes >= 0]

    def get_patterns(self) -> DiffractionPatterns:
        return self._patterns[self._indexes >= 0]

    def get_pattern_counts(self) -> DiffractionPatternCounts:
        good_pixels = numpy.logical_not(self._bad_pixels)
        assembled_patterns = self.get_patterns()
        pattern_counts = numpy.sum(assembled_patterns[:, good_pixels], axis=-1)
        return pattern_counts

    def get_average_pattern(self) -> DiffractionPattern:
        assembled_patterns = self.get_patterns()
        return numpy.mean(assembled_patterns, axis=0)

    def prepare_reconstruct_input(
        self,
        product: Product,
        index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
    ) -> ReconstructInput:
        # TODO also filter OPR weights
        pattern_indexes = [int(index) for index in self.get_indexes()]
        logger.debug(f'{pattern_indexes=}')
        position_indexes = [
            int(point.index) for point in product.probe_positions if index_filter(point.index)
        ]
        logger.debug(f'{position_indexes=}')
        common_indexes = sorted(set(pattern_indexes).intersection(position_indexes))
        logger.debug(f'{common_indexes=}')

        patterns = numpy.take(
            self.get_patterns(),
            common_indexes,
            axis=0,
        )

        point_list: list[ProbePosition] = list()
        point_iterator = iter(product.probe_positions)

        for index in common_indexes:
            while True:
                point = next(point_iterator)

                if point.index == index:
                    point_list.append(point)
                    break

        product = Product(
            metadata=product.metadata,
            probe_positions=ProbePositionSequence(point_list),
            probes=product.probes,  # TODO remap if needed
            object_=product.object_,
            losses=product.losses,
        )

        return ReconstructInput(patterns, self._bad_pixels, product)

    def __str__(self) -> str:
        number, height, width = self._patterns.shape
        dtype = str(self._patterns.dtype)
        size_MB = self._patterns.nbytes / BYTES_PER_MEGABYTE  # noqa: N806
        return f'{number} x {height}H x {width}W {dtype} [{size_MB:.2f}MB]'
