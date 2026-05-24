"""Reconstructor interfaces, I/O data containers, and assembled diffraction data management."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from pathlib import Path
import logging

import numpy

from .common import BYTES_PER_MEGABYTE, RealArrayType
from .diffraction import (
    BadPixels,
    DiffractionIndexes,
    DiffractionPattern,
    DiffractionPatternCounts,
    DiffractionPatternDType,
    DiffractionPatterns,
)
from .geometry import PixelGeometry
from .object import Object
from .probe import ProbeSequence
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
        return ''

    def load_model_from_file(self, file_path: Path) -> None:
        pass

    def get_training_data_file_filter(self) -> str:
        return ''

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
                'Patterns shape does not match bad pixels shape! '
                f'(actual={patterns.shape[1:]} expected={bad_pixels.shape})'
            )

    @classmethod
    def create_null(cls) -> AssembledDiffractionData:
        return cls(
            indexes=numpy.zeros(1, dtype=numpy.intp),
            patterns=numpy.zeros((1, 1, 1), dtype=numpy.intp),
            pixel_geometry=PixelGeometry(0, 0),
            bad_pixels=numpy.zeros((1, 1), dtype=numpy.bool_),
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


@dataclass(frozen=True)
class ReconstructionAmbiguities:
    """The four scalar ambiguities that ptychography cannot constrain from intensities alone.

    A reconstructed object is determined only up to a global complex scale and a 2D
    linear phase ramp. Concretely, the convention used here is

    ``object = standardized_object * scale * exp(i * (phi + k_x * x + k_y * y))``

    where ``(x, y)`` are measured in meters from the geometric center of the
    object **array** (``(width_px - 1) / 2`` in each axis, not
    ``Object.get_center()``). The corresponding complementary transform on the
    probe leaves the diffraction-pattern intensities at every scan position
    unchanged. See :meth:`standardize_product` for what is and is not exactly
    preserved.

    Attributes:
        object_scale_factor: ``scale`` above. Must be non-zero and finite.
        phase_offset_rad: ``phi`` above, in radians.
        phase_ramp_x_rad_per_m: ``k_x`` above, in rad/m.
        phase_ramp_y_rad_per_m: ``k_y`` above, in rad/m.
    """

    object_scale_factor: float
    phase_offset_rad: float
    phase_ramp_x_rad_per_m: float
    phase_ramp_y_rad_per_m: float

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not numpy.isfinite(value):
                raise ValueError(f'{f.name} must be finite, got {value!r}')

        if self.object_scale_factor == 0.0:
            raise ValueError('object_scale_factor must be non-zero')

    @classmethod
    def identity(cls) -> ReconstructionAmbiguities:
        """Return the no-op instance: unit scale, zero phase, zero ramp."""
        return cls(
            object_scale_factor=1.0,
            phase_offset_rad=0.0,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )

    def _phase_ramp_grid(
        self, position_x_m: RealArrayType, position_y_m: RealArrayType
    ) -> RealArrayType:
        return (
            self.phase_ramp_x_rad_per_m * position_x_m + self.phase_ramp_y_rad_per_m * position_y_m
        )

    def standardize_product(self, product: Product) -> Product:
        """Remove these ambiguities from ``product`` and return the standardized result.

        The inverse correction is applied to layer 0 of the object (other layers
        pass through unchanged) and the complementary correction is applied to
        every coherent and incoherent mode of the probe in the probe's own local
        frame. Probe positions, OPR weights, losses, and metadata pass through.

        What is preserved:

        - **Diffraction-pattern intensities** at every scan position, exactly
          (up to floating-point precision).
        - **Exit waves** (``probe * object``) exactly, when the phase ramp is
          zero (``phase_ramp_x_rad_per_m == phase_ramp_y_rad_per_m == 0``).

        When the phase ramp is non-zero, the standardized exit wave at scan
        position ``r_pos`` differs from the original by a per-position scalar
        factor ``exp(-i * k * (r_pos - r_obj_array_center))``. This is itself
        one of the unmeasurable ambiguities of ptychography (diffraction
        intensities are insensitive to a global phase per pattern), so all
        observable quantities still agree exactly.
        """
        obj = product.object_
        obj_array = obj.get_array()
        obj_geometry = obj.get_geometry()
        obj_coords = obj_geometry.get_transverse_coordinates()

        obj_ramp = self._phase_ramp_grid(obj_coords.position_x_m, obj_coords.position_y_m)
        obj_correction = (
            numpy.exp(-1j * (self.phase_offset_rad + obj_ramp)) / self.object_scale_factor
        )
        standardized_array = obj_array.copy()
        standardized_array[0] = (obj_array[0] * obj_correction).astype(obj_array.dtype)
        standardized_object = Object(
            array=standardized_array,
            pixel_geometry=obj.get_pixel_geometry().copy(),
            center=obj.get_center().copy(),
            layer_spacing_m=list(obj.layer_spacing_m),
        )

        probes = product.probes
        probe_array = probes.get_array()
        probe_coords = probes.get_geometry().get_transverse_coordinates()

        probe_ramp = self._phase_ramp_grid(probe_coords.position_x_m, probe_coords.position_y_m)
        probe_correction = self.object_scale_factor * numpy.exp(
            1j * (self.phase_offset_rad + probe_ramp)
        )
        opr_weights = probes.get_opr_weights_or_none()
        standardized_probes = ProbeSequence(
            array=(probe_array * probe_correction).astype(probe_array.dtype),
            opr_weights=None if opr_weights is None else opr_weights.copy(),
            pixel_geometry=probes.get_pixel_geometry().copy(),
        )

        return Product(
            metadata=product.metadata,
            probe_positions=product.probe_positions,
            probes=standardized_probes,
            object_=standardized_object,
            losses=product.losses,
        )
