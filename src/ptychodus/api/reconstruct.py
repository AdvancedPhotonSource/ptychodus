"""Reconstructor interfaces, I/O data containers, and product standardization."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from pathlib import Path
import logging

import numpy

from .diffraction import AssembledDiffractionData, BadPixels, DiffractionPatterns
from .object import Object
from .probe import ProbeSequence
from .probe_positions import ProbePositionSequence, ProbePosition
from .product import LossValue, Product
from .typing import RealArrayType

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
    def get_model_file_extension(self) -> str:
        """Return the file extension (with leading dot) used when saving the model."""
        pass

    @abstractmethod
    def save_model(self, file_path: Path) -> None:
        """Write the currently-loaded model to file_path."""
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

    def get_model_file_extension(self) -> str:
        return ''

    def save_model(self, file_path: Path) -> None:
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


def prepare_reconstruct_input(
    assembled_data: AssembledDiffractionData,
    product: Product,
    index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
) -> ReconstructInput:
    """Pair diffraction patterns to probe positions by scan index and package them for a reconstructor.

    Pattern indexes are authoritative. The position sequence is conditioned to the pattern-index
    axis in three steps:

    - duplicate position indexes are averaged into a single anchor;
    - pattern indexes inside the position-index range with no matching position are linearly
      interpolated from neighboring anchors;
    - pattern indexes outside the position-index range are dropped (no extrapolation).

    Positions whose index has no matching pattern naturally do not appear in the output. The OPR
    caveat below is unchanged from the prior implementation; interpolation makes the misalignment
    worse because interpolated entries have no natural OPR row at all.
    """
    # TODO also filter OPR weights
    pattern_indexes = assembled_data.get_indexes()
    valid_patterns = assembled_data.get_patterns()

    if pattern_indexes.size == 0:
        raise ValueError('Cannot prepare reconstruct input from empty diffraction dataset.')

    pattern_keep = numpy.fromiter(
        (index_filter(int(i)) for i in pattern_indexes),
        dtype=numpy.bool_,
        count=pattern_indexes.size,
    )
    filtered_pattern_indexes = pattern_indexes[pattern_keep]
    filtered_pattern_offsets = numpy.flatnonzero(pattern_keep)

    n_positions = len(product.probe_positions)
    pos_indexes_all = numpy.empty(n_positions, dtype=numpy.intp)
    pos_x_all = numpy.empty(n_positions, dtype=numpy.float64)
    pos_y_all = numpy.empty(n_positions, dtype=numpy.float64)
    for k, position in enumerate(product.probe_positions):
        pos_indexes_all[k] = position.index
        pos_x_all[k] = position.coordinate_x_m
        pos_y_all[k] = position.coordinate_y_m

    pos_keep = numpy.fromiter(
        (index_filter(int(i)) for i in pos_indexes_all),
        dtype=numpy.bool_,
        count=n_positions,
    )
    pos_indexes = pos_indexes_all[pos_keep]
    pos_x = pos_x_all[pos_keep]
    pos_y = pos_y_all[pos_keep]

    if filtered_pattern_indexes.size == 0 or pos_indexes.size == 0:
        raise ValueError('Index filter eliminated all pattern indexes and/or all position indexes.')

    # Average duplicate position indexes. numpy.unique sorts ascending, so
    # the resulting (unique_pos_indexes, mean_x, mean_y) triple is the
    # canonical interpolation anchor set.
    unique_pos_indexes, inverse = numpy.unique(pos_indexes, return_inverse=True)
    counts = numpy.bincount(inverse)
    mean_x = numpy.bincount(inverse, weights=pos_x) / counts
    mean_y = numpy.bincount(inverse, weights=pos_y) / counts

    lo = int(unique_pos_indexes[0])
    hi = int(unique_pos_indexes[-1])
    in_range_mask = (filtered_pattern_indexes >= lo) & (filtered_pattern_indexes <= hi)
    in_range_pattern_indexes = filtered_pattern_indexes[in_range_mask]
    in_range_pattern_offsets = filtered_pattern_offsets[in_range_mask]

    if in_range_pattern_indexes.size == 0:
        pat_lo = int(filtered_pattern_indexes[0])
        pat_hi = int(filtered_pattern_indexes[-1])
        raise ValueError(
            'No probe positions overlap the diffraction pattern indexes; '
            f'pattern indexes span [{pat_lo}, {pat_hi}] and position indexes '
            f'span [{lo}, {hi}].'
        )

    # When unique_pos_indexes.size == 1, lo == hi, so only pattern indexes
    # equal to that single anchor survive the in_range trim -- and those
    # match exactly, never requiring interpolation. So the "need >= 2
    # anchors to interpolate" rule is enforced by the in_range trim itself;
    # no separate guard is needed.
    exact_match = numpy.isin(in_range_pattern_indexes, unique_pos_indexes)

    # numpy.interp returns the exact value at coincident xp entries and
    # linearly interpolates between them. Out-of-range is impossible here
    # because in_range_mask already trimmed to [lo, hi].
    x_coords = numpy.interp(in_range_pattern_indexes, unique_pos_indexes, mean_x)
    y_coords = numpy.interp(in_range_pattern_indexes, unique_pos_indexes, mean_y)

    n_averaged = pos_indexes.size - unique_pos_indexes.size
    n_interpolated = int((~exact_match).sum())
    logger.debug(
        f'prepare_reconstruct_input: matched {in_range_pattern_indexes.size} of '
        f'{filtered_pattern_indexes.size} pattern indexes; averaged {n_averaged} '
        f'duplicate positions; interpolated {n_interpolated} positions'
    )

    point_list = [
        ProbePosition(index=int(i), coordinate_x_m=float(x), coordinate_y_m=float(y))
        for i, x, y in zip(in_range_pattern_indexes, x_coords, y_coords)
    ]
    patterns = valid_patterns[in_range_pattern_offsets]

    product = Product(
        metadata=product.metadata,
        probe_positions=ProbePositionSequence(point_list),
        # OPR weight rows are still array-indexed by the original scan
        # ordering, not by the merged-and-interpolated subset. Correct only
        # when matched positions are contiguous from zero with no
        # interpolation; latent misalignment otherwise.
        probes=product.probes,
        object_=product.object_,
        losses=product.losses,
    )

    return ReconstructInput(patterns, assembled_data.get_bad_pixels(), product)


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

    """

    object_scale_factor: float
    """``scale`` above. Must be non-zero and finite."""
    phase_offset_rad: float
    """``phi`` above, in radians."""
    phase_ramp_x_rad_per_m: float
    """``k_x`` above, in rad/m."""
    phase_ramp_y_rad_per_m: float
    """``k_y`` above, in rad/m."""

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if not numpy.isfinite(value):
                raise ValueError(f'{f.name} must be finite, got {value!r}')

        if self.object_scale_factor == 0.0:
            raise ValueError('object_scale_factor must be non-zero')

    @classmethod
    def create_identity(cls) -> ReconstructionAmbiguities:
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
