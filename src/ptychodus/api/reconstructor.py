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
        # Pattern indexes are authoritative. The position sequence is conditioned
        # to the pattern-index axis in three steps:
        #   - duplicate position indexes are averaged into a single anchor;
        #   - pattern indexes inside the position-index range with no matching
        #     position are linearly interpolated from neighboring anchors;
        #   - pattern indexes outside the position-index range are dropped
        #     (no extrapolation).
        # Positions whose index has no matching pattern naturally do not appear
        # in the output. The OPR caveat below is unchanged from the prior
        # implementation; interpolation makes the misalignment worse because
        # interpolated entries have no natural OPR row at all.
        valid_mask = self._indexes >= 0
        pattern_indexes = self._indexes[valid_mask]
        valid_patterns = self._patterns[valid_mask]

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
            raise ValueError(
                'Index filter eliminated all pattern indexes and/or all position indexes.'
            )

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
    def create_identity(cls) -> ReconstructionAmbiguities:
        """Return the no-op instance: unit scale, zero phase, zero ramp."""
        return cls(
            object_scale_factor=1.0,
            phase_offset_rad=0.0,
            phase_ramp_x_rad_per_m=0.0,
            phase_ramp_y_rad_per_m=0.0,
        )

    @classmethod
    def estimate(
        cls,
        product: Product,
        *,
        reference: Product | None = None,
        weights: RealArrayType | None = None,
    ) -> ReconstructionAmbiguities:
        """Estimate the ambiguities present in ``product``.

        Without ``reference``: estimate ``(phi, k_x, k_y)`` that flatten layer
        0's phase in the amplitude-weighted circular-mean sense.
        ``object_scale_factor`` is fixed at ``1.0`` because there is no
        reference amplitude to normalize against.

        With ``reference``: estimate all four ambiguities ``(s, phi, k_x, k_y)``
        on ``product`` such that
        ``estimate.standardize_product(product)`` best matches ``reference`` in
        the weighted least-squares sense. The two products must agree in
        layer-0 shape and object pixel geometry. The driving signal becomes
        ``S = product[0] * conj(reference[0])``, whose phase is exactly
        ``phi + k_x*x + k_y*y`` and whose magnitude ``|product| * |reference|``
        provides natural amplitude weighting (pixels where either product is
        weak contribute little).

        The estimate is fully complex-domain (sums of phasors, ``numpy.angle``
        of complex weighted sums) and so requires no phase unwrapping. Pixels
        of zero amplitude contribute exactly zero to the relevant sums and are
        therefore ignored automatically.

        Args:
            product: Product whose ambiguities are being measured. The result
                is returned in this product's coordinate frame.
            reference: Optional anchor product. When supplied, the scale factor
                is estimated too; when ``None``, scale is fixed at ``1.0``.
            weights: Optional non-negative per-pixel weight array, shape
                ``(height_px, width_px)`` matching layer 0. Multiplies the
                natural amplitude weighting. Pass a 0/1 mask to restrict the
                estimate to a region of interest.
        """
        obj = product.object_
        layer_zero = obj.get_array()[0].astype(numpy.complex128)
        pixel_geometry = obj.get_pixel_geometry()
        coords = obj.get_geometry().get_transverse_coordinates()
        weights_arr = _validate_weights(weights, layer_zero.shape)

        if reference is None:
            ref_layer_zero = None
            signal = layer_zero
        else:
            ref_obj = reference.object_
            ref_shape = ref_obj.get_array().shape[-2:]

            if ref_shape != layer_zero.shape:
                raise ValueError(
                    f'Object layer-0 shape mismatch: reference {ref_shape} vs product {layer_zero.shape}!'
                )

            ref_pixel_geometry = ref_obj.get_pixel_geometry()

            if ref_pixel_geometry != pixel_geometry:
                raise ValueError(
                    f'Object pixel geometry mismatch: reference {ref_pixel_geometry} vs product {pixel_geometry}!'
                )

            ref_layer_zero = ref_obj.get_array()[0].astype(numpy.complex128)
            signal = layer_zero * numpy.conj(ref_layer_zero)

        phi, k_x, k_y = _estimate_phase_offset_and_ramp(
            signal=signal,
            weights=weights_arr,
            pixel_width_m=pixel_geometry.width_m,
            pixel_height_m=pixel_geometry.height_m,
            position_x_m=coords.position_x_m,
            position_y_m=coords.position_y_m,
        )

        if ref_layer_zero is None:
            s = 1.0
        else:
            # Weighted-LS solution for s in product ≈ s * exp(i(phi + k·r)) * ref:
            # s = Re(sum w * signal * exp(-i(phi + ramp))) / sum(w * |ref|^2).
            ramp = k_x * coords.position_x_m + k_y * coords.position_y_m
            correction = numpy.exp(-1j * (phi + ramp))
            ref_intensity = numpy.square(numpy.abs(ref_layer_zero))

            if weights_arr is None:
                numerator = float(numpy.real(numpy.sum(signal * correction)))
                denominator = float(numpy.sum(ref_intensity))
            else:
                numerator = float(numpy.real(numpy.sum(weights_arr * signal * correction)))
                denominator = float(numpy.sum(weights_arr * ref_intensity))

            if not (denominator > 0.0):
                raise ValueError(
                    'Cannot estimate scale: weighted reference object intensity is zero.'
                )

            s = numerator / denominator

            # Convention: keep object_scale_factor > 0. Fold any sign flip into phi.
            if s < 0.0:
                s = -s
                phi = phi + float(numpy.pi)

        return cls(
            object_scale_factor=s,
            phase_offset_rad=phi,
            phase_ramp_x_rad_per_m=k_x,
            phase_ramp_y_rad_per_m=k_y,
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


def _validate_weights(
    weights: RealArrayType | None, expected_shape: tuple[int, ...]
) -> RealArrayType | None:
    if weights is None:
        return None
    weights_arr = numpy.asarray(weights, dtype=numpy.float64)
    if weights_arr.shape != expected_shape:
        raise ValueError(
            f'weights shape {weights_arr.shape} does not match'
            f' object layer 0 shape {expected_shape}!'
        )
    if not numpy.all(numpy.isfinite(weights_arr)):
        raise ValueError('weights must all be finite!')
    if numpy.any(weights_arr < 0.0):
        raise ValueError('weights must all be non-negative!')
    return weights_arr


def _estimate_phase_offset_and_ramp(
    *,
    signal: numpy.ndarray,
    weights: RealArrayType | None,
    pixel_width_m: float,
    pixel_height_m: float,
    position_x_m: RealArrayType,
    position_y_m: RealArrayType,
) -> tuple[float, float, float]:
    """Recover (phi, k_x_rad_per_m, k_y_rad_per_m) from a complex 2D signal.

    The signal's phase is assumed to be ``phi + k_x*x + k_y*y`` plus
    high-frequency content; its magnitude provides the natural amplitude
    weighting. The ramp is recovered from per-pixel complex differences along
    each axis (so unwrapping is unnecessary), then ``phi`` is recovered as the
    weighted circular mean of the de-ramped signal.
    """
    # Differential phase along x: arg(S[:, x+1] * conj(S[:, x])) carries
    # k_x * pixel_width_m modulo 2pi without ever wrapping per-pair.
    delta_x = signal[:, 1:] * numpy.conj(signal[:, :-1])
    if weights is None:
        accum_x = numpy.sum(delta_x)
    else:
        w_x = weights[:, 1:] * weights[:, :-1]
        accum_x = numpy.sum(w_x * delta_x)
    k_x_per_px = float(numpy.angle(accum_x))

    delta_y = signal[1:, :] * numpy.conj(signal[:-1, :])
    if weights is None:
        accum_y = numpy.sum(delta_y)
    else:
        w_y = weights[1:, :] * weights[:-1, :]
        accum_y = numpy.sum(w_y * delta_y)
    k_y_per_px = float(numpy.angle(accum_y))

    k_x_rad_per_m = k_x_per_px / pixel_width_m
    k_y_rad_per_m = k_y_per_px / pixel_height_m

    ramp = k_x_rad_per_m * position_x_m + k_y_rad_per_m * position_y_m
    signal_deramped = signal * numpy.exp(-1j * ramp)
    if weights is None:
        phi_accum = numpy.sum(signal_deramped)
    else:
        phi_accum = numpy.sum(weights * signal_deramped)

    if phi_accum == 0:
        raise ValueError('Cannot estimate phase offset: weighted signal magnitude is zero.')

    phi = float(numpy.angle(phi_accum))
    return phi, k_x_rad_per_m, k_y_rad_per_m
