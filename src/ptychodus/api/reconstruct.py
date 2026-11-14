"""Reconstructor interfaces, I/O data containers, and product standardization."""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
import logging

import numpy
from pydantic import BaseModel

from .assemble import AssembledDiffractionData
from .files import FileWriter
from .diffraction import (
    BadPixels,
    DetectorBackground,
    DiffractionPatterns,
    FlatField,
)
from .geometry import ImageExtent
from .interpolate import interpolate_at_indexes
from .object import Object
from .probe import ProbeEnsemble
from .probe_positions import ProbePositionSequence, ProbePosition
from .product import LossValue, Product
from .propagate import compute_full_aperture_fresnel_number
from .scattering.geometry import find_scattering_issues
from .typing import RealArrayType

__all__ = [
    'NoOptions',
    'PositionIndexFilter',
    'ReconstructInput',
    'ReconstructionAmbiguities',
    'ReconstructOutput',
    'Reconstructor',
    'ReconstructorLibrary',
    'TrainOutput',
    'TrainableReconstructor',
    'TrainingDataFileWriter',
    'prepare_reconstruct_input',
    'warn_if_propagation_regime_disagrees',
    'warn_if_scattering_geometry_unsupported',
]

logger = logging.getLogger(__name__)


class NoOptions(BaseModel):
    """Declared by a reconstructor that does not yet describe its options.

    A reconstructor states what it accepts by returning a pydantic object from
    :meth:`Reconstructor.get_default_options`. Until a backend declares one it keeps
    reading whatever configuration it already holds, and this stands in so that
    callers can build the ``options`` argument uniformly.
    """


@dataclass(frozen=True)
class ReconstructInput:
    """All data required to start a reconstruction: patterns, bad-pixel mask, and initial product.

    `background` and `flat_field` are measured detector maps the patterns have not
    been corrected for, on the pattern grid; each is None when absent or already
    applied. A flat field is the detector's response, so a model prediction is
    multiplied by it and the background added to compare with the patterns.
    """

    diffraction_patterns: DiffractionPatterns
    bad_pixels: BadPixels
    product: Product
    background: DetectorBackground | None = None
    flat_field: FlatField | None = None


class TrainingDataFileWriter(FileWriter[ReconstructInput]):
    """Plugin interface for writing what a trainable reconstructor learns from."""


@dataclass(frozen=True)
class ReconstructOutput:
    """Reconstruction result yielded at each iteration.

    ``progress_goal`` is what ``progress`` counts towards, reported by the run
    rather than predicted before it: a backend that resolves its own iteration
    count can only say so once it has started. Zero means unknown, which a
    progress bar reads as indeterminate.
    """

    product: Product
    progress: int = 0
    progress_goal: int = 0
    status: int = 0


class Reconstructor(ABC):
    """Abstract interface for iterative ptychography reconstruction algorithms.

    An implementation is picklable, and whatever runs :meth:`reconstruct` may do
    so in another process: the reconstructor and its arguments travel there,
    and only what the method yields comes back. So :meth:`reconstruct` leaves
    ``self`` unchanged and imports the framework it computes with inside the
    method, while every other method -- the options, and for a trainable
    reconstructor the model file handling -- imports no GPU framework at all.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def get_default_options(self) -> BaseModel:
        """Return this reconstructor's options, populated with their defaults.

        A description, not live state: plain pydantic carrying values, types,
        ranges, enums and nesting. A caller edits a copy and passes it back as
        ``options``; the settings layer generates live parameters from it, and
        its type is what an incoming ``options`` is validated against.

        An instance rather than the class, because a field declared with a
        ``default_factory`` reports its class-level default as undefined. Only
        an instance carries the value such a field actually starts at.

        Override when the backend declares its options.
        """
        return NoOptions()

    @abstractmethod
    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        pass


@dataclass(frozen=True)
class TrainOutput:
    """Training result yielded at each step: loss curves, progress, and status code.

    ``progress_goal`` is reported by the run, as for :class:`ReconstructOutput`.

    ``trained_model`` is the model the run wrote, in the form
    :meth:`TrainableReconstructor.load_model` accepts, set on the output that
    reports it. Training does not load it; whoever asked for the training does.
    """

    training_loss: Sequence[LossValue] = field(default_factory=list)
    validation_loss: Sequence[LossValue] = field(default_factory=list)
    progress: int = 0
    progress_goal: int = 0
    status: int = 0
    trained_model: Path | None = None


class TrainableReconstructor(Reconstructor):
    """Reconstructor that also supports ML model loading and training.

    :meth:`train` is held to what :class:`Reconstructor` asks of
    :meth:`~Reconstructor.reconstruct`, and so announces the model it wrote in
    :attr:`TrainOutput.trained_model` rather than loading it. :meth:`load_model`
    records what a later :meth:`~Reconstructor.reconstruct` reads, and does no
    more than that needs.
    """

    @abstractmethod
    def is_model_loaded(self) -> bool:
        pass

    @abstractmethod
    def get_model_file_description(self) -> str:
        """Return the human-readable model format name, without extensions."""
        pass

    @abstractmethod
    def get_model_file_extensions(self) -> tuple[str, ...]:
        """Return the model file extensions, each with a leading dot.

        A model is read from any of them and saved under the first.
        """
        pass

    @abstractmethod
    def load_model(self, file_path: Path) -> None:
        pass

    @abstractmethod
    def save_model(self, file_path: Path) -> None:
        """Write the currently-loaded model to file_path."""
        pass

    @abstractmethod
    def train(
        self, input_directory: Path, output_directory: Path, *, options: BaseModel
    ) -> Iterator[TrainOutput]:
        """Train from a StandardFileLayout in *input_directory*, writing to *output_directory*."""
        pass

    def supports_distributed_training(self) -> bool:
        """Whether one ``train`` call per rank of a distributed launch trains one model.

        True promises that, started once per rank by a launcher such as ``torchrun``, the
        backend joins a single process group from the environment it inherits and writes
        its outputs from global rank zero alone. False, the default, means each rank would
        train a model of its own into the same output directory.

        Override when the backend makes that promise.
        """
        return False


class ReconstructorLibrary(Iterable[Reconstructor], ABC):
    """Iterable collection of Reconstructor instances provided by a plugin library.

    A library does not name itself. What a person selecting an algorithm sees
    is the name it was registered under, which the plugin registry holds
    beside it -- so a library carries only what it needs to find and report
    its own package: ``distribution`` is what that package is called on PyPI,
    the name a version lives under, and ``requires`` lists the modules the
    library cannot work without.

    What a library offers is settled on first use rather than at
    construction, because registration happens in every process that loads
    plugins including the many that never reconstruct. However it turns out,
    it is announced: a library that builds reports the version it built
    against, one whose packages are absent says so, and one whose packages
    are present but will not import says *that*, as a warning carrying the
    traceback, since it is a fault rather than a choice.
    """

    def __init__(
        self, logger_name: str, distribution: str, *, requires: Sequence[str] = ()
    ) -> None:
        super().__init__()
        self._logger = logging.getLogger(logger_name)
        self._distribution = distribution
        self._requires = tuple(requires)
        self._reconstructors: list[Reconstructor] | None = None

    def get_logger(self) -> logging.Logger:
        return self._logger

    def is_available(self) -> bool:
        """Report whether every module this library needs can be located.

        Names are resolved without being imported, so asking costs nothing in
        a process that will never reconstruct. It is a precondition and not a
        promise: a module that resolves can still fail to import, which is
        why building is guarded separately.
        """
        return all(find_spec(module) is not None for module in self._requires)

    def get_version(self) -> str:
        """Return the installed version of the package this wraps, or 'unknown'.

        Which build produced a result is the first question asked when two
        machines disagree, so this is logged where a reconstruction's own log
        records it rather than only reported by a diagnostic command.
        """
        try:
            return version(self._distribution)
        except PackageNotFoundError:
            return 'unknown'

    def log_version(self) -> None:
        """Record the version through this library's own logger.

        Its own, not the module's: the logger named at construction is the one
        a task monitor attaches a handler to, so anything logged elsewhere
        never reaches whatever is watching the run.

        Reported under the distribution name, because a library is not told
        what it was registered as and this is the name whose version it is.
        """
        self._logger.info(f'{self._distribution} {self.get_version()}')

    def __iter__(self) -> Iterator[Reconstructor]:
        if self._reconstructors is None:
            self._reconstructors = self._enumerate()

        return iter(self._reconstructors)

    def _enumerate(self) -> list[Reconstructor]:
        """Build once, reporting either outcome through this library's logger."""
        if not self.is_available():
            self._logger.info(f'{self._distribution} not found.')
            return []

        try:
            reconstructors = list(self._build())
        except ImportError:
            # Locating a module does not execute it, so a package that is
            # present but unusable -- a dependency of its own missing, a
            # partial install -- is only discovered here. Said differently
            # from the branch above, and louder: not installing a backend is
            # a choice, while installing one that cannot import is a fault
            # nobody asked for, and a run that silently proceeds without it
            # is the failure worth reporting. The traceback goes with it as
            # the only thing that names what is actually broken.
            self._logger.warning(
                f'{self._distribution} is installed but failed to import; '
                'this backend is disabled.',
                exc_info=True,
            )
            return []

        self.log_version()
        return reconstructors

    @abstractmethod
    def _build(self) -> Iterable[Reconstructor]:
        """Create the reconstructors, importing whatever they need.

        Called at most once, and only after ``is_available`` has agreed, so
        an implementation imports its backend directly rather than guarding.
        """


class PositionIndexFilter(Enum):
    """Filter scan points by scan index.

    ``ODD`` and ``EVEN`` split a scan into two interleaved halves by index parity. How
    the halves sit on the sample follows from how the scan is numbered: a raster
    numbered by a running count is split into a checkerboard when a line holds an odd
    number of points, and into alternating columns when it holds an even number. Line-
    major indexes (``line * stride + column`` with an even stride) always split into
    columns, so each half keeps the full step along the lines and doubles it across them.
    """

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

    Positions whose index has no matching pattern naturally do not appear in the output.

    OPR weights carry one row per probe position, and are conditioned exactly as the
    coordinates are: filtered, averaged over duplicate indexes and interpolated at the
    pattern indexes. The output therefore holds one weight row per output pattern, in the
    same order as the output positions.

    Raises:
        ValueError: If the OPR weights do not have one row per probe position, if the
            index filter eliminates every pattern or position, or if no position index
            overlaps the pattern indexes.
    """
    pattern_indexes = assembled_data.get_indexes()
    valid_patterns = assembled_data.get_patterns()

    if pattern_indexes.size == 0:
        raise ValueError('Cannot prepare reconstruct input from empty diffraction dataset.')

    warn_if_scattering_geometry_unsupported(product)

    pattern_keep = numpy.fromiter(
        (index_filter(int(i)) for i in pattern_indexes),
        dtype=numpy.bool,
        count=pattern_indexes.size,
    )
    filtered_pattern_indexes = pattern_indexes[pattern_keep]
    filtered_pattern_offsets = numpy.flatnonzero(pattern_keep)

    n_positions = len(product.probe_positions)
    pos_indexes_all = product.probe_positions.get_indexes()
    pos_x_all = product.probe_positions.get_coordinates_x_m()
    pos_y_all = product.probe_positions.get_coordinates_y_m()

    pos_photon_counts_all = product.probe_positions.get_probe_photon_counts()
    opr_weights_all = product.probes.get_opr_weights_or_none()

    if opr_weights_all is not None and opr_weights_all.shape[0] != n_positions:
        raise ValueError(
            f'OPR weights have {opr_weights_all.shape[0]} rows but the product has'
            f' {n_positions} probe positions; each row belongs to one probe position.'
            ' Choose an OPR weight policy (Average or Reinitialize) to conform them.'
        )

    pos_keep = numpy.fromiter(
        (index_filter(int(i)) for i in pos_indexes_all),
        dtype=numpy.bool,
        count=n_positions,
    )
    pos_indexes = pos_indexes_all[pos_keep]
    pos_x = pos_x_all[pos_keep]
    pos_y = pos_y_all[pos_keep]
    pos_photon_counts = None if pos_photon_counts_all is None else pos_photon_counts_all[pos_keep]

    if filtered_pattern_indexes.size == 0 or pos_indexes.size == 0:
        raise ValueError('Index filter eliminated all pattern indexes and/or all position indexes.')

    # Duplicate position indexes average into one anchor; pattern indexes between
    # anchors are interpolated and those beyond them dropped.
    in_range_mask, coordinates = interpolate_at_indexes(
        pos_indexes, numpy.column_stack((pos_x, pos_y)), filtered_pattern_indexes
    )
    in_range_pattern_indexes = filtered_pattern_indexes[in_range_mask]
    in_range_pattern_offsets = filtered_pattern_offsets[in_range_mask]
    unique_pos_indexes = numpy.unique(pos_indexes)

    if in_range_pattern_indexes.size == 0:
        pat_lo = filtered_pattern_indexes[0]
        pat_hi = filtered_pattern_indexes[-1]
        raise ValueError(
            'No probe positions overlap the diffraction pattern indexes; '
            f'pattern indexes span [{pat_lo}, {pat_hi}] and position indexes '
            f'span [{unique_pos_indexes[0]}, {unique_pos_indexes[-1]}].'
        )

    # With a single anchor the range is one index wide, so every surviving pattern
    # index matches it exactly; the range trim enforces the "two anchors to
    # interpolate" rule with no separate guard.
    exact_match = numpy.isin(in_range_pattern_indexes, unique_pos_indexes)
    x_coords = coordinates[:, 0]
    y_coords = coordinates[:, 1]

    # Averaging duplicate-index photon counts uses the arithmetic mean because the
    # position readers store per-trigger integrated *counts* over equal-length trigger
    # windows -- each duplicate is an independent sample of the same per-trigger count.
    # A harmonic or geometric mean would only apply if a future reader supplied true
    # rates (Hz) over unequal integration windows. NaN placeholders (positions whose
    # reader did not supply a count) propagate through the mean, marking the anchor
    # invalid so any pattern index interpolated from it comes out NaN.
    photon_counts: RealArrayType | None = None
    if pos_photon_counts is not None:
        _, photon_counts = interpolate_at_indexes(
            pos_indexes, pos_photon_counts, filtered_pattern_indexes
        )

    probes = product.probes
    if opr_weights_all is not None:
        opr_weights = opr_weights_all[pos_keep]
        _, mean_weights = interpolate_at_indexes(pos_indexes, opr_weights, filtered_pattern_indexes)
        opr_weights_out = mean_weights.astype(opr_weights.dtype)

        probes = ProbeEnsemble(
            array=probes.get_array(),
            opr_weights=opr_weights_out,
            pixel_geometry=probes.get_pixel_geometry(),
        )

    n_averaged = pos_indexes.size - unique_pos_indexes.size
    n_interpolated = int((~exact_match).sum())
    logger.debug(
        f'prepare_reconstruct_input: matched {in_range_pattern_indexes.size} of '
        f'{filtered_pattern_indexes.size} pattern indexes; averaged {n_averaged} '
        f'duplicate positions; interpolated {n_interpolated} positions'
    )

    # The all-or-nothing invariant on ProbePositionSequence means photon_counts is either
    # None (no reader ever measured it) or a fully-populated finite array; interpolation
    # over a fully-populated anchor array preserves that. So the output list carries a
    # count on every point or on none of them, preserving the invariant for the trimmed
    # ProbePositionSequence built below.
    point_list = [
        ProbePosition(
            index=int(i),
            x_m=float(x),
            y_m=float(y),
            probe_photon_count=None if photon_counts is None else float(photon_counts[k]),
        )
        for k, (i, x, y) in enumerate(zip(in_range_pattern_indexes, x_coords, y_coords))
    ]
    patterns = valid_patterns[in_range_pattern_offsets]

    product = Product(
        metadata=product.metadata,
        probe_positions=ProbePositionSequence(point_list),
        probes=probes,
        object_=product.object_,
        losses=product.losses,
    )

    return ReconstructInput(
        patterns,
        assembled_data.get_bad_pixels(),
        product,
        background=assembled_data.get_background(),
        flat_field=assembled_data.get_flat_field(),
    )


def warn_if_scattering_geometry_unsupported(product: Product) -> None:
    """Log a warning for each way the product's geometry falls outside the exit-frame model.

    A transmission reconstructor reconstructs an off-axis product correctly only in the
    exit frame, which needs far field, a single-slice object and beams crossing the
    surface at a finite angle (:func:`ptychodus.api.scattering.geometry.find_scattering_issues`).
    The reconstruction still runs; the warnings say what it will get wrong. A forward
    geometry logs nothing.
    """
    geometry = product.metadata.scattering_geometry

    for issue in find_scattering_issues(
        geometry,
        far_field=product.metadata.far_field,
        num_object_layers=product.object_.num_layers,
    ):
        logger.warning('%s geometry %s: %s', geometry.regime, geometry, issue.describe())


def warn_if_propagation_regime_disagrees(
    product: Product,
    *,
    far_field_fresnel_number: float = 1.0,
    near_field_fresnel_number: float = 1.0,
) -> None:
    """Warn when a product's declared propagation regime contradicts its own geometry.

    The regime is a declaration, so nothing stops it disagreeing with the geometry it
    describes -- and the consequence is a reconstruction on the wrong sample-plane
    sampling, which fails as a bad image rather than as an error. The object-plane
    Fresnel number is the regime indicator: much less than one is far field, much
    greater is near field. A far-field declaration is judged at the detector distance,
    because its sampling does not depend on the magnification and a focus only curves
    the probe; a near-field one at the demagnified distance ``z_d / M``.

    Raising `far_field_fresnel_number` tolerates a more transitional geometry before
    objecting to a far-field declaration; lowering `near_field_fresnel_number` does the
    same for a near-field one. Between the two the geometry is transitional and neither
    declaration is questioned.

    Degenerate geometry -- no probe, zero wavelength, detector at the focus -- is
    reported by nothing and warns about nothing; this never raises.
    """
    metadata = product.metadata

    try:
        probe_geometry = product.probes.get_geometry()
        propagation_distance_m = (
            metadata.detector_distance_m
            if metadata.far_field
            else metadata.detector_distance_m / metadata.magnification
        )
        fresnel_number = compute_full_aperture_fresnel_number(
            probe_geometry.get_pixel_geometry(),
            ImageExtent(width_px=probe_geometry.width_px, height_px=probe_geometry.height_px),
            wavelength_m=metadata.photon_wavelength_m,
            propagation_distance_m=propagation_distance_m,
        )
    except (ValueError, ZeroDivisionError):
        logger.debug('Product geometry is degenerate; skipping the propagation-regime check.')
        return

    if metadata.far_field:
        if fresnel_number > far_field_fresnel_number:
            logger.warning(
                'Product "%s" declares far-field propagation, but its object-plane '
                'Fresnel number is %g, which indicates near field. The sample-plane '
                'sampling follows the declaration, so verify it before reconstructing.',
                metadata.name,
                fresnel_number,
            )
    elif fresnel_number < near_field_fresnel_number:
        logger.warning(
            'Product "%s" declares near-field propagation, but its object-plane '
            'Fresnel number is %g, which indicates far field. The sample-plane '
            'sampling follows the declaration, so verify it before reconstructing.',
            metadata.name,
            fresnel_number,
        )


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

        obj_ramp = self._phase_ramp_grid(obj_coords.x_m, obj_coords.y_m)
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

        probe_ramp = self._phase_ramp_grid(probe_coords.x_m, probe_coords.y_m)
        probe_correction = self.object_scale_factor * numpy.exp(
            1j * (self.phase_offset_rad + probe_ramp)
        )
        opr_weights = probes.get_opr_weights_or_none()
        standardized_probes = ProbeEnsemble(
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
