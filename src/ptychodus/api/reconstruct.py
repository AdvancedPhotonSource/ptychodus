"""Reconstructor interfaces, I/O data containers, and product standardization.

Also home to :class:`SubprocessReconstructor`, the adapter that runs a backend
in a spawned child so the parent process never acquires a GPU context. It sits
beside the interface it implements because it implements the whole of it, and
the only thing it adds is where the work happens.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import auto, Enum
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from pathlib import Path
from typing import Any
import logging
import pickle
import shutil

import numpy
from pydantic import BaseModel

from .assemble import AssembledDiffractionData
from .diffraction import BadPixels, DiffractionPatterns
from .geometry import ImageExtent
from .object import Object
from .probe import ProbeEnsemble
from .probe_positions import ProbePositionSequence, ProbePosition
from .product import LossValue, Product
from .propagate import compute_full_aperture_fresnel_number
from .spawn import run_subprocess
from .typing import RealArrayType

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
    """All data required to start a reconstruction: patterns, bad-pixel mask, and initial product."""

    diffraction_patterns: DiffractionPatterns
    bad_pixels: BadPixels
    product: Product


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
    """Abstract interface for iterative ptychography reconstruction algorithms."""

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
    """

    training_loss: Sequence[LossValue] = field(default_factory=list)
    validation_loss: Sequence[LossValue] = field(default_factory=list)
    progress: int = 0
    progress_goal: int = 0
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
    def load_model(self, file_path: Path) -> None:
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
    def export_training_data(
        self, file_path: Path, input_: ReconstructInput, *, options: BaseModel
    ) -> None:
        pass

    @abstractmethod
    def train(
        self, input_directory: Path, output_directory: Path, *, options: BaseModel
    ) -> Iterator[TrainOutput]:
        """Train from a StandardFileLayout in *input_directory*, writing to *output_directory*."""
        pass


class NullReconstructor(Reconstructor):
    """Stands in for a selection that resolves to no reconstructor at all.

    A plain :class:`Reconstructor` rather than a trainable one, deliberately:
    there is nothing here to train, so a test for trainability should say so
    rather than find a stub. Reconstructing yields nothing.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        yield from ()


# Tags emitted by child entry points and consumed by this adapter.
TAG_OUTPUT = 'output'  # payload = pickle.dumps(ReconstructOutput)
TAG_TRAIN_OUTPUT = 'train_output'  # payload = pickle.dumps(TrainOutput)
TAG_MODEL_SAVED = 'model_saved'  # payload = str path where child wrote checkpoint
TAG_SETTINGS_SYNC = 'settings_sync'  # payload = {group_name: {param_name: value_str}}


OPERATION_RECONSTRUCT = 'reconstruct'
OPERATION_TRAIN = 'train'


@dataclass(frozen=True)
class SubprocessPayload:
    """What a child is handed, whichever backend and whichever operation.

    One envelope rather than a pair of dataclasses per backend, so every
    child reads the same shape and a new backend states only what goes in
    ``options``. Which fields are filled follows from ``operation``:
    reconstructing supplies ``input_`` and, where a backend loads one,
    ``model_path``; training supplies the two directories.

    ``options`` is the subset of the declaration this operation's child
    reads, in the form that child consumes -- usually a mapping, since
    dumping only what is read keeps a reconstruct payload free of
    training-only values. A declaration that cannot dump itself to a mapping
    without changing what it holds travels as the object it built instead;
    the payload is pickled either way, so nothing is gained by a round trip
    that would coerce the values on the way through.
    """

    operation: str
    reconstructor: str
    options: Any
    input_: ReconstructInput | None = None
    input_directory: Path | None = None
    output_directory: Path | None = None
    model_path: Path | None = None

    def require_input(self) -> ReconstructInput:
        """The data to reconstruct, or a complaint naming what was missing.

        A child is spawned for one operation and reads the fields that
        operation fills. Asking rather than assuming turns a payload built
        wrong into one message here, instead of an attribute error on None
        several frames into a GPU framework.
        """
        if self.input_ is None:
            raise ValueError(f'{self.reconstructor}: {self.operation} payload carries no input.')

        return self.input_

    def require_model_path(self) -> Path:
        """The model to load, or a complaint naming what was missing."""
        if self.model_path is None:
            raise ValueError(
                f'{self.reconstructor}: {self.operation} payload carries no model path.'
            )

        return self.model_path

    def require_directories(self) -> tuple[Path, Path]:
        """The directories to read from and write to, or a complaint."""
        if self.input_directory is None or self.output_directory is None:
            raise ValueError(
                f'{self.reconstructor}: {self.operation} payload carries no directories.'
            )

        return self.input_directory, self.output_directory


# Type aliases for the callbacks each backend supplies.
DumpReconstructOptions = Callable[[ReconstructInput, BaseModel], Any]
"""(reconstruct_input, options) -> the subset of the declaration the child reads."""

DumpTrainOptions = Callable[[BaseModel], Any]
"""(options) -> the subset of the declaration the training child reads."""

ResolveModelPath = Callable[['Path | None'], Path]
"""(path recorded by load_model) -> the path the child loads from.

Runs parent-side, before the child is spawned, so a backend that unpacks an
archive or picks a file out of a bundle directory does it once and complains
here rather than in the child when there is nothing to load.
"""

ExportTrainingData = Callable[[Path, ReconstructInput, BaseModel], None]
"""(file_path, reconstruct_input, options) -> None. Runs parent-side; must not touch GPU."""

SaveModel = Callable[[Path, Path], None]
"""(source_path_loaded_by_child, destination_path) -> None. Runs parent-side."""

ApplySettingsSync = Callable[[dict[str, dict[str, str]]], None]
"""Optional handler for ('settings_sync', dict) messages the child emits."""

DefaultOptions = Callable[[], BaseModel]
"""() -> a fresh object describing what this backend accepts, at its defaults."""

DeviceEnvironment = Callable[[], Mapping[str, str]]
"""() -> the environment entries restricting a child to the chosen devices."""


class SubprocessReconstructor(TrainableReconstructor):
    """Parent-side adapter that runs any reconstructor in a fresh subprocess.

    Wraps any backend reachable as a child-side entry point, so a caller can
    dispatch to it without importing a GPU framework. When ``is_trainable`` is
    False the trainable methods raise :class:`NotImplementedError`, so the
    object is still a single kind that needs no special-casing in a list.

    The parent-side object holds no GPU state. It stores:
    - the child entry-point strings (dotted module paths),
    - backend-supplied callbacks that dump options and export training data,
    - the loaded-model path (as recorded by ``load_model`` /
      remembered from the last ``train`` call).

    Each ``reconstruct()`` / ``train()`` call spawns exactly one child; the
    child dies before the call returns, and the iterator yields the same
    per-iteration output an in-process backend would.

    **Trainable-model lifecycle.** ``load_model`` records the path and nothing
    else -- the fresh inference child does the real load, once per call.
    ``save_model`` copies or archives that path to the destination through a
    backend-supplied callback, so a backend persisting one file and a backend
    persisting a whole bundle directory share this adapter unchanged.
    ``export_training_data`` runs in the parent, which is safe only because
    every implementation of it is pure numpy.
    """

    def __init__(
        self,
        *,
        name: str,
        reconstruct_entry_point: str,
        dump_reconstruct_options: DumpReconstructOptions,
        default_options: DefaultOptions = NoOptions,
        resolve_model_path: ResolveModelPath | None = None,
        is_trainable: bool = False,
        train_entry_point: str | None = None,
        dump_train_options: DumpTrainOptions | None = None,
        model_file_filter: str = '',
        model_file_extension: str = '',
        training_data_file_filter: str = '',
        export_training_data: ExportTrainingData | None = None,
        save_model: SaveModel | None = None,
        apply_settings_sync: ApplySettingsSync | None = None,
        terminate_grace_sec: float = 10.0,
    ) -> None:
        super().__init__()
        self._name = name
        self._reconstruct_entry_point = reconstruct_entry_point
        self._dump_reconstruct_options = dump_reconstruct_options
        self._default_options = default_options
        self._resolve_model_path = resolve_model_path
        self._is_trainable = is_trainable
        self._train_entry_point = train_entry_point
        self._dump_train_options = dump_train_options
        self._model_file_filter = model_file_filter
        self._model_file_extension = model_file_extension
        self._training_data_file_filter = training_data_file_filter
        self._export_training_data = export_training_data
        self._save_model_fn = save_model
        self._apply_settings_sync = apply_settings_sync
        self._terminate_grace_sec = terminate_grace_sec

        self._loaded_model_path: Path | None = None
        # Where the work runs, which is the spawning process's to decide and
        # not the backend's. Replaced after construction by whatever owns the
        # selection; until then a child sees every device this process does.
        self.device_environment: DeviceEnvironment = dict

    @property
    def name(self) -> str:
        return self._name

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    def get_default_options(self) -> BaseModel:
        # A fresh object per call, so a caller editing what it was handed
        # cannot change what the next caller is handed.
        return self._default_options()

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        payload = SubprocessPayload(
            operation=OPERATION_RECONSTRUCT,
            reconstructor=self._name,
            options=self._dump_reconstruct_options(input_, options),
            input_=input_,
            model_path=(
                self._loaded_model_path
                if self._resolve_model_path is None
                else self._resolve_model_path(self._loaded_model_path)
            ),
        )

        with run_subprocess(
            self._reconstruct_entry_point,
            payload,
            terminate_grace_sec=self._terminate_grace_sec,
            environment=self.device_environment(),
        ) as events:
            for event in events:
                tag = event[0]
                if tag == TAG_OUTPUT:
                    yield pickle.loads(event[1])
                elif tag == TAG_SETTINGS_SYNC:
                    if self._apply_settings_sync is not None:
                        try:
                            self._apply_settings_sync(event[1])
                        except Exception:
                            logger.exception('Failed to apply settings sync from subprocess.')
                else:
                    logger.debug(
                        f'{self._name}: dropping unrecognized subprocess message tag {tag!r}'
                    )

    def is_model_loaded(self) -> bool:
        return self._loaded_model_path is not None

    def get_model_file_filter(self) -> str:
        return self._model_file_filter

    def load_model(self, file_path: Path) -> None:
        # Parent-side: just remember the path. The child does the actual load
        # (which touches GPU frameworks) per reconstruct/train call.
        self._loaded_model_path = file_path

    def get_model_file_extension(self) -> str:
        return self._model_file_extension

    def save_model(self, file_path: Path) -> None:
        if self._loaded_model_path is None:
            raise RuntimeError(
                f'Cannot save {self._name} model: no model has been loaded or trained.'
            )
        if self._save_model_fn is not None:
            self._save_model_fn(self._loaded_model_path, file_path)
        else:
            shutil.copyfile(self._loaded_model_path, file_path)

    def get_training_data_file_filter(self) -> str:
        return self._training_data_file_filter

    def export_training_data(
        self, file_path: Path, input_: ReconstructInput, *, options: BaseModel
    ) -> None:
        if self._export_training_data is None:
            raise NotImplementedError(f'{self._name} does not support exporting training data.')
        self._export_training_data(file_path, input_, options)

    def train(
        self, input_directory: Path, output_directory: Path, *, options: BaseModel
    ) -> Iterator[TrainOutput]:
        if not self._is_trainable:
            raise NotImplementedError(f'{self._name} does not support training.')
        if self._train_entry_point is None or self._dump_train_options is None:
            raise NotImplementedError(f'{self._name} does not support training.')

        payload = SubprocessPayload(
            operation=OPERATION_TRAIN,
            reconstructor=self._name,
            options=self._dump_train_options(options),
            input_directory=input_directory,
            output_directory=output_directory,
        )

        with run_subprocess(
            self._train_entry_point,
            payload,
            terminate_grace_sec=self._terminate_grace_sec,
            environment=self.device_environment(),
        ) as events:
            for event in events:
                tag = event[0]
                if tag == TAG_TRAIN_OUTPUT:
                    yield pickle.loads(event[1])
                elif tag == TAG_MODEL_SAVED:
                    self._loaded_model_path = Path(event[1])
                elif tag == TAG_SETTINGS_SYNC:
                    if self._apply_settings_sync is not None:
                        try:
                            self._apply_settings_sync(event[1])
                        except Exception:
                            logger.exception('Failed to apply settings sync from subprocess.')
                else:
                    logger.debug(
                        f'{self._name}: dropping unrecognized subprocess message tag {tag!r}'
                    )


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
    plugins including the many that never reconstruct. Whichever way it
    turns out, it is announced the same way: a library whose packages are
    missing offers nothing and says so, and one that builds reports the
    version it built against.
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
            # partial install -- is only discovered here. The traceback goes
            # with it, because this is the one case the caller cannot tell
            # apart from simply not having installed the backend.
            self._logger.info(f'{self._distribution} not found.', exc_info=True)
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
    pos_indexes_all = product.probe_positions.get_indexes()
    pos_x_all = product.probe_positions.get_coordinates_x_m()
    pos_y_all = product.probe_positions.get_coordinates_y_m()

    pos_photon_counts_all = product.probe_positions.get_probe_photon_counts()

    pos_keep = numpy.fromiter(
        (index_filter(int(i)) for i in pos_indexes_all),
        dtype=numpy.bool_,
        count=n_positions,
    )
    pos_indexes = pos_indexes_all[pos_keep]
    pos_x = pos_x_all[pos_keep]
    pos_y = pos_y_all[pos_keep]
    pos_photon_counts = None if pos_photon_counts_all is None else pos_photon_counts_all[pos_keep]

    if filtered_pattern_indexes.size == 0 or pos_indexes.size == 0:
        raise ValueError('Index filter eliminated all pattern indexes and/or all position indexes.')

    # Average duplicate position indexes. numpy.unique sorts ascending, so
    # the resulting (unique_pos_indexes, mean_x, mean_y) triple is the
    # canonical interpolation anchor set.
    unique_pos_indexes, inverse = numpy.unique(pos_indexes, return_inverse=True)
    counts = numpy.bincount(inverse)
    mean_x = numpy.bincount(inverse, weights=pos_x) / counts
    mean_y = numpy.bincount(inverse, weights=pos_y) / counts

    # Averaging duplicate-index photon counts uses the arithmetic mean because the
    # position readers store per-trigger integrated *counts* over equal-length trigger
    # windows -- each duplicate is an independent sample of the same per-trigger count.
    # A harmonic or geometric mean would only apply if a future reader supplied true
    # rates (Hz) over unequal integration windows. NaN placeholders (positions whose
    # reader did not supply a count) propagate through bincount -> mean, marking the
    # anchor invalid so the subsequent interp yields NaN for any pattern index derived
    # from it.
    mean_photon_counts: numpy.ndarray | None = None
    if pos_photon_counts is not None:
        mean_photon_counts = numpy.bincount(inverse, weights=pos_photon_counts) / counts

    lo = unique_pos_indexes[0]
    hi = unique_pos_indexes[-1]
    in_range_mask = (filtered_pattern_indexes >= lo) & (filtered_pattern_indexes <= hi)
    in_range_pattern_indexes = filtered_pattern_indexes[in_range_mask]
    in_range_pattern_offsets = filtered_pattern_offsets[in_range_mask]

    if in_range_pattern_indexes.size == 0:
        pat_lo = filtered_pattern_indexes[0]
        pat_hi = filtered_pattern_indexes[-1]
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
    photon_counts: numpy.ndarray | None = None
    if mean_photon_counts is not None:
        photon_counts = numpy.interp(
            in_range_pattern_indexes, unique_pos_indexes, mean_photon_counts
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
        # OPR weight rows are still array-indexed by the original scan
        # ordering, not by the merged-and-interpolated subset. Correct only
        # when matched positions are contiguous from zero with no
        # interpolation; latent misalignment otherwise.
        probes=product.probes,
        object_=product.object_,
        losses=product.losses,
    )

    return ReconstructInput(patterns, assembled_data.get_bad_pixels(), product)


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
    greater is near field.

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
        propagation_distance_m = metadata.detector_distance_m / metadata.magnification
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
