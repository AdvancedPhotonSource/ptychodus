"""Shared reconstruction tail for the ``cli/reconstruct_*.py`` commands.

Every driver assembles diffraction data and builds an initial
:class:`~ptychodus.api.product.Product` -- that part differs per instrument. From there on,
every command resolves its options, saves ``diffraction.h5``, runs the reconstruction to
convergence, and writes the standard-layout outputs identically. :func:`read_reconstruction`,
:func:`save_assembled_diffraction` and :func:`run_reconstruction` are that common tail,
extracted so it is written and tested once rather than duplicated in eleven files.

A run is any registered reconstructor, named by an options document (library, algorithm and
the library's own options; see :mod:`ptychodus.api.options`) and run in this process through
its plugin. Nothing here imports a library, so a command built on it works with whichever
libraries are installed; pty-chi is only the fallback when no options are given.

Both writers write under :func:`is_main_process`, so a multi-process launch produces one copy of
each file instead of having every rank truncate the same paths concurrently.

Cancellation lives here for the same reason. A reconstruction has no interruption point inside
the library, so a signal cannot stop it where it lands;
:func:`ptychodus.cli.install_signal_handlers` instead raises a flag that
:func:`run_reconstruction` reads at each output, and a partially converged ``product.h5`` is
left behind. Under a multi-process launch that flag is agreed across ranks
before it is acted on, because a rank that left the loop alone would hang its peers at their
next collective.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, Final, Literal

import numpy
from pydantic import BaseModel

from ptychodus.api.accumulate import (
    AssemblyMomentAccumulator,
    IncidentBeamEstimate,
    estimate_incident_beam,
)
from ptychodus.api.assemble import (
    AssembledDiffractionData,
    DetectorCorrections,
    DiffractionSummary,
    suggest_crop_region,
    summarize_dataset,
)
from ptychodus.api.diffraction import (
    BadPixels,
    BeamCenter,
    CropRegion,
    DiffractionDataset,
    DiffractionPattern,
    DiffractionPatternDType,
)
from ptychodus.api.geometry import ImageExtent
from ptychodus.api.io import StandardFileLayout, save_diffraction_data, save_product
from ptychodus.api.object import ObjectGeometry
from ptychodus.api.options import (
    OptionsError,
    apply_options_document,
    dump_options_document,
    parse_options_document,
)
from ptychodus.api.parallax import estimate_defocus_from_assembled_data
from ptychodus.api.plugins import PluginCatalog
from ptychodus.api.preprocess.diffraction import FilterValuesStep, estimate_beam_center
from ptychodus.api.probe import ProbeGeometry
from ptychodus.api.probe_positions import ProbePositionSequence
from ptychodus.api.product import Product
from ptychodus.api.reconstruct import (
    PositionIndexFilter,
    ReconstructInput,
    ReconstructOutput,
    Reconstructor,
    ReconstructorLibrary,
    TrainableReconstructor,
    prepare_reconstruct_input,
)
from ptychodus.cli import CancellationToken
from ptychodus.cli._launch import environment_rank
from ptychodus.cli._options import format_options_document, resolve_reconstructor

if TYPE_CHECKING:
    from ptychodus.plugins.ptychi.options import PtyChiOptions

__all__ = [
    'ResolvedReconstruction',
    'add_options_argument',
    'add_print_options_argument',
    'check_ptychi_options',
    'default_reconstruction',
    'estimate_dataset_beam_center',
    'estimate_incident_beam_from_assembled_data',
    'find_options_document',
    'invalid_count_threshold',
    'is_main_process',
    'load_trained_model',
    'pair_patterns_with_positions',
    'parse_reconstruction',
    'process_rank',
    'ptychi_reconstruction',
    'read_reconstruction',
    'require_ptychi_options',
    'resolve_crop_region',
    'resolve_ptychi_reconstruction',
    'resolve_quantity',
    'run_reconstruction',
    'save_assembled_diffraction',
    'save_reconstruct_options',
]

logger = logging.getLogger(__name__)


def _group_rank() -> int:
    """This process's rank in the torch process group, or 0 where there is none.

    Read off torch only if something has imported it: a library that runs ranks brings
    ``torch.distributed`` in, and a run that never imports torch has no process group.
    """
    dist = sys.modules.get('torch.distributed')

    if dist is None or not dist.is_available() or not dist.is_initialized():
        return 0

    return int(dist.get_rank())


def process_rank() -> int:
    """This process's rank, as either the process group or the launcher reports it.

    Two independent sources are consulted because neither alone covers the whole run. The
    torch process group does not exist until the library builds it, so until then every
    process reports rank 0 and only the environment tells them apart; a group created
    programmatically rather than by a launcher, in turn, exports nothing. A run under
    neither -- an ordinary single-process invocation -- is rank 0 by both, which is what
    makes this invisible outside a multi-process launch.
    """
    return _group_rank() or environment_rank()


def is_main_process() -> bool:
    """Whether this process is the one that writes a run's output files.

    Every rank of a multi-process reconstruction runs the same work and ends up holding
    the same product, so each output path would otherwise be written concurrently by all
    of them, every write truncating.
    """
    return process_rank() == 0


def _agree_on_cancellation(token: CancellationToken | None) -> bool:
    """Whether every rank stops after this sync point, or none of them does.

    Each rank catches its own signals, and a launcher does not deliver them to every rank at
    the same instant, so two ranks can read their own flags on opposite sides of one sync
    point. The one that left the loop would then never reach the next collective and the
    others would block there forever -- a cancellation that hangs the job instead of ending
    it. Reducing the flag with MAX makes one rank's request everyone's.

    The reduction is itself a collective, so it is only correct because every rank reaches
    this point the same number of times: a library that runs ranks yields in lockstep. With
    no process group -- an ordinary single-process run, and every rank before the library
    creates one -- there is nobody to agree with and the local flag stands.
    """
    if token is None:
        return False

    is_cancelled = token.is_cancelled
    dist = sys.modules.get('torch.distributed')

    if dist is None or not dist.is_available() or not dist.is_initialized():
        return is_cancelled

    # Present in sys.modules, so importing it is a lookup rather than a load.
    import torch

    # NCCL reduces on the GPU and raises on a CPU tensor, so the device follows the
    # process group's backend.
    # On the GPU it must be the one this rank's communicator was built on, which is the
    # default device a library sets per rank rather than torch's current device: that stays
    # cuda:0 on every rank, and a flag there would have one rank build a second communicator
    # its peers never join, hanging both.
    device = torch.device('cpu')

    if dist.get_backend() == 'nccl':
        device = torch.get_default_device()

        if device.type != 'cuda':
            device = torch.device('cuda', torch.cuda.current_device())

    flag = torch.tensor([int(is_cancelled)], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag[0])


def resolve_quantity(
    logger: logging.Logger,
    quantity: str,
    override: float | None,
    from_file: float | None,
    fallback: float | None,
    flag: str,
    *,
    level: int = logging.INFO,
) -> float:
    """Pick a geometry value from the command line, the file, or a fallback, and say which.

    Reporting the source is the point: a run that silently substituted a built-in default
    for a value the instrument actually recorded would be indistinguishable from one that
    read it. Lower `level` where the provenance would otherwise be repeated per scan across
    a long batch; the fallback warning is unaffected, since a guessed value is worth saying
    out loud however many times it happens.
    """
    if override is not None:
        logger.log(level, '%s: %g (from %s)', quantity, override, flag)
        return override

    if from_file is not None:
        logger.log(level, '%s: %g (from the diffraction file)', quantity, from_file)
        return from_file

    if fallback is not None:
        logger.warning(
            '%s: %g (built-in default for this instrument; the file recorded none). '
            'Pass %s if this scan differs.',
            quantity,
            fallback,
            flag,
        )
        return fallback

    raise ValueError(f'{quantity} is not in the file and has no default; pass {flag}.')


def resolve_flag(
    logger: logging.Logger,
    quantity: str,
    override: bool | None,
    from_file: bool | None,
    fallback: bool,
    flag: str,
    *,
    level: int = logging.INFO,
) -> bool:
    """Pick a boolean from the command line, the file, or a fallback, and say which.

    The boolean sibling of :func:`resolve_quantity`, with the same precedence and the
    same reason for reporting it: a run that guessed the propagation regime must stay
    distinguishable from one that read it off the instrument. Unlike a geometry
    quantity there is always a fallback, so this never raises.
    """
    if override is not None:
        logger.log(level, '%s: %s (from %s)', quantity, override, flag)
        return override

    if from_file is not None:
        logger.log(level, '%s: %s (from the diffraction file)', quantity, from_file)
        return from_file

    logger.log(
        level,
        '%s: %s (built-in default for this instrument; the file recorded none). '
        'Pass %s if this scan differs.',
        quantity,
        fallback,
        flag,
    )
    return fallback


def resolve_stated_focus_object_distance_m(
    logger: logging.Logger,
    override: float | None,
    from_file: float | None,
    *,
    source: str = '--focus-object-distance-m',
    level: int = logging.INFO,
) -> float | None:
    """The signed focus position the user or the data file states, or None.

    The focus is the one statement of the defocus every model probe is built at, its
    negation, and in near field it sets the magnification besides. No instrument
    presets one: the explicit *override*, named *source* in the log, wins over the
    file's record, and with neither the caller estimates it or asks for it.
    """
    if override is not None:
        logger.log(level, 'Focus-object distance (m): %g (from %s)', override, source)
        return override

    if from_file is not None:
        logger.log(level, 'Focus-object distance (m): %g (from the diffraction file)', from_file)
        return from_file

    return None


def require_near_field_focus_object_distance_m(stated: float | None, *, source: str) -> float:
    """The stated focus position, which near field cannot do without.

    The magnification follows from it and no parallax estimate answers in near field,
    so a missing one is an error rather than a parallel beam assumed for the user.

    Raises:
        ValueError: if *stated* is None; the message names *source*.
    """
    if stated is None:
        raise ValueError(
            'Near field needs the focus position, which the data cannot estimate; set '
            f'{source} (0 for a parallel beam).'
        )

    return stated


def estimate_focus_object_distance_m(
    logger: logging.Logger,
    assembled_data: AssembledDiffractionData,
    positions: ProbePositionSequence,
    object_geometry: ObjectGeometry,
    probe_geometry: ProbeGeometry,
    *,
    photon_wavelength_m: float,
    source: str,
    i0_photons_per_count: float = 0.0,
    bin_size_px: int = 4,
    minimum_significance: float = 3.0,
    minimum_defocus_dof: float = 1.0,
) -> float:
    """The focus position the far-field parallax measures, or zero when it resolves none.

    One binned pass over the patterns (see
    :func:`ptychodus.api.parallax.estimate_defocus_from_assembled_data`). The estimate is
    applied only when it reaches *minimum_significance* standard errors and
    *minimum_defocus_dof* depths of focus; otherwise the focus is zero -- indistinguishable
    from the focus at those thresholds -- and a warning names *source*, where a value can
    be stated instead.
    """
    remedy = f'Set {source} to state the focus.'

    try:
        estimate = estimate_defocus_from_assembled_data(
            assembled_data,
            positions,
            object_geometry,
            probe_geometry,
            photon_wavelength_m=photon_wavelength_m,
            i0_photons_per_count=i0_photons_per_count,
            bin_size_px=bin_size_px,
        )
    except ValueError as exc:
        logger.warning('No parallax defocus estimate (%s); the focus is zero. %s', exc, remedy)
        return 0.0

    if estimate is None:
        logger.warning('The parallax gives no defocus estimate; the focus is zero. %s', remedy)
        return 0.0

    logger.info(
        'Parallax defocus (m): %g +/- %g, Fresnel number %.3g',
        estimate.defocus_distance_m,
        estimate.standard_error_m,
        estimate.fresnel_number,
    )

    if estimate.is_resolved(
        minimum_significance=minimum_significance, minimum_defocus_dof=minimum_defocus_dof
    ):
        focus_object_distance_m = -estimate.defocus_distance_m
        logger.info(
            'Focus-object distance (m): %g (from the parallax estimate)', focus_object_distance_m
        )
        return focus_object_distance_m

    logger.warning(
        'The parallax defocus is not resolved at %g standard errors and %g depths of focus; '
        'the focus is zero. %s',
        minimum_significance,
        minimum_defocus_dof,
        remedy,
    )
    return 0.0


def _estimate_crop_region(
    logger: logging.Logger,
    mean_pattern: DiffractionPattern,
    beam_center: BeamCenter,
    detector_extent: ImageExtent,
    bad_pixels: BadPixels,
    value_filter: FilterValuesStep | None,
    corrections: DetectorCorrections | None,
) -> CropRegion:
    """The power-of-two square about `beam_center` holding the signal in `mean_pattern`.

    Raises ``ValueError`` as :func:`suggest_crop_region` does.
    """
    suggestion = suggest_crop_region(
        mean_pattern,
        beam_center,
        detector_extent,
        bad_pixels=bad_pixels,
        value_filter=value_filter,
        corrections=corrections,
    )
    signal = suggestion.signal
    region = suggestion.region
    logger.info(
        'Signal radius: %d px (ring mean above background %g by more than %g)',
        signal.radius_px,
        signal.noise_floor.median,
        signal.threshold,
    )

    if suggestion.is_capped:
        logger.warning(
            'The signal reaches %d px from the beam center, past the largest power-of-two '
            'square that fits on the detector; cropping to %d px',
            signal.radius_px,
            region.width_px,
        )
    else:
        logger.info('Crop extent: %d px from the signal radius', region.width_px)

    return region


def estimate_dataset_beam_center(
    raw_dataset: DiffractionDataset,
    *,
    bad_pixels: BadPixels | None,
    value_filter: FilterValuesStep | None,
) -> tuple[BeamCenter, DiffractionSummary]:
    """Estimate the beam center from the mean of every pattern, and return that summary too.

    Reads every pattern. :func:`summarize_dataset` inpaints the bad pixels that
    :func:`estimate_beam_center` requires the caller to have handled, and `value_filter` cuts
    the mean as assembly cuts each pattern, so invalid-pixel markers cannot pull the estimate.
    """
    summary = summarize_dataset(raw_dataset, bad_pixels=bad_pixels)
    mean_pattern = summary.mean_pattern
    beam_center = estimate_beam_center(
        mean_pattern if value_filter is None else value_filter.apply(mean_pattern)
    )
    return beam_center, summary


def resolve_crop_region(
    logger: logging.Logger,
    raw_dataset: DiffractionDataset,
    *,
    crop_extent_px: int | Literal['auto', 'full'] | None,
    beam_center_x_px: int | None,
    beam_center_y_px: int | None,
    bad_pixels: BadPixels | None,
    value_filter: FilterValuesStep | None,
    corrections: DetectorCorrections | None = None,
    warm_start_extent: ImageExtent | None = None,
    subject: str = 'dataset',
) -> CropRegion | None:
    """The detector region to read, or None to read the whole frame.

    `crop_extent_px` is a square side in pixels, ``auto`` to estimate one, ``full`` to read
    the whole frame, or None when none was asked for. None takes `warm_start_extent`, the
    extent of an initial probe read from a file, which the patterns must match; without one
    it is ``auto``. Every extent but ``full`` is centered on the beam center, taken from the
    arguments, then the file, then an estimate from the data.

    An ``auto`` extent is the smallest power-of-two square about the beam center that holds
    the signal above the noise floor of the mean pattern, capped at the largest that fits.
    The mean pattern is cut by `value_filter` and corrected by `corrections` first, as the
    assembled patterns are; one summary serves both that and any beam-center estimate. When
    no extent can be estimated, an explicit ``auto`` is an error, while an omitted extent
    falls back to the whole frame with a warning, so a run that asked for nothing still runs.

    `subject` names what the beam center is estimated over, for the two log lines that say
    so: a caller that settles one region for a whole run of scans is not summarizing the
    same thing as a caller handling a single dataset.
    """
    if crop_extent_px == 'full':
        logger.info('Reading the whole detector frame')
        return None

    metadata = raw_dataset.get_metadata()
    summary: DiffractionSummary | None = None

    # Beam center precedence: command line, then the file, then an estimate from the data.
    # The estimate is logged as such -- a wrong center crops the wrong part of the
    # detector, and nothing downstream would reveal it.
    if beam_center_x_px is not None and beam_center_y_px is not None:
        beam_center = BeamCenter(x_px=beam_center_x_px, y_px=beam_center_y_px)
        center_source = '--beam-center-{x,y}-px'
    elif metadata.beam_center is not None:
        beam_center = metadata.beam_center
        center_source = 'the diffraction file'
    else:
        logger.info('Summarizing the %s to estimate the beam center', subject)
        beam_center, summary = estimate_dataset_beam_center(
            raw_dataset, bad_pixels=bad_pixels, value_filter=value_filter
        )
        center_source = f'an estimate over the whole {subject}'

    logger.info('Beam center: (%d, %d) from %s', beam_center.x_px, beam_center.y_px, center_source)

    if crop_extent_px is None and warm_start_extent is not None:
        extent = warm_start_extent
        logger.info(
            'Crop extent: %dx%d px from the initial probe file',
            extent.width_px,
            extent.height_px,
        )
    elif crop_extent_px is None or crop_extent_px == 'auto':
        if summary is None:
            logger.info('Summarizing the %s to estimate the crop extent', subject)
            summary = summarize_dataset(raw_dataset, bad_pixels=bad_pixels)

        try:
            region = _estimate_crop_region(
                logger,
                summary.mean_pattern,
                beam_center,
                metadata.detector_extent,
                raw_dataset.get_bad_pixels() if bad_pixels is None else bad_pixels,
                value_filter,
                corrections,
            )
        except ValueError as exc:
            hint = 'Give --crop-extent-px a size in pixels, or "full" to read the whole frame.'

            if crop_extent_px == 'auto':
                raise ValueError(f'{exc} {hint}') from exc

            logger.warning('Reading the whole detector frame: %s %s', exc, hint)
            return None

        extent = ImageExtent(width_px=region.width_px, height_px=region.height_px)
    else:
        extent = ImageExtent(width_px=crop_extent_px, height_px=crop_extent_px)

    if extent == metadata.detector_extent:
        return None

    region = CropRegion.from_center_extent(beam_center, extent)

    # from_center_extent does not clip. Silently clamping would quietly reconstruct a
    # different region than asked for, so an overhanging crop is an error.
    if region.clamp_to_detector_extent(metadata.detector_extent) != region:
        raise ValueError(
            f'A {extent.width_px}x{extent.height_px}px crop about ({beam_center.x_px}, '
            f'{beam_center.y_px}) runs off the '
            f'{metadata.detector_extent.width_px}x{metadata.detector_extent.height_px} '
            f'detector (x={region.x_range} y={region.y_range}). '
            'Give a smaller --crop-extent-px or an explicit beam center.'
        )

    logger.info('Cropping to x=%s y=%s', region.x_range, region.y_range)
    return region


def invalid_count_threshold(
    dtype: DiffractionPatternDType, invalid_counts: Mapping[str, int] | None = None
) -> int | None:
    """Count at and above which a pixel is invalid rather than merely bright.

    Dectris detectors flag dead, masked and saturated pixels with the maximum value their
    counter holds. That is usually the maximum of the pattern dtype -- 4294967295 on a
    uint32 Eiger -- but an instrument storing a shallower counter in a wider dtype marks
    them lower, as a 12-bit Eiger written as uint16 does with 4095; *invalid_counts* names
    such a marker by dtype name. Left in place those markers dominate every statistic
    computed from the frame: the beam-center estimator rejects the real beam as noise and
    silently returns the detector midpoint instead.
    """
    if invalid_counts is not None and str(dtype) in invalid_counts:
        return invalid_counts[str(dtype)]

    if numpy.issubdtype(dtype, numpy.integer):
        # issubdtype does not narrow the pattern dtype union for the checker.
        return int(numpy.iinfo(cast(numpy.dtype[numpy.integer], dtype)).max)

    return None


_OPTIONS_HELP = (
    f'Options document, as {StandardFileLayout.RECONSTRUCT_OPTIONS} holds one and '
    '--print-options writes one: "library", "algorithm" and the library\'s "options". '
    'Fields it omits keep their defaults. "-" reads from stdin. Hand-written YAML must '
    'spell scientific notation 1.0e-3 rather than 1e-3, which YAML reads as a string.'
)

# The library a run falls back to when given no options, and its algorithm by regime: BH
# for near field, which reconstructs a hologram better than stock LSQML does.
_DEFAULT_LIBRARY: Final = 'PtyChi'
_DEFAULT_FAR_FIELD_ALGORITHM: Final = 'LSQML'
_DEFAULT_NEAR_FIELD_ALGORITHM: Final = 'BH'


@dataclass(frozen=True)
class ResolvedReconstruction:
    """The library, algorithm and options a run reconstructs with."""

    library: str
    reconstructor: Reconstructor
    options: BaseModel

    def selectors(self) -> dict[str, str]:
        """What an options document names, beside the options it gives."""
        return {'library': self.library, 'algorithm': self.reconstructor.name}

    def dump(self) -> str:
        """These options as the canonical document, which :func:`read_reconstruction` reads."""
        return dump_options_document(self.selectors(), self.options)


def add_options_argument(
    parser: argparse.ArgumentParser,
    *,
    default_note: str = 'Defaults to stock pty-chi LSQML, or BH for a near-field run.',
) -> None:
    """Add ``--options``, so its help text is written once rather than ten times."""
    parser.add_argument(
        '--options',
        metavar='OPTIONS_FILE',
        type=Path,
        default=None,
        help=f'{_OPTIONS_HELP} {default_note}',
    )


class _PrintOptionsAction(argparse.Action):
    """Print one algorithm's default options and exit, during parsing.

    An action rather than a flag the command checks, for two reasons. Every command
    carrying this marks its real input required, and argparse rejects a missing required
    argument before any command code runs -- so a flag would make the command unusable on
    its own, which is the only way anyone would call it. And a command that redirects its
    standard output into a log file would print into the log. Parsing happens before both.
    """

    def __init__(
        self,
        option_strings: Sequence[str],
        dest: str,
        *,
        library: str | None,
        **kwargs: Any,
    ) -> None:
        self._library = library
        super().__init__(option_strings, dest, nargs='?', default=argparse.SUPPRESS, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: str | Sequence[Any] | None,
        option_string: str | None = None,
    ) -> None:
        from ptychodus.api.plugins import PluginRegistry

        text = '' if values is None else str(values)
        library_name, _, algorithm_name = text.partition('/')

        if self._library is None:
            library_name = library_name or _DEFAULT_LIBRARY
        else:
            # A command bound to one library takes the algorithm alone.
            library_name, algorithm_name = self._library, text

        libraries = PluginRegistry.load_plugins().reconstructor_libraries
        named = libraries.find_plugin(library_name)

        # Unnamed, pty-chi's algorithm is the fallback a run makes; another library's is
        # its only one, which resolving with no name picks or asks for.
        if not algorithm_name and named is not None and named.simple_name == _DEFAULT_LIBRARY:
            algorithm_name = _DEFAULT_FAR_FIELD_ALGORITHM

        try:
            plugin, reconstructor = resolve_reconstructor(
                libraries, library_name, algorithm_name or None
            )
        except OptionsError as exc:
            parser.error(str(exc))

        document = format_options_document(
            f'{plugin.display_name} {reconstructor.name} options',
            {'library': plugin.simple_name, 'algorithm': reconstructor.name},
            reconstructor.get_default_options(),
        )
        # Explicitly stdout. ``parser.exit(message=...)`` writes to stderr, which would
        # leave ``--print-options > options.yaml`` -- the whole point of the flag --
        # producing an empty file.
        parser._print_message(document, sys.stdout)
        parser.exit()


def add_print_options_argument(
    parser: argparse.ArgumentParser, *, library: str | None = None
) -> None:
    """Add ``--print-options``, the way to obtain a file for ``--options``.

    Bare, it prints the options a run falls back to when given none, so the printed file
    reproduces the default run. With `library`, the command runs only that library and the
    argument names an algorithm alone.
    """
    if library is None:
        metavar = 'LIBRARY[/ALGORITHM]'
        what = (
            f"a library's algorithm (default {_DEFAULT_LIBRARY}/{_DEFAULT_FAR_FIELD_ALGORITHM}"
            f'; {_DEFAULT_LIBRARY}/{_DEFAULT_NEAR_FIELD_ALGORITHM} is the near-field fallback)'
        )
    else:
        metavar = 'ALGORITHM'
        what = (
            f'an algorithm (default {_DEFAULT_FAR_FIELD_ALGORITHM}; '
            f'{_DEFAULT_NEAR_FIELD_ALGORITHM} is the near-field fallback)'
        )

    parser.add_argument(
        '--print-options',
        action=_PrintOptionsAction,
        library=library,
        metavar=metavar,
        help=(
            f"Print {what}'s default options as a YAML document, then exit. Redirect to a "
            'file, edit it, and pass it back with --options.'
        ),
    )


def read_reconstruction(
    libraries: PluginCatalog[ReconstructorLibrary], options_file: Path
) -> ResolvedReconstruction:
    """The run an options document describes, its options merged onto the defaults.

    `options_file` named ``-`` reads stdin.

    Raises:
        OptionsError: if the document is malformed, names no installed library or
            algorithm, or gives options its library does not accept.
        OSError: if `options_file` cannot be read.
    """
    text = sys.stdin.read() if options_file == Path('-') else options_file.read_text()
    return parse_reconstruction(libraries, text, source=str(options_file))


def parse_reconstruction(
    libraries: PluginCatalog[ReconstructorLibrary], text: str, *, source: str
) -> ResolvedReconstruction:
    """As :func:`read_reconstruction`, from the document's text; `source` names it in errors."""
    selectors, fields = parse_options_document(text, ('library', 'algorithm'), source=source)
    plugin, reconstructor = resolve_reconstructor(
        libraries, selectors['library'], selectors['algorithm']
    )
    options = apply_options_document(
        reconstructor.get_default_options(),
        fields,
        subject=f'{plugin.display_name} {reconstructor.name}',
    )
    return ResolvedReconstruction(plugin.simple_name, reconstructor, options)


def default_reconstruction(
    libraries: PluginCatalog[ReconstructorLibrary], *, far_field: bool
) -> ResolvedReconstruction:
    """The run made when no options are given: stock pty-chi, its algorithm by regime.

    Raises:
        OptionsError: if pty-chi is not installed, since only an options document can then
            say what to run.
    """
    algorithm = _DEFAULT_FAR_FIELD_ALGORITHM if far_field else _DEFAULT_NEAR_FIELD_ALGORITHM

    try:
        plugin, reconstructor = resolve_reconstructor(libraries, _DEFAULT_LIBRARY, algorithm)
    except OptionsError as exc:
        raise OptionsError(f'{exc} Pass --options to name a library that is.') from exc

    if not far_field:
        logger.info('No options given; using %s for a near-field reconstruction.', algorithm)

    return ResolvedReconstruction(
        plugin.simple_name, reconstructor, reconstructor.get_default_options()
    )


def find_options_document(options_file: Path | None, input_directory: Path) -> Path | None:
    """The options document for a prepared directory, or `None` to fall back to the default.

    An explicit argument always wins, including the ``-`` that means stdin. Otherwise the
    options staged in `input_directory` are used when they are there. A directory staged by
    hand, or by a tool that chooses no algorithm, carries none, so their absence is warned
    about rather than treated as an error.
    """
    if options_file is not None:
        return options_file

    staged_file = StandardFileLayout.RECONSTRUCT_OPTIONS.path(input_directory)

    if staged_file.is_file():
        logger.info('Using the options staged in %s', staged_file)
        return staged_file

    logger.warning(
        'No %s in %s; reconstructing with stock options. Pass --options to choose another.',
        StandardFileLayout.RECONSTRUCT_OPTIONS.value,
        input_directory,
    )
    return None


def ptychi_reconstruction(
    libraries: PluginCatalog[ReconstructorLibrary], options: PtyChiOptions
) -> ResolvedReconstruction:
    """The pty-chi run `options` describe: the algorithm whose options they are.

    Raises:
        OptionsError: if pty-chi is not installed or offers no algorithm taking `options`.
    """
    plugin, _ = resolve_reconstructor(libraries, _DEFAULT_LIBRARY, _DEFAULT_FAR_FIELD_ALGORITHM)

    for reconstructor in plugin.strategy:
        if type(reconstructor.get_default_options()) is type(options):
            return ResolvedReconstruction(plugin.simple_name, reconstructor, options)

    raise OptionsError(f'{plugin.display_name} has no algorithm taking {type(options).__name__}.')


def require_ptychi_options(resolved: ResolvedReconstruction) -> PtyChiOptions:
    """The pty-chi options a pty-chi driver runs, refusing any other library's.

    Raises:
        OptionsError: if `resolved` names another library.
    """
    # Deferred, so that importing this module needs no pty-chi.
    from ptychodus.plugins.ptychi.options import PtyChiOptions

    if not isinstance(resolved.options, PtyChiOptions):
        raise OptionsError(
            f'This command reconstructs with {_DEFAULT_LIBRARY}; the options name '
            f'{resolved.library}. Use ptychodus-reconstruct for other libraries.'
        )

    return resolved.options


def resolve_ptychi_reconstruction(
    libraries: PluginCatalog[ReconstructorLibrary],
    options_file: Path | None,
    *,
    far_field: bool,
    num_sync_epochs: int | None = None,
) -> ResolvedReconstruction:
    """The pty-chi run a pty-chi driver makes: from `options_file`, else the default.

    A fresh object each call, so per-scan edits never leak into the next scan.
    `num_sync_epochs`, when given, overrides the document's.

    Raises:
        OptionsError: if the document is unusable or names another library.
    """
    resolved = (
        default_reconstruction(libraries, far_field=far_field)
        if options_file is None
        else read_reconstruction(libraries, options_file)
    )
    options = require_ptychi_options(resolved)

    if num_sync_epochs is not None:
        options.num_sync_epochs = num_sync_epochs

    return resolved


def check_ptychi_options(resolved: ResolvedReconstruction, product: Product) -> None:
    """Refuse now options pty-chi would refuse when the reconstruction starts.

    Aligns a copy against `product` as the reconstruction will and runs pty-chi's own
    check, so a prepared directory a queued job would reject is rejected while preparing.

    Raises:
        ValueError: if pty-chi rejects the aligned options.
    """
    # Deferred, so that importing this module needs no pty-chi.
    from ptychodus.plugins.ptychi.options import apply_to_task
    from ptychodus.plugins.ptychi.task import align_task_options_with_product

    options = require_ptychi_options(resolved).model_copy(deep=True)
    align_task_options_with_product(apply_to_task(options), product).check()


def load_trained_model(
    logger: logging.Logger, reconstructor: Reconstructor, input_directory: Path
) -> None:
    """Load the trained model a trainable `reconstructor` finds in `input_directory`, if any."""
    if not isinstance(reconstructor, TrainableReconstructor):
        return

    model_files = [
        input_directory / f'{StandardFileLayout.MODEL_BASENAME}{extension}'
        for extension in reconstructor.get_model_file_extensions()
    ]
    model_file = next((path for path in model_files if path.is_file()), None)

    if model_file is None:
        logger.info(
            'No model at %s; %s runs without one.',
            ', '.join(str(path) for path in model_files),
            reconstructor.name,
        )
    else:
        logger.info('Loading the model in %s', model_file)
        reconstructor.load_model(model_file)


def save_assembled_diffraction(
    logger: logging.Logger,
    output_directory: Path,
    assembled_data: AssembledDiffractionData,
    *,
    skip: bool,
) -> None:
    """Create `output_directory` and write the assembled patterns to ``diffraction.h5``.

    Called ahead of the reconstruction, so an interrupted run still leaves the directory
    usable: the assembled patterns are the one file that cannot be rebuilt without the
    raw beamline files. Pass `skip` to create the directory without writing them, for a
    run whose patterns would be larger than they are worth keeping.

    The directory is created on every rank, since every rank needs somewhere to work; only
    the write is confined to the main process.
    """
    output_directory.mkdir(parents=True, exist_ok=True)
    diffraction_file = StandardFileLayout.DIFFRACTION.path(output_directory)

    if skip:
        logger.info('Skipping %s as requested', diffraction_file.name)
    elif is_main_process():
        logger.info('Writing %s', diffraction_file)
        save_diffraction_data(diffraction_file, assembled_data)


def estimate_incident_beam_from_assembled_data(
    assembled_data: AssembledDiffractionData,
    positions: ProbePositionSequence,
    *,
    assembly_moments: AssemblyMomentAccumulator | None = None,
    i0_photons_per_count: float = 0.0,
) -> IncidentBeamEstimate:
    """The incident photon count and mean transmission the assembled patterns imply.

    :func:`estimate_incident_beam` judges the moments of the patterns with the I0 the
    positions carry, calibrated by *i0_photons_per_count* when positive. The moments
    come from *assembly_moments* when it was passed as the assembly's
    ``on_array_assembled`` callback, so they were taken while the arrays were read; a
    filter that dropped patterns since is fine. Without it, or where it cannot serve,
    one pass over the assembled patterns takes them.

    Raises:
        ValueError: if no patterns were assembled.
    """
    if assembly_moments is None:
        assembly_moments = AssemblyMomentAccumulator()

    moments = assembly_moments.get_moments(assembled_data)

    if moments is None:
        raise ValueError('No diffraction patterns were assembled to estimate the beam from.')

    return estimate_incident_beam(
        moments,
        positions.get_i0_by_index(),
        i0_photons_per_count=i0_photons_per_count,
    )


def pair_patterns_with_positions(
    logger: logging.Logger,
    assembled_data: AssembledDiffractionData,
    product: Product,
    *,
    index_filter: PositionIndexFilter = PositionIndexFilter.ALL,
) -> ReconstructInput:
    """Pair the assembled patterns with the product's positions, and say if any went unpaired.

    A shortfall is ordinary on a scan that was stopped mid-raster: the detector keeps the
    line it was on, the position record truncates at its last completed line, and the
    patterns past that end have no position to pair with. It is reported as a warning
    rather than raised, because the patterns that did pair are still a valid scan. Patterns
    an `index_filter` excludes on purpose are not counted as unpaired.
    """
    reconstruct_input = prepare_reconstruct_input(assembled_data, product, index_filter)
    num_paired = reconstruct_input.diffraction_patterns.shape[0]
    num_patterns = assembled_data.get_num_patterns()

    if index_filter is PositionIndexFilter.ALL and num_paired < num_patterns:
        logger.warning(
            'Paired %d of %d assembled patterns; %d had no probe position, which is'
            ' expected when the detector recorded more scan lines than the position file.',
            num_paired,
            num_patterns,
            num_patterns - num_paired,
        )

    return reconstruct_input


def save_reconstruct_options(
    logger: logging.Logger, resolved: ResolvedReconstruction, output_directory: Path
) -> None:
    """Write `resolved` to ``reconstruct_options.yaml``, from the main process only."""
    if is_main_process():
        options_file = StandardFileLayout.RECONSTRUCT_OPTIONS.path(output_directory)
        logger.info('Writing %s', options_file)
        options_file.write_text(resolved.dump())


def run_reconstruction(
    logger: logging.Logger,
    resolved: ResolvedReconstruction,
    reconstruct_input: ReconstructInput,
    output_directory: Path,
    *,
    cancellation: CancellationToken | None = None,
    on_sync: Callable[[ReconstructOutput, Path | None], None] | None = None,
) -> Product | None:
    """Run `resolved` to completion in this process, and write the outputs.

    Assumes `output_directory` already exists, and does not touch `diffraction.h5`: that
    file depends on the assembled dataset rather than on the reconstruction, so it is
    written by :func:`save_assembled_diffraction` before this runs. Writes the options to
    ``reconstruct_options.yaml``, a ``product.NNNNNN.h5`` checkpoint at every output the
    reconstructor yields, then the final ``product.h5``. Returns the final product.

    Output files are written once, by the main process. Only the filesystem writes are
    gated: every rank runs the reconstruction to completion and returns the same product.

    With a `cancellation` token, the loop stops at the first output after the request and
    still writes ``product.h5`` from what did run -- a partially converged product is the
    whole reason to stop politely rather than kill the process. The return is then `None`
    only when the request arrived before any output, since there is no product in that
    case; an *uncancelled* run that produces nothing is still an error.

    `on_sync` is called at each output with it and the checkpoint path written for it,
    which is `None` on a rank that writes nothing. It is how a caller reports progress
    without reimplementing this loop.
    """
    main_process = is_main_process()
    product_file = StandardFileLayout.PRODUCT.path(output_directory)
    save_reconstruct_options(logger, resolved, output_directory)

    logger.info(
        'Starting reconstruction: %s %s on %d patterns',
        resolved.library,
        resolved.reconstructor.name,
        reconstruct_input.diffraction_patterns.shape[0],
    )

    if _agree_on_cancellation(cancellation):
        # Asked to stop while the inputs were being prepared, before the first output.
        logger.warning('Cancelled before the first iteration; nothing was reconstructed.')
        return None

    final_output = None

    for output in resolved.reconstructor.reconstruct(reconstruct_input, options=resolved.options):
        losses = output.product.losses
        last_loss = losses[-1].value if losses else float('nan')
        logger.info('Iteration %d/%d: loss=%.6g', output.progress, output.progress_goal, last_loss)

        checkpoint_file: Path | None = None

        if main_process:
            checkpoint_file = StandardFileLayout.PRODUCT.checkpoint_path(
                output_directory, output.progress
            )
            save_product(checkpoint_file, output.product)

        if on_sync is not None:
            on_sync(output, checkpoint_file)

        final_output = output

        if _agree_on_cancellation(cancellation):
            logger.warning(
                'Cancelled after iteration %d of %d.', output.progress, output.progress_goal
            )
            break

    if final_output is None:
        raise RuntimeError('Reconstruction produced no output.')

    if main_process:
        save_product(product_file, final_output.product)
        logger.info('Saved reconstructed product to %s', product_file)

    return final_output.product
