#!/usr/bin/env python
"""Run one APS ptychography reconstruction from a flat JSON parameter file.

The parameter file is the one PEAR's batch loop writes before spawning a worker, so
this script substitutes for a ``ptycho_recon`` invocation on the same inputs. Every
stage below runs through ``ptychodus.api``; only the pty-chi task boundary comes from
``ptychodus.model.ptychi.task``, as in ``reconstruct_lamni.py``.

Scope and limitations
---------------------

- The settings-encoding output directory convention is preserved, so a run lands where
  it would have. There is deliberately no output argument: the parameter file already
  names the destination through ``recon_dir_base``, or through the per-instrument
  convention derived from ``data_directory``, and an argument could only override it.
- That directory is filled in the ptychodus standard layout -- ``diffraction.h5``,
  ``ptychi_options.json``, per-epoch ``product.NNNNNN.h5`` checkpoints, and the final
  ``product.h5``. The per-iteration TIFF stacks and plots of the original workflow are
  still not written.
- One reconstruction only: no batch queue, no multi-scan shared-object mode, no
  affine-calibration loop.
- No GPU selection, ``CUDA_VISIBLE_DEVICES``, thread-count, or default-device side
  effects. Choose a device with ``CUDA_VISIBLE_DEVICES`` in the environment.
- Positions carry their file-provided integer index, so patterns and positions pair
  through :func:`prepare_reconstruct_input` rather than by array order. For
  softGlueZynq LamNI files this corrects an off-by-one, since ``Detector_Count``
  starts at 1.
- ``object_regularization_llm`` is accepted and ignored: no released pty-chi
  implements it.
- X-rays only. A file declaring ``beam_source`` as anything else is rejected rather
  than reconstructed at the photon wavelength.
- Detector pixel pitch comes from the diffraction reader, which supplies its
  instrument's default when the file itself carries none.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal

import numpy
from pydantic import BaseModel, ConfigDict, Field

from ptychi.api import (
    BatchingModes,
    Dtypes,
    ImageGradientMethods,
    ImageIntegrationMethods,
    LSQMLOptions,
    NoiseModels,
    OPRWeightSmoothingMethods,
    Optimizers,
    OrthogonalizationMethods,
    PatchInterpolationMethods,
    PositionCorrectionTypes,
)

from ptychodus.api.assemble import assemble_dataset
from ptychodus.api.constants import energy_eV_to_wavelength_m
from ptychodus.api.diffraction import CropRegion
from ptychodus.api.geometry import ImageExtent, PixelGeometry
from ptychodus.api.object import Object, compute_object_geometry
from ptychodus.api.io import StandardFileLayout, save_diffraction_data, save_product
from ptychodus.api.plugins import PluginRegistry
from ptychodus.api.preprocess.diffraction import (
    DiffractionPrepPipeline,
    DiffractionPrepStepUnion,
    HorizontalFlipStep,
    TransposeStep,
    VerticalFlipStep,
)
from ptychodus.api.preprocess.probe_positions import AffineTransform, transform_probe_positions
from ptychodus.api.probe import Probe, ProbeGeometry, ProbeSequence
from ptychodus.api.probe_positions import ProbePosition, ProbePositionSequence
from ptychodus.api.product import Product, ProductMetadata
from ptychodus.api.propagate import (
    AngularSpectrumPropagator,
    PropagatorParameters,
    compute_magnification,
)
from ptychodus.api.reconstruct import prepare_reconstruct_input
from ptychodus.api.simulate.object import generate_random_object
from ptychodus.api.simulate.probe import (
    generate_coherent_probe_modes,
    generate_fresnel_zone_plate_probe,
)
from ptychodus.model.ptychi.task import (
    align_task_options_with_product,
    dump_task_options,
    reconstruct_with_ptychi,
)

logger = logging.getLogger('reconstruct_pear')


@dataclass(frozen=True)
class _Readers:
    """Reader plugin names for one instrument."""

    diffraction: str
    positions: str


# Keyed by the parameter file's `instrument` value, lowercased.
_INSTRUMENT_READERS: Final[dict[str, _Readers]] = {
    '2ide': _Readers('APS_2IDE', 'APS_2IDE'),
    'atomic': _Readers('APS_Atomic', 'APS_Atomic'),
    '2xfm': _Readers('APS_2IDE', 'APS_2IDE'),
    'bnp': _Readers('APS_BNP', 'APS_BNP'),
    'bionanoprobe': _Readers('APS_BNP', 'APS_BNP'),
    '12idc': _Readers('APS_PtychoSAXS', 'APS_PtychoSAXS'),
    'isn': _Readers('APS_ISN', 'APS_ISN_MDA'),
    'lynx': _Readers('APS_LamNI', 'APS_LamNI'),
    'lynx_v2': _Readers('APS_LamNI', 'APS_LamNI'),
    'velo': _Readers('APS_Velociprobe', 'APS_Velociprobe_PE'),
    'velociprobe': _Readers('APS_Velociprobe', 'APS_Velociprobe_PE'),
    'simu': _Readers('fold_slice', 'fold_slice'),
}

_BATCHING_MODES: Final[dict[str, BatchingModes]] = {
    'random': BatchingModes.RANDOM,
    'uniform': BatchingModes.UNIFORM,
    'compact': BatchingModes.COMPACT,
}

_DIFFERENTIATION_METHODS: Final[dict[str, ImageGradientMethods]] = {
    'gaussian': ImageGradientMethods.GAUSSIAN,
    'fourier': ImageGradientMethods.FOURIER_DIFFERENTIATION,
}


class PearParameters(BaseModel):
    """Validated view of the JSON parameter file, using its own key names.

    Defaults mirror the source workflow's fallbacks, so a file that omits a key
    configures the same run there and here. Unknown keys are ignored rather than
    rejected -- a real file carries batch-loop bookkeeping this script has no use for
    -- but they are logged, so a mistyped key is still discoverable.
    """

    model_config = ConfigDict(extra='ignore')

    # --- instrument and inputs ---
    instrument: str = ''
    scan_num: int = 0
    data_directory: Path = Path()
    path_to_diffraction_file: Path | None = None
    path_to_position_file: Path | None = None
    diffraction_file_type: str | None = None
    position_file_type: str | None = None

    # --- beam and geometry ---
    # Only x-rays: the electron wavelength and its reciprocal-space pixel convention
    # are not implemented, and running an electron scan through the photon wavelength
    # would reconstruct at a silently wrong pixel size.
    beam_source: Literal['xray'] = 'xray'
    beam_energy_kev: float = 0.0
    det_sample_dist_m: float = 0.0
    focal_sample_dist_m: float = 0.0
    near_field_ptycho: bool = False

    # --- diffraction preprocessing ---
    diff_pattern_size_pix: int = 0
    diff_pattern_center_x: int | None = None
    diff_pattern_center_y: int | None = None
    minimal_num_of_diff_pattern: int = 1
    flip_diffraction_patterns_up_down: bool = False
    flip_diffraction_patterns_left_right: bool = False
    transpose_diffraction_patterns: bool = False
    avg_photon_threshold_lb: float = 0.0
    avg_photon_threshold_ub: float = 0.0
    burst_ptycho: bool = False

    # --- positions ---
    init_position_affine_matrix: list[float] = Field(default_factory=list)

    # --- probe ---
    use_model_FZP_probe: bool = False  # noqa: N815
    fzp_preset: str = 'APS 33-ID-C VelociProbe'
    fzp_defocus_m: float = 0.0
    path_to_init_probe: Path | None = None
    init_probe_file_type: str = 'NPY'
    number_probe_modes: int = 1
    number_opr_modes: int = 0
    init_probe_shifts_pix: list[float] = Field(default_factory=list)
    init_probe_propagation_distance_mm: float = 0.0

    # --- object ---
    path_to_init_object: Path | None = None
    init_object_file_type: str = 'NPY'
    obj_pad_size_m: float = 1e-6
    number_of_slices: int = 1
    object_thickness_m: float = 0.0
    layer_regularization: float = 0.0
    object_smoothness_alpha: float = 0.0
    update_object_w_higher_probe_modes: bool = False
    init_layer_select: list[int] = Field(default_factory=list)
    init_layer_preprocess: str = ''
    init_layer_append_mode: str = 'vac'
    init_layer_scaling_factor: float = 1.0

    # --- optimization ---
    number_of_iterations: int = 0
    update_batch_size: int | None = None
    number_of_batches: int | None = None
    batch_selection_scheme: Literal['random', 'uniform', 'compact'] = 'random'
    momentum_acceleration: bool = False
    noise_model: Literal['gaussian', 'poisson'] = 'gaussian'
    diffraction_pattern_blur: float | None = None
    probe_update_start_iteration: int = 1
    center_probe: bool = False
    probe_support: bool = False
    intensity_correction: bool = False
    position_correction: bool = False
    position_correction_start_iteration: int = 1
    position_correction_gradient_method: Literal['gaussian', 'fourier'] = 'gaussian'
    position_correction_update_limit: float = 0.0
    position_correction_affine_constraint: bool = False
    position_correction_layer: int | None = None

    # --- output ---
    recon_parent_dir: str = ''
    recon_dir_base: Path | None = None
    recon_dir_suffix: str = ''
    save_freq_iterations: int = 10
    seed: int = 0

    @classmethod
    def from_file(cls, file: Path) -> PearParameters:
        with file.open() as stream:
            raw = json.load(stream)

        if not isinstance(raw, dict):
            raise ValueError(f'{file} does not contain a JSON object.')

        parameters = cls.model_validate(raw)
        ignored = sorted(set(raw) - set(cls.model_fields))

        if ignored:
            logger.debug('Ignoring unmodelled keys: %s', ', '.join(ignored))

        return parameters


def _slice_spacings_m(object_thickness_m: float, number_of_slices: int) -> list[float]:
    """Uniform slice spacing: the thickness split evenly across the slices."""
    if number_of_slices <= 1 or object_thickness_m <= 0.0:
        return []

    return [object_thickness_m / number_of_slices] * (number_of_slices - 1)


def _batch_size(params: PearParameters, num_patterns: int) -> int:
    """Resolve the minibatch size.

    An explicit size wins; otherwise a requested batch count divides the patterns;
    otherwise the count defaults by batching mode, one for compact and ten otherwise.
    """
    update_batch_size = params.update_batch_size

    if update_batch_size is not None:
        return update_batch_size

    number_of_batches = params.number_of_batches

    if number_of_batches is None:
        number_of_batches = 1 if params.batch_selection_scheme == 'compact' else 10

    return max(1, num_patterns // number_of_batches)


def _output_directory(params: PearParameters, options: LSQMLOptions, pattern_width_px: int) -> Path:
    """Build the settings-encoding output directory name.

    The name records the configuration that produced the run, so sibling directories
    under one scan are self-describing. Preserved so a run lands where it otherwise
    would; the contents differ (see the module docstring).
    """
    explicit = params.recon_dir_base

    if explicit is not None:
        base = explicit
    else:
        data_directory = Path(params.data_directory)
        instrument = params.instrument.lower()
        recons = data_directory / 'ptychi_recons'

        if instrument in ('lynx', 'lynx_v2'):
            recons = data_directory / 'analysis' / 'ptychi_recons'

        base = recons / params.recon_parent_dir / f'S{params.scan_num:04d}'

    reconstructor_options = options.reconstructor_options
    batching_suffix = {
        BatchingModes.RANDOM: 'r',
        BatchingModes.UNIFORM: 's',
        BatchingModes.COMPACT: 'c',
    }.get(reconstructor_options.batching_mode, '')

    # Ndp is the pattern edge in pixels, matching pear_save_recon.py, which builds it
    # from data.shape[1]. Naming it after the pattern count instead puts every run in
    # a directory the original workflow would never have written.
    name = f'Ndp{pattern_width_px}_LSQML_{batching_suffix}{reconstructor_options.batch_size}'

    momentum_gain = reconstructor_options.momentum_acceleration_gain or 0.0

    if batching_suffix == 'c' and momentum_gain > 0:
        name += f'_m{momentum_gain}'

    name += '_poisson' if params.noise_model == 'poisson' else '_gaussian'

    if params.near_field_ptycho:
        name += f'_nf_fsd{params.focal_sample_dist_m / 1e-3:.2f}mm'

    name += f'_p{options.probe_options.initial_guess.shape[1]}'

    if options.probe_options.center_constraint.enabled:
        name += '_cp'
    if options.object_options.multimodal_update:
        name += '_mm'
    if options.opr_mode_weight_options.optimizable:
        name += f'_opr{options.probe_options.initial_guess.shape[0] - 1}'
    if options.opr_mode_weight_options.optimize_intensity_variation:
        name += '_ic'

    number_of_slices = params.number_of_slices
    object_thickness_m = params.object_thickness_m

    if object_thickness_m > 0 and number_of_slices > 1:
        if params.beam_source == 'electron':
            name += f'_Ns{number_of_slices}_T{object_thickness_m / 1e-9:.2f}nm'
        else:
            name += f'_Ns{number_of_slices}_T{object_thickness_m / 1e-6:.2f}um'

        multislice = options.object_options.multislice_regularization

        if multislice.enabled and multislice.weight > 0:
            name += f'_reg{multislice.weight}'

    position_options = options.probe_position_options

    if position_options.optimizable:
        name += f'_pc{position_options.optimization_plan.start}'
        name += '_f' if params.position_correction_gradient_method == 'fourier' else '_g'

        update_limit = position_options.correction_options.update_magnitude_limit or 0.0

        if update_limit > 0:
            name += f'_ul{update_limit}'
        if position_options.correction_options.slice_for_correction:
            name += f'_layer{position_options.correction_options.slice_for_correction}'
        if position_options.affine_transform_constraint.apply_constraint:
            name += '_affine'

    if params.init_probe_propagation_distance_mm != 0:
        name += f'_pd{params.init_probe_propagation_distance_mm}'
    if options.object_options.smoothness_constraint.enabled:
        name += f'_smooth{options.object_options.smoothness_constraint.alpha}'

    blur_sigma = options.reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma

    if blur_sigma:
        name += f'_dpBlur{blur_sigma}'

    flips = (
        ('_ud', params.flip_diffraction_patterns_up_down),
        ('_lr', params.flip_diffraction_patterns_left_right),
        ('_tr', params.transpose_diffraction_patterns),
    )

    if any(enabled for _, enabled in flips):
        name += '_dpFlip' + ''.join(suffix for suffix, enabled in flips if enabled)

    suffix = params.recon_dir_suffix

    if suffix:
        name += f'_{suffix}'

    return base / name


def _resolve_inputs(params: PearParameters, readers: _Readers) -> tuple[Path, Path]:
    """Locate the diffraction and position files for this scan.

    Explicit ``path_to_diffraction_file`` / ``path_to_position_file`` keys win. Failing
    that the per-instrument directory convention is applied to ``data_directory`` and
    ``scan_num``. The diffraction path names one member of a series or a master file;
    the readers glob the rest themselves.
    """
    explicit_diffraction = params.path_to_diffraction_file
    explicit_positions = params.path_to_position_file

    if explicit_diffraction is not None and explicit_positions is not None:
        return explicit_diffraction, explicit_positions

    base = Path(params.data_directory)
    scan = params.scan_num
    instrument = params.instrument.lower()

    if instrument in ('2ide', '2xfm'):
        # Three digits, not six: the beamline writes fly054_data_001.h5. The reader
        # globs the remaining frames from whichever member it is handed.
        diffraction = base / 'ptycho' / f'fly{scan:03d}_data_001.h5'
        positions = base / 'mda' / f'2xfm_{scan:04d}.mda'
    elif instrument == 'isn':
        diffraction = base / 'PTYCHO' / f'19ide_{scan:04d}_000.h5'
        positions = base / 'mda' / f'19ide_{scan:04d}.mda'
    elif instrument in ('bnp', 'bionanoprobe'):
        diffraction = base / 'ptycho' / f'bnp_fly{scan:04d}_000000.h5'
        positions = base / 'mda' / f'bnp_fly{scan:04d}.mda'
    elif instrument == '12idc':
        diffraction = base / 'ptycho' / f'{scan:03d}' / f'{scan:03d}_00001_0.h5'
        positions = base / 'positions' / f'{scan:03d}' / f'{scan:03d}_00001_0.dat'
    elif instrument in ('lynx', 'lynx_v2'):
        block_start = (scan // 1000) * 1000
        block = f'S{block_start:05d}-{block_start + 999:05d}'
        stem = f'scan_{scan:05d}_000' if instrument == 'lynx_v2' else f'run_{scan:05d}_000000000000'
        diffraction = base / 'data' / 'eiger_4' / block / f'S{scan:05d}' / f'{stem}.h5'
        positions = base / 'data' / 'scan_positions' / f'scan_{scan:05d}.dat'
    elif instrument in ('velo', 'velociprobe'):
        diffraction = base / 'ptycho' / f'fly{scan:03d}' / f'fly{scan:03d}_master.h5'
        positions = base / 'positions' / f'fly{scan:03d}_0.txt'
    elif instrument == 'simu':
        diffraction, positions = _resolve_fold_slice_pair(base, scan)
    else:
        raise ValueError(
            f'No directory convention for instrument {instrument!r}; give explicit '
            '"path_to_diffraction_file" and "path_to_position_file" keys.'
        )

    diffraction = explicit_diffraction or diffraction
    positions = explicit_positions or positions

    for role, path in (('diffraction', diffraction), ('position', positions)):
        if not path.exists():
            raise FileNotFoundError(
                f'Resolved {role} path does not exist: {path}. Override it with '
                f'"path_to_{role}_file" in the parameter file.'
            )

    return diffraction, positions


def _resolve_fold_slice_pair(base: Path, scan: int) -> tuple[Path, Path]:
    """Locate the `data_roi*_dp.hdf5` / `_para.hdf5` pair for one scan.

    Unlike the raw layouts, the file name embeds preprocessing choices -- the ROI, the
    pattern size, the downsampling -- that the scan number does not determine, so the
    directory is derived and the file within it is globbed. A scan preprocessed several
    ways yields several candidates and is reported rather than guessed at.
    """
    candidates = [
        base / 'results' / f'{scan:03d}',
        base / 'results' / f'{scan}',
        base / 'results' / 'ML_recon' / f'scan{scan:03d}',
        base / 'results' / 'ML_recon' / f'scan{scan:06d}',
        base / 'results' / 'ML_recon' / f'fly{scan:03d}',
        base,
    ]
    found = [
        match
        for directory in candidates
        if directory.is_dir()
        for match in sorted(directory.glob('data_roi*_dp.hdf5'))
    ]

    if not found:
        tried = ', '.join(str(directory) for directory in candidates)
        raise FileNotFoundError(
            f'No "data_roi*_dp.hdf5" found for scan {scan}. Looked in: {tried}. '
            'Give explicit "path_to_diffraction_file" and "path_to_position_file" keys.'
        )

    if len(found) > 1:
        names = ', '.join(str(match) for match in found)
        raise ValueError(
            f'Scan {scan} has several preprocessed pairs: {names}. Name the one you want '
            'with "path_to_diffraction_file" and "path_to_position_file".'
        )

    diffraction = found[0]
    positions = diffraction.with_name(diffraction.name.replace('_dp.hdf5', '_para.hdf5'))

    if not positions.is_file():
        raise FileNotFoundError(f'Found {diffraction} but no matching "{positions.name}".')

    return diffraction, positions


def _prep_pipeline(params: PearParameters) -> DiffractionPrepPipeline | None:
    """Orientation transforms, applied in the order the parameter names imply."""
    steps: list[DiffractionPrepStepUnion] = []

    if params.flip_diffraction_patterns_up_down:
        steps.append(VerticalFlipStep())
    if params.flip_diffraction_patterns_left_right:
        steps.append(HorizontalFlipStep())
    if params.transpose_diffraction_patterns:
        steps.append(TransposeStep())

    return DiffractionPrepPipeline(steps=tuple(steps)) if steps else None


def _total_counts_bounds(params: PearParameters, num_pixels: int) -> tuple[int | None, int | None]:
    """Convert per-pixel average photon thresholds into whole-pattern count bounds.

    The parameter file states an average over the cropped frame; the assembler filters
    on the total, so the two differ by the pixel count. Zero disables a bound.
    """
    lower = params.avg_photon_threshold_lb
    upper = params.avg_photon_threshold_ub

    return (
        int(lower * num_pixels) if lower else None,
        int(upper * num_pixels) if upper else None,
    )


def _cluster_burst_positions(
    positions: ProbePositionSequence, cluster_radius_m: float
) -> ProbePositionSequence:
    """Collapse positions lying within one object pixel onto their centroid.

    Burst acquisitions revisit the same point repeatedly. Giving every member of a
    cluster the same index leaves the averaging to `prepare_reconstruct_input`, which
    already folds duplicate indexes into one anchor, rather than averaging patterns
    here.
    """
    # Optional dependency, and only this branch needs it; sklearn ships no stubs.
    from sklearn.cluster import DBSCAN  # type: ignore[import-untyped]

    coordinates = numpy.array([[position.y_m, position.x_m] for position in positions])
    labels = DBSCAN(eps=cluster_radius_m, min_samples=1).fit(coordinates).labels_
    num_clusters = len({label for label in labels if label >= 0})
    logger.info('Burst clustering: %d positions into %d clusters', len(positions), num_clusters)

    return ProbePositionSequence(
        [
            ProbePosition(index=int(label), x_m=position.x_m, y_m=position.y_m)
            for label, position in zip(labels, positions)
        ]
    )


def _set_num_incoherent_modes(array: numpy.ndarray, num_modes: int) -> numpy.ndarray:
    """Pad with copies of the last mode, or truncate, to reach *num_modes*."""
    if array.shape[0] >= num_modes:
        return array[:num_modes]

    padding = numpy.repeat(array[-1:], num_modes - array.shape[0], axis=0)
    return numpy.concatenate((array, padding), axis=0)


def _shift_probe(array: numpy.ndarray, shift_px: tuple[float, float]) -> numpy.ndarray:
    """Shift every incoherent mode by (y, x) pixels."""
    import scipy.ndimage

    shifted = numpy.empty_like(array)

    for index, mode in enumerate(array):
        shifted[index] = scipy.ndimage.shift(
            mode, shift=shift_px, mode='nearest', order=1, prefilter=True
        )

    return shifted


def _propagate_probe(
    array: numpy.ndarray, *, distance_m: float, wavelength_m: float, pixel_geometry: PixelGeometry
) -> numpy.ndarray:
    """Propagate every incoherent mode by *distance_m* through the angular spectrum."""
    parameters = PropagatorParameters(
        wavelength_m=wavelength_m,
        width_px=array.shape[-1],
        height_px=array.shape[-2],
        pixel_width_m=pixel_geometry.width_m,
        pixel_height_m=pixel_geometry.height_m,
        propagation_distance_m=distance_m,
    )
    propagator = AngularSpectrumPropagator(parameters)

    return numpy.stack([propagator.propagate(mode) for mode in array])


def _select_layers(array: numpy.ndarray, params: PearParameters) -> numpy.ndarray:
    """Apply the layer selection and collapse modes to a multislice initial object."""
    selection = [int(index) for index in params.init_layer_select]
    selection = [index for index in selection if 0 <= index < array.shape[0]]

    if selection:
        logger.info('Selecting object layers %s', selection)
        array = array[selection]

    preprocess = params.init_layer_preprocess

    if preprocess in ('avg', 'avg1'):
        # The layers compose multiplicatively, so their product is the whole object;
        # dividing the phase evenly redistributes it across the requested slices.
        combined = numpy.prod(array, axis=0)
        num_layers = array.shape[0]
        phase = numpy.angle(combined) / num_layers
        averaged = numpy.abs(combined) ** (1.0 / num_layers) * numpy.exp(1j * phase)
        array = (
            averaged[numpy.newaxis]
            if preprocess == 'avg1'
            else numpy.repeat(averaged[numpy.newaxis], num_layers, axis=0)
        )

    scaling = params.init_layer_scaling_factor

    if scaling != 1.0:
        logger.info('Scaling initial object phase by %g', scaling)
        array = numpy.abs(array) * numpy.exp(1j * numpy.angle(array) * scaling)

    return array


def _fit_layer_count(array: numpy.ndarray, num_slices: int, append_mode: str) -> numpy.ndarray:
    """Grow or shrink the layer stack to *num_slices*, padding alternately front and back."""
    if array.shape[0] == num_slices:
        return array

    if array.shape[0] > num_slices:
        if num_slices == 1:
            return numpy.prod(array, axis=0)[numpy.newaxis]

        start = (array.shape[0] - num_slices) // 2
        return array[start : start + num_slices]

    if append_mode == 'avg':
        filler = numpy.mean(array, axis=0)
    elif append_mode == 'edge':
        filler = array[-1]
    else:  # vacuum
        filler = numpy.ones_like(array[0])

    for index in range(num_slices - array.shape[0]):
        block = filler[numpy.newaxis]
        array = (
            numpy.concatenate((array, block), axis=0)
            if index % 2 == 0
            else numpy.concatenate((block, array), axis=0)
        )

    return array


def _build_lsqml_options(
    params: PearParameters, probe: ProbeSequence, num_patterns: int
) -> LSQMLOptions:
    """Translate the parameter file into pty-chi LSQML options.

    Product-derived fields are left alone for `align_task_options_with_product`: the
    wavelength, the object pixel size, the slice spacings, the probe power, and the
    near-field propagation distance, which is emitted as the NaN placeholder the
    alignment recognizes.
    """
    options = LSQMLOptions()

    is_multislice = params.object_thickness_m > 0 and params.number_of_slices > 1

    # --- data ---
    data_options = options.data_options
    data_options.save_data_on_device = True
    data_options.free_space_propagation_distance_m = (
        math.nan if params.near_field_ptycho else math.inf
    )

    if params.near_field_ptycho:
        data_options.fft_shift = False

    # --- object ---
    object_options = options.object_options
    object_options.optimizable = True
    object_options.optimizer = Optimizers.SGD
    object_options.step_size = 1
    object_options.build_preconditioner_with_all_modes = False
    object_options.patch_interpolation_method = PatchInterpolationMethods.FOURIER
    object_options.remove_object_probe_ambiguity.enabled = True
    object_options.multimodal_update = params.update_object_w_higher_probe_modes

    smoothness_alpha = params.object_smoothness_alpha

    if smoothness_alpha > 0:
        object_options.smoothness_constraint.enabled = True
        object_options.smoothness_constraint.alpha = smoothness_alpha

    if is_multislice:
        layer_regularization = params.layer_regularization
        object_options.optimal_step_size_scaler = 0.9
        object_options.multislice_regularization.enabled = layer_regularization > 0
        object_options.multislice_regularization.weight = layer_regularization
        object_options.multislice_regularization.unwrap_phase = True
        object_options.multislice_regularization.unwrap_image_grad_method = (
            ImageGradientMethods.FOURIER_DIFFERENTIATION
        )
        object_options.multislice_regularization.unwrap_image_integration_method = (
            ImageIntegrationMethods.FOURIER
        )

    # --- probe ---
    probe_options = options.probe_options
    probe_options.initial_guess = probe.get_array()
    probe_options.optimizable = True
    probe_options.optimizer = Optimizers.SGD
    probe_options.step_size = 1
    probe_options.optimization_plan.start = params.probe_update_start_iteration
    probe_options.orthogonalize_incoherent_modes.enabled = True
    probe_options.orthogonalize_incoherent_modes.method = OrthogonalizationMethods.SVD
    probe_options.orthogonalize_opr_modes.enabled = True
    probe_options.center_constraint.enabled = params.center_probe
    probe_options.support_constraint.enabled = params.probe_support

    # --- probe positions ---
    position_options = options.probe_position_options
    position_options.optimizable = params.position_correction
    position_options.optimizer = Optimizers.SGD
    position_options.step_size = 1
    position_options.optimization_plan.start = params.position_correction_start_iteration
    position_options.correction_options.correction_type = PositionCorrectionTypes.GRADIENT

    position_options.correction_options.differentiation_method = _DIFFERENTIATION_METHODS[
        params.position_correction_gradient_method
    ]

    position_options.correction_options.update_magnitude_limit = (
        params.position_correction_update_limit
    )
    # Always on so the affine matrix is measured even when it is not applied.
    position_options.affine_transform_constraint.enabled = True
    position_options.affine_transform_constraint.apply_constraint = (
        params.position_correction_affine_constraint
    )
    position_options.affine_transform_constraint.position_weight_update_interval = 100

    correction_layer = params.position_correction_layer

    if is_multislice and correction_layer and position_options.optimizable:
        position_options.correction_options.slice_for_correction = correction_layer  # type: ignore[assignment]

    # --- OPR mode weights ---
    opr_options = options.opr_mode_weight_options

    if params.number_opr_modes > 0:
        opr_options.optimizable = True
        opr_options.update_relaxation = 0.1
        opr_options.smoothing.enabled = False
        opr_options.smoothing.method = OPRWeightSmoothingMethods.MEDIAN
        opr_options.smoothing.polynomial_degree = 4

    opr_options.optimize_intensity_variation = params.intensity_correction

    # --- reconstructor ---
    reconstructor_options = options.reconstructor_options
    reconstructor_options.num_epochs = params.number_of_iterations
    reconstructor_options.batch_size = _batch_size(params, num_patterns)

    reconstructor_options.batching_mode = _BATCHING_MODES[params.batch_selection_scheme]

    if params.batch_selection_scheme == 'compact':
        reconstructor_options.compact_mode_update_clustering = False

    if params.momentum_acceleration:
        reconstructor_options.momentum_acceleration_gain = 0.5
        reconstructor_options.momentum_acceleration_gradient_mixing_factor = 1

    reconstructor_options.solve_step_sizes_only_using_first_probe_mode = True
    reconstructor_options.noise_model = (
        NoiseModels.POISSON if params.noise_model == 'poisson' else NoiseModels.GAUSSIAN
    )
    reconstructor_options.use_double_precision_for_fft = False
    reconstructor_options.default_dtype = Dtypes.FLOAT32
    reconstructor_options.allow_nondeterministic_algorithms = True
    reconstructor_options.forward_model_options.diffraction_pattern_blur_sigma = (
        params.diffraction_pattern_blur
    )

    if params.near_field_ptycho:
        reconstructor_options.forward_model_options.pad_for_shift = 50

    return options


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--params',
        required=True,
        type=Path,
        help="Flat JSON parameter file, as written by PEAR's batch loop.",
    )
    parser.add_argument(
        '--no-save-diffraction',
        action='store_true',
        help='Skip diffraction.h5. The assembled patterns can run to several GB.',
    )
    parser.add_argument(
        '--num-sync-epochs',
        type=int,
        default=None,
        help='Epochs between progress logs; save_freq_iterations otherwise.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build everything and report the resolved configuration without reconstructing.',
    )
    parser.add_argument('--log-level', default=logging.INFO, type=int, help='Python logging level.')
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level,
        stream=sys.stderr,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    params = PearParameters.from_file(args.params)
    instrument = params.instrument.lower()

    try:
        readers = _INSTRUMENT_READERS[instrument]
    except KeyError:
        raise ValueError(
            f'Unknown instrument {instrument!r}; expected one of {sorted(_INSTRUMENT_READERS)}.'
        ) from None

    registry = PluginRegistry.load_plugins()
    scan_num = params.scan_num

    diffraction_file, position_file = _resolve_inputs(params, readers)

    # The per-instrument defaults assume the usual acquisition layout; a site that
    # stores positions in another format can name the reader instead.
    diffraction_file_type = params.diffraction_file_type or readers.diffraction
    position_file_type = params.position_file_type or readers.positions

    logger.info('Reading diffraction from %s as %s', diffraction_file, diffraction_file_type)
    diffraction_reader = registry.diffraction_file_readers.get_strategy_by_name(
        diffraction_file_type
    )
    raw_dataset = diffraction_reader.read(diffraction_file)
    raw_metadata = raw_dataset.get_metadata()

    detector_pixel_geometry = raw_metadata.detector_pixel_geometry

    if detector_pixel_geometry is None:
        raise ValueError(
            f'The {diffraction_file_type} reader reported no detector pixel geometry; '
            'give that reader a default pitch.'
        )

    # Crop to the requested square around the beam center, clamped so an oversized
    # request or an edge-adjacent center cannot produce a region off the detector.
    crop_px = params.diff_pattern_size_pix
    read_region: CropRegion | None = None

    if crop_px > 0 and raw_metadata.detector_extent is not None:
        from ptychodus.api.diffraction import BeamCenter

        extent = raw_metadata.detector_extent

        # Precedence: the parameter file, then whatever the instrument recorded, then
        # the detector midpoint. The midpoint is a guess and is logged as one -- on a
        # 1028x512 Eiger the real center sits nowhere near it.
        if params.diff_pattern_center_x is not None and params.diff_pattern_center_y is not None:
            center = BeamCenter(
                x_px=params.diff_pattern_center_x, y_px=params.diff_pattern_center_y
            )
            center_source = 'the parameter file'
        elif raw_metadata.beam_center is not None:
            center = raw_metadata.beam_center
            center_source = 'the diffraction file'
        else:
            center = BeamCenter(x_px=extent.width_px // 2, y_px=extent.height_px // 2)
            center_source = 'the detector midpoint (no value given or recorded)'

        logger.info('Beam center (%d, %d) from %s', center.x_px, center.y_px, center_source)
        region = CropRegion.from_center_extent(center, ImageExtent(crop_px, crop_px))
        read_region = region.clamp_to_detector_extent(raw_metadata.detector_extent)

        if read_region != region:
            logger.warning('Crop clamped to the detector: %s', read_region)

    num_pixels = crop_px * crop_px if crop_px > 0 else 0
    lower_counts, upper_counts = _total_counts_bounds(params, num_pixels)

    logger.info('Assembling diffraction patterns')
    assembled_data = assemble_dataset(
        raw_dataset,
        _prep_pipeline(params),
        raw_pixel_geometry=detector_pixel_geometry,
        read_region=read_region,
        total_counts_lower_bound=lower_counts,
        total_counts_upper_bound=upper_counts,
    )
    num_patterns = assembled_data.get_patterns().shape[0]

    minimum_patterns = params.minimal_num_of_diff_pattern

    if num_patterns < minimum_patterns:
        raise ValueError(f'Too few diffraction patterns: {num_patterns} < {minimum_patterns}.')

    logger.info('Reading probe positions from %s as %s', position_file, position_file_type)
    position_reader = registry.probe_position_file_readers.get_strategy_by_name(position_file_type)
    positions = position_reader.read(position_file)

    affine_values = params.init_position_affine_matrix

    if affine_values:
        flat = numpy.asarray(affine_values, dtype=float).reshape(-1)

        if flat.size != 4:
            raise ValueError('init_position_affine_matrix must hold four values.')

        transform = AffineTransform(flat[0], flat[1], 0.0, flat[2], flat[3], 0.0)
        positions = ProbePositionSequence(
            [*transform_probe_positions(positions, transform, numpy.random.default_rng(0), 0.0)]
        )

    # --- geometry ---
    wavelength_m = energy_eV_to_wavelength_m(params.beam_energy_kev * 1.0e3)
    detector_distance_m = params.det_sample_dist_m
    focus_object_distance_m = params.focal_sample_dist_m if params.near_field_ptycho else 0.0
    magnification = compute_magnification(detector_distance_m, focus_object_distance_m)
    image_extent = assembled_data.get_image_extent()

    if params.near_field_ptycho:
        # Near field: the detector pixels project onto the sample through the cone,
        # demagnified by M. Keying this off `magnification != 1.0` instead would send
        # the parallel-beam case -- no focusing optic, so M is exactly 1 and the object
        # pixel equals the detector pixel -- down the far-field branch, which returns a
        # completely different pixel size with nothing to signal the substitution.
        pixel_m = detector_pixel_geometry.width_m / magnification
        probe_geometry = ProbeGeometry(
            width_px=image_extent.width_px,
            height_px=image_extent.height_px,
            pixel_width_m=pixel_m,
            pixel_height_m=pixel_m,
        )
    else:
        probe_geometry = ProbeGeometry.from_far_field(
            assembled_data.get_pixel_geometry(),
            image_extent,
            wavelength_m=wavelength_m,
            distance_m=detector_distance_m,
        )

    logger.info('Probe geometry: %s (magnification %g)', probe_geometry, magnification)

    if params.burst_ptycho:
        positions = _cluster_burst_positions(positions, probe_geometry.pixel_width_m)

    # --- initial probe ---
    rng = numpy.random.default_rng(params.seed)
    probe_file = params.path_to_init_probe
    num_incoherent_modes = max(1, params.number_probe_modes)

    if params.use_model_FZP_probe or probe_file is None:
        preset = params.fzp_preset
        logger.info('Simulating the initial probe from zone plate preset %r', preset)
        zone_plate = registry.fresnel_zone_plates.get_strategy_by_name(preset)
        probe_array = generate_fresnel_zone_plate_probe(
            probe_geometry,
            zone_plate,
            probe_wavelength_m=wavelength_m,
            defocus_distance_m=params.fzp_defocus_m,
        ).get_array()
    else:
        logger.info('Reading the initial probe from %s', probe_file)
        probe_reader = registry.probe_file_readers.get_strategy_by_name(params.init_probe_file_type)
        probe_array = probe_reader.read(probe_file)[0].get_array()

    probe_array = _set_num_incoherent_modes(probe_array, num_incoherent_modes)

    shifts = params.init_probe_shifts_pix

    if shifts and any(shifts):
        logger.info('Shifting the initial probe by %s px', shifts)
        probe_array = _shift_probe(probe_array, (float(shifts[0]), float(shifts[1])))

    propagation_mm = params.init_probe_propagation_distance_mm

    if propagation_mm != 0.0:
        logger.info('Propagating the initial probe by %g mm', propagation_mm)
        probe_array = _propagate_probe(
            probe_array,
            distance_m=propagation_mm * 1e-3,
            wavelength_m=wavelength_m,
            pixel_geometry=probe_geometry.get_pixel_geometry(),
        )

    probe_sequence = generate_coherent_probe_modes(
        rng,
        Probe(array=probe_array, pixel_geometry=probe_geometry.get_pixel_geometry()),
        num_cmodes=max(1, params.number_opr_modes + 1),
        num_diffraction_patterns=num_patterns,
    )

    # --- initial object ---
    number_of_slices = max(1, params.number_of_slices)
    layer_spacing_m = _slice_spacings_m(params.object_thickness_m, number_of_slices)
    padding_px = round(params.obj_pad_size_m / probe_geometry.pixel_width_m)
    object_geometry = compute_object_geometry(positions, probe_geometry, padding_px=padding_px)
    object_file = params.path_to_init_object

    if object_file is None:
        object_ = generate_random_object(
            rng,
            object_geometry,
            amplitude_mean=1.0,
            amplitude_deviation=0.0,
            phase_mean=0.0,
            phase_deviation_tr=0.0,
            blur_deviation_px=0.0,
        )

        if number_of_slices > 1:
            object_ = Object(
                array=numpy.repeat(object_.get_array()[:1], number_of_slices, axis=0),
                pixel_geometry=object_.get_pixel_geometry(),
                center=object_.get_center(),
                layer_spacing_m=layer_spacing_m,
            )
    else:
        logger.info('Reading the initial object from %s', object_file)
        object_reader = registry.object_file_readers.get_strategy_by_name(
            params.init_object_file_type
        )
        array = object_reader.read(object_file).get_array()
        array = _select_layers(array, params)
        array = _fit_layer_count(array, number_of_slices, params.init_layer_append_mode)
        object_ = Object(
            array=array,
            pixel_geometry=object_geometry.get_pixel_geometry(),
            center=object_geometry.get_center(),
            layer_spacing_m=layer_spacing_m,
        )

    logger.info('Object geometry: %s (%d layer(s))', object_geometry, object_.num_layers)

    product = Product(
        metadata=ProductMetadata(
            name=f'S{scan_num:04d}',
            comments=f'Reconstructed from {args.params.name}',
            detector_distance_m=detector_distance_m,
            probe_energy_eV=params.beam_energy_kev * 1.0e3,
            probe_photon_count=float(assembled_data.get_probe_photon_count()),
            exposure_time_s=float(raw_metadata.exposure_time_s or 0.0),
            mass_attenuation_m2_kg=0.0,
            tomography_angle_deg=0.0,
            focus_object_distance_m=focus_object_distance_m,
        ),
        probe_positions=positions,
        probes=probe_sequence,
        object_=object_,
        losses=[],
    )

    options = _build_lsqml_options(params, probe_sequence, num_patterns)

    # The parameter file already names where a run belongs -- `recon_dir_base`, or the
    # per-instrument convention derived from `data_directory` -- so there is no output
    # argument to supersede it. That directory becomes the standard-layout root.
    output_directory = _output_directory(params, options, image_extent.width_px)
    diffraction_file = StandardFileLayout.DIFFRACTION.path(output_directory)
    options_file = StandardFileLayout.PTYCHI_OPTIONS.path(output_directory)
    product_file = StandardFileLayout.PRODUCT.path(output_directory)
    num_sync_epochs = args.num_sync_epochs or params.save_freq_iterations

    if args.dry_run:
        logger.info('Resolved output directory: %s', output_directory)
        logger.info('Would write %s', product_file)
        logger.info(
            'Would write checkpoints like %s',
            StandardFileLayout.PRODUCT.checkpoint_path(output_directory, num_sync_epochs),
        )

        if not args.no_save_diffraction:
            logger.info('Would write %s', diffraction_file)

        # The options file records the ALIGNED options, which only exist past the call
        # this dry run stops before, so it is the one artifact a dry run cannot preview.
        logger.info('Would write %s once the options are aligned', options_file)
        logger.info(
            'Resolved options: %d epochs, batch %d (%s), %s noise, probe %s, object %s',
            options.reconstructor_options.num_epochs,
            options.reconstructor_options.batch_size,
            options.reconstructor_options.batching_mode.name,
            options.reconstructor_options.noise_model.name,
            probe_sequence.get_array().shape,
            object_.get_array().shape,
        )
        logger.info('Nothing written: stopping before the reconstruction.')
        return 0

    output_directory.mkdir(parents=True, exist_ok=True)

    # Ahead of the reconstruction, so an interrupted run still leaves the directory
    # usable: the assembled patterns are the one artifact that cannot be rebuilt without
    # the raw beamline files.
    if args.no_save_diffraction:
        logger.info('Skipping %s as requested', diffraction_file.name)
    else:
        logger.info('Writing %s', diffraction_file)
        save_diffraction_data(diffraction_file, assembled_data)

    task_options = align_task_options_with_product(options, product)
    task_options.check()

    # The aligned options are the ones that ran: they carry the object pixel size, the
    # wavelength and the slice spacings the product supplied.
    logger.info('Writing %s', options_file)
    options_file.write_text(dump_task_options(task_options))

    reconstruct_input = prepare_reconstruct_input(assembled_data, product)
    num_epochs = int(task_options.reconstructor_options.num_epochs)

    logger.info(
        'Starting reconstruction: %d patterns, %d epochs, sync every %d',
        reconstruct_input.diffraction_patterns.shape[0],
        num_epochs,
        num_sync_epochs,
    )

    final_output = None

    for output in reconstruct_with_ptychi(
        reconstruct_input, task_options, num_sync_epochs=num_sync_epochs
    ):
        losses = output.product.losses
        last_loss = losses[-1].value if losses else float('nan')
        logger.info('Epoch %d/%d: loss=%.6g', output.progress, num_epochs, last_loss)

        checkpoint_file = StandardFileLayout.PRODUCT.checkpoint_path(
            output_directory, output.progress
        )
        save_product(checkpoint_file, output.product)
        final_output = output

    if final_output is None:
        raise RuntimeError('Reconstruction produced no output.')

    save_product(product_file, final_output.product)
    logger.info('Saved the reconstructed product to %s', product_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
