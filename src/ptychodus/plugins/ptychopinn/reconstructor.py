"""PtychoPINN as a ptychodus :class:`TrainableReconstructor`.

Importing this module imports no ``ptycho_torch``, no lightning and no torch, and
so acquires no GPU context: every library import is local to the method or
helper that needs it. The declared options are translated here into a plain
mapping of the library's documented resolver names, which the library turns into
resolved configuration.

Two operations:

- :meth:`PtychoPINNReconstructor.reconstruct` -- run one inference pass with the
  loaded bundle and yield a single :class:`ReconstructOutput`.
- :meth:`PtychoPINNReconstructor.train` -- run one training session through the
  library's public dataset-to-bundle door on every training archive in the input
  directory, then yield the final :class:`TrainOutput`, naming the saved bundle.
"""

from __future__ import annotations

import csv
import importlib
import logging
import os
import shutil
import tempfile
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Final

import numpy
from pydantic import BaseModel

from ptychodus.api.object import Object, ObjectCenter, ObjectGeometry, ObjectPosition
from ptychodus.api.preprocess.diffraction import inpaint_bad_pixels
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstruct import (
    ReconstructInput,
    ReconstructOutput,
    TrainableReconstructor,
    TrainOutput,
)
from ptychodus.api.typing import ComplexArrayType

from .options import PtychoPINNOptions, default_options
from .training_file import (
    find_ptychopinn_training_archives,
    merge_ptychopinn_training_archives,
)

__all__ = [
    'PtychoPINNReconstructor',
]

logger = logging.getLogger(__name__)

MODEL_FILE_NAME: Final[str] = 'wts.h5.zip'


# 'None' in a settings combo box means "no auxiliary loss", which the library
# spells as the absent value rather than a string.
_NO_AUXILIARY_LOSS = 'none'


def _optional_auxiliary_loss(value: str) -> str | None:
    return None if value.casefold() == _NO_AUXILIARY_LOSS else value


def _declared(options: BaseModel) -> PtychoPINNOptions:
    """Narrow the options a call was given to what this reconstructor declares.

    Checked rather than asserted: the dispatching layer validates the type
    against what ``get_default_options`` returns, so a mismatch here means
    those two have drifted apart, and saying so beats an attribute error
    several frames later.
    """
    if not isinstance(options, PtychoPINNOptions):
        raise TypeError(f'Expected PtychoPINNOptions, got {type(options).__name__}.')

    return options


def _build_settings(model_training_mode: str, options: PtychoPINNOptions) -> dict[str, Any]:
    """Translate the declared options into the library's resolver vocabulary.

    The result is a flat mapping of documented setting names to values. Names
    absent from it keep the library's own defaults, which is deliberate in two
    cases:

    ``nphotons`` is omitted. Ptychodus exports diffraction patterns that are
    already photon counts and declares them as such, so the source and the
    bundle target agree and nothing needs rescaling. Supplying ``nphotons``
    would mark the photon count as explicitly chosen and trigger an
    amplitude-to-counts conversion of data that is already counts, silently
    changing its scale.

    ``loss_function`` and ``nll`` are omitted because the library derives both
    from ``torch_loss_mode``; supplying them directly is rejected.

    Enum members are rendered as their values, because the library reads the
    mapping as plain configuration.
    """
    data = options.data
    model = options.model
    training = options.training
    inference = options.inference

    object_big = model.object_big
    gridsize = data.gridsize

    # A group holds gridsize**2 members, so the neighbour search must return at
    # least that many candidates minus the centre itself. The library raises
    # rather than shrinking the group, so clamp here where the option is read.
    group_size = gridsize * gridsize
    neighbor_count = max(data.neighbor_count, group_size - 1)

    gradient_clip_val = training.gradient_clip_val

    return {
        # -- data ------------------------------------------------------------
        'N': data.model_size,
        'gridsize': gridsize,
        'neighbor_count': neighbor_count,
        'group_padding_step': data.group_padding_step,
        'normalize': str(data.data_normalization_mode),
        'probe_normalize': data.normalize_probe,
        'probe_scale': data.probe_scale,
        'data_scaling': str(data.data_scaling_method),
        'phase_subtraction': data.subtract_mean_phase,
        'x_bounds': (data.x_lower_bound, data.x_upper_bound),
        'y_bounds': (data.y_lower_bound, data.y_upper_bound),
        # -- model -----------------------------------------------------------
        'mode': model_training_mode,
        # The object policy is two fields that must agree; only these two
        # pairings are accepted, and the deprecated single-boolean spelling is
        # derived from them rather than supplied alongside.
        'object_layout': 'grouped_patches' if object_big else 'single_patch',
        'training_canvas': 'relative_overlap' if object_big else 'independent',
        'probe_big': model.probe_big,
        'amp_activation': str(model.amplitude_activation_function),
        'cbam_encoder': model.cbam_encoder,
        'use_shared_decoder': model.use_shared_decoder,
        'intensity_scale_trainable': model.intensity_scale_trainable,
        'intensity_scale': model.intensity_scale,
        'max_position_jitter': model.max_position_jitter,
        'num_datasets': model.num_datasets,
        'amp_loss': _optional_auxiliary_loss(str(model.auxiliary_amplitude_loss)),
        'phase_loss': _optional_auxiliary_loss(str(model.auxiliary_phase_loss)),
        'amp_loss_coeff': model.auxiliary_amplitude_loss_coeff,
        'phase_loss_coeff': model.auxiliary_phase_loss_coeff,
        'n_filters_scale': model.num_filters_scale,
        'eca_decoder': model.eca_decoder,
        'batch_norm': model.use_batch_normalization,
        'edge_pad': model.edge_pad,
        'decoder_last_c_outer_fraction': model.decoder_last_c_outer_fraction,
        'cbam_bottleneck': model.cbam_bottleneck,
        'cbam_decoder': model.cbam_decoder,
        'spatial_decoder': model.spatial_decoder,
        'decoder_spatial_kernel': model.decoder_spatial_kernel,
        'eca_encoder': model.eca_encoder,
        'offset': model.offset,
        # -- training --------------------------------------------------------
        'epochs': training.epochs,
        'batch_size': training.batch_size,
        'learning_rate': training.learning_rate,
        'training_groups': training.training_groups,
        'accum_steps': training.gradient_accumulation_steps,
        'epochs_fine_tune': training.epochs_finetune,
        'fine_tune_gamma': training.finetune_gamma,
        'gradient_clip_val': gradient_clip_val if gradient_clip_val > 0.0 else None,
        'scheduler': str(training.learning_rate_scheduler),
        'lr_warmup_epochs': training.learning_rate_warmup_epochs,
        'lr_min_ratio': training.minimum_learning_rate_ratio,
        'notes': training.notes,
        'model_name': training.model_name,
        'torch_loss_mode': str(model.loss_function).casefold(),
        'stage_1_epochs': training.stage_1_epochs,
        'stage_2_epochs': training.stage_2_epochs,
        'stage_3_epochs': training.stage_3_epochs,
        'stage_3_lr_factor': training.stage_3_lr_factor,
        'physics_weight_schedule': str(training.physics_weight_schedule),
        # -- inference -------------------------------------------------------
        'middle_trim': inference.middle_trim,
        'inference_batch_size': inference.batch_size,
        'experiment_number': inference.experiment_number,
        'pad_eval': inference.pad_eval,
        'window': inference.window,
        'patch_weighting': str(inference.patch_weighting_method),
    }


def _extract_bundle_dir(model_bundle_path: Path) -> Path:
    """Resolve a trained-model path to the bundle directory the library loads from.

    Accepts the bundle directory itself, a ``wts.h5.zip`` inside a bundle
    directory, or an outer zip archive to unpack into a fresh tempdir. Pure
    zipfile/tempfile work with no GPU involvement.
    """
    if model_bundle_path.is_dir():
        return model_bundle_path

    if model_bundle_path.name == MODEL_FILE_NAME:
        return model_bundle_path.parent

    if model_bundle_path.suffix == '.zip':
        bundle_dir = Path(tempfile.mkdtemp(prefix='ptychopinn-bundle-'))
        logger.debug(f'Extracting bundle "{model_bundle_path}" -> "{bundle_dir}"')

        with zipfile.ZipFile(model_bundle_path) as archive:
            archive.extractall(bundle_dir)

        return bundle_dir

    logger.warning(
        f'PtychoPINN expected the file name {MODEL_FILE_NAME!r}; got {model_bundle_path.name!r}.'
    )
    return model_bundle_path.parent


def _save_model_bundle(loaded_from: Path, dest: Path) -> None:
    """Write the trained model recorded at ``loaded_from`` to ``dest`` as a zip of its bundle.

    The recorded path is either a bundle directory (as training reports it) or
    a ``wts.h5.zip`` file (as a user picks it). A bundle is that one
    archive: it carries its own manifest, and PtychoPINN loads nothing else from
    the directory holding it. So only it is saved, wrapped in the outer zip
    :func:`_extract_bundle_dir` unpacks, and whatever else training left beside
    it -- logs, checkpoints, metrics -- stays behind.

    The outer zip is assembled beside ``dest`` and renamed onto it, so ``dest``
    is written exactly as named, a ``dest`` inside the bundle directory is never
    part of what is archived, and a failed write leaves nothing at ``dest``.
    """
    if loaded_from.is_dir() or loaded_from.name == MODEL_FILE_NAME:
        bundle_dir = loaded_from if loaded_from.is_dir() else loaded_from.parent
        archive = bundle_dir / MODEL_FILE_NAME

        if not archive.is_file():
            raise RuntimeError(f'Cannot save PtychoPINN model: {archive} is not a file.')

        logger.debug(f'Archiving bundle {archive!r} -> {dest!r}')

        with tempfile.NamedTemporaryFile(
            dir=dest.parent, prefix=f'.{dest.name}.', suffix='.tmp', delete=False
        ) as handle:
            staging = Path(handle.name)

        try:
            with zipfile.ZipFile(staging, 'w') as outer:
                outer.write(archive, arcname=MODEL_FILE_NAME)

            staging.replace(dest)
        except BaseException:
            staging.unlink(missing_ok=True)
            raise

        return

    if loaded_from.suffix == '.zip':
        shutil.copyfile(loaded_from, dest)
        return

    raise RuntimeError(f'Cannot save PtychoPINN model: unrecognized source path {loaded_from!r}.')


# Ptychodus exports detector patterns as photon counts and labels them as such,
# so the exported source and the bundle target agree and nothing is rescaled.
_SCALE_CONTRACT_VERSION = 'ci_intensity_v2'
_MEASUREMENT_DOMAIN = 'count_intensity'


def _resolve_device(accelerator: str) -> str:
    """Map the configured accelerator onto a torch device string.

    Runs with the computation because deciding between CUDA and CPU means asking
    torch whether a device is actually present, which only the process computing
    may do.
    """
    import torch

    requested = accelerator.strip().casefold()

    if requested == 'cpu':
        return 'cpu'

    if requested == 'mps':
        return 'mps'

    if torch.cuda.is_available():
        return 'cuda'

    if requested in ('gpu', 'cuda'):
        logger.warning(
            'Accelerator %r requested but no CUDA device is available; using CPU.', accelerator
        )

    return 'cpu'


def _canvas_center(
    object_geometry: ObjectGeometry, canvas_anchor: Mapping[str, Any], canvas_shape: Any
) -> ObjectCenter:
    """World position of the returned canvas's center pixel, ``(N - 1) / 2``.

    PtychoPINN anchors pixel ``canvas_size // 2`` of its canvas on the scan's center of
    mass, ``scan_com``, which it measures in the input object's pixel coordinates -- the
    coordinates :func:`_build_arrays` hands it. Canvas pixel ``i`` therefore sits at input
    pixel ``scan_com + i - canvas_size // 2``, whatever size the canvas came back.
    """
    com_x_px, com_y_px = (float(value) for value in numpy.asarray(canvas_anchor['scan_com'])[:2])
    height_px, width_px = (int(size) for size in canvas_shape)
    center = object_geometry.map_coordinates_object_to_probe(
        ObjectPosition(
            index=-1,
            x_px=com_x_px + (width_px - 1) / 2 - width_px // 2,
            y_px=com_y_px + (height_px - 1) / 2 - height_px // 2,
        )
    )
    return ObjectCenter(x_m=center.x_m, y_m=center.y_m)


def _build_arrays(parameters: Any) -> dict[str, Any]:
    """Convert a ReconstructInput into the library's flat-acquisition mapping.

    Mirrors the training-data export so inference sees the same preprocessing
    and the same declared measurement domain the model was trained on.
    """
    object_geometry = parameters.product.object_.get_geometry()
    positions = parameters.product.probe_positions
    xcoords = object_geometry.map_probe_positions_to_object_x_px(positions)
    ycoords = object_geometry.map_probe_positions_to_object_y_px(positions)

    return {
        'xcoords': xcoords,
        'ycoords': ycoords,
        'xcoords_start': xcoords,
        'ycoords_start': ycoords,
        # Repaired the same way exported training data is.
        'diff3d': inpaint_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels),
        'probeGuess': parameters.product.probes.get_mean_probe().get_array(),
        'objectGuess': parameters.product.object_.get_layer(0),
        'scan_index': numpy.zeros(len(positions), dtype=int),
        'scale_contract_version': _SCALE_CONTRACT_VERSION,
        'measurement_domain': _MEASUREMENT_DOMAIN,
    }


def _latest_metrics_csv(output_path: Path) -> Path | None:
    """Return the newest Lightning CSV metrics file, or None if there is none."""
    candidates: list[tuple[int, Path]] = []

    for version_dir in (output_path / 'lightning_logs').glob('version_*'):
        metrics_path = version_dir / 'metrics.csv'
        if not metrics_path.is_file():
            continue
        try:
            version = int(version_dir.name.removeprefix('version_'))
        except ValueError:
            continue
        candidates.append((version, metrics_path))

    if not candidates:
        return None

    return max(candidates)[1]


def _select_loss_column(field_names: Sequence[str], infix: str) -> str | None:
    """Return the single loss column for one split, or None if it is ambiguous.

    The library assembles these names from the loss mode and the optional
    auxiliary terms, so they are matched by shape rather than enumerated.
    """
    matches = [name for name in field_names if infix in name and name.endswith('_loss')]

    if len(matches) == 1:
        return matches[0]

    if matches:
        logger.warning('Ambiguous %r loss columns in the training log: %s', infix, matches)
    else:
        logger.warning('No %r loss column found in the training log.', infix)

    return None


def _read_epoch_losses(output_path: Path) -> tuple[list[LossValue], list[LossValue]]:
    """Read per-epoch training and validation losses from the Lightning CSV log.

    The public training door takes no callback, so losses are recovered after
    the run rather than streamed during it. A missing or unreadable log costs
    the loss curves, never the trained model.
    """
    metrics_path = _latest_metrics_csv(output_path)

    if metrics_path is None:
        logger.warning('No Lightning metrics log found under "%s".', output_path)
        return [], []

    with metrics_path.open(newline='') as metrics_file:
        rows = list(csv.DictReader(metrics_file))
        field_names = rows[0].keys() if rows else []

    train_column = _select_loss_column(list(field_names), '_train')
    val_column = _select_loss_column(list(field_names), '_val')

    def collect(column: str | None) -> list[LossValue]:
        if column is None:
            return []

        values: list[LossValue] = []

        for row in rows:
            raw_value = row.get(column)
            raw_epoch = row.get('epoch')

            if not raw_value or raw_epoch is None or raw_epoch == '':
                continue

            try:
                values.append(LossValue(epoch=int(float(raw_epoch)), value=float(raw_value)))
            except ValueError:
                continue

        return values

    return collect(train_column), collect(val_column)


def _compute_truth_patches(file_path: Path) -> ComplexArrayType:
    """Each frame's ground-truth object patch, cut from the archive's objectGuess by the library.

    Cutting with the library's own grouping keeps the patch convention its supervised loss
    expects; one-member groups taken in order give one patch per frame.
    """
    # ptycho ships beside ptycho_torch, without type information
    raw_data = importlib.import_module('ptycho.raw_data').RawData.from_file(str(file_path))
    num_frames, patch_size = raw_data.diff3d.shape[:2]
    grouped = raw_data.generate_grouped_data(
        patch_size, nsamples=num_frames, gridsize=1, sequential_sampling=True
    )

    if grouped.get('Y') is None:
        raise ValueError(
            f'{file_path.name} holds no objectGuess, which supervised training of several '
            'archives cuts its ground truth from.'
        )

    return numpy.asarray(grouped['Y'])[..., 0]


class PtychoPINNReconstructor(TrainableReconstructor):
    """One PtychoPINN mode, 'Unsupervised' or 'Supervised'.

    The mode is both the reconstructor's display name and the mode string the
    library resolves. Loading a trained model records it and unpacks its archive once,
    so repeated reconstructs against one model pay for that once.
    """

    def __init__(self, model_training_mode: str) -> None:
        super().__init__()
        self._mode = model_training_mode
        self._model_path: Path | None = None
        self._bundle_dir: Path | None = None

    @property
    def name(self) -> str:
        return self._mode

    def get_default_options(self) -> BaseModel:
        return default_options()

    def supports_distributed_training(self) -> bool:
        # Lightning's DDP strategy joins the process group a launcher such as torchrun
        # describes, and PtychoPINN persists its bundle from global rank zero alone.
        return True

    def is_model_loaded(self) -> bool:
        return self._model_path is not None

    def get_model_file_description(self) -> str:
        return 'Zipped Archive'

    def get_model_file_extensions(self) -> tuple[str, ...]:
        return ('.zip',)

    def load_model(self, file_path: Path) -> None:
        self._bundle_dir = _extract_bundle_dir(file_path)
        self._model_path = file_path

    def save_model(self, file_path: Path) -> None:
        if self._model_path is None:
            raise RuntimeError(
                f'Cannot save {self._mode} model: no model has been loaded or trained.'
            )

        _save_model_bundle(self._model_path, file_path)

    def reconstruct(
        self, input_: ReconstructInput, *, options: BaseModel
    ) -> Iterator[ReconstructOutput]:
        """Run one inference pass with the loaded bundle and yield its single output."""
        if self._bundle_dir is None:
            raise RuntimeError('Cannot reconstruct: no PtychoPINN model has been loaded.')

        declared = _declared(options)

        from ptycho_torch.inference import ReconstructionRuntimeParams, reconstruct_from_arrays
        from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

        loaded_model, _bundle_params = load_inference_bundle_torch(self._bundle_dir)

        bundle_mode = getattr(loaded_model.model_config, 'mode', None)
        if bundle_mode is not None and bundle_mode != self._mode:
            logger.warning(
                'Loaded bundle mode %r does not match reconstructor mode %r; '
                'predictions may be inconsistent.',
                bundle_mode,
                self._mode,
            )

        # The bundle owns geometry and trimming; only the presentation knobs below
        # are ptychodus's to choose at reconstruct time.
        inference_config = replace(
            loaded_model.inference_config,
            patch_weighting=str(declared.inference.patch_weighting_method),
            batch_size=declared.inference.batch_size,
        )

        device = _resolve_device(str(declared.training.accelerator))

        with tempfile.TemporaryDirectory(prefix='ptychopinn-reconstruct-') as workspace:
            result = reconstruct_from_arrays(
                loaded_model,
                _build_arrays(input_),
                runtime_params=ReconstructionRuntimeParams(
                    # Replaced during staging from the model and the arrays; the
                    # call still requires the fields to be present.
                    data_config=loaded_model.data_config,
                    training_config=loaded_model.training_config,
                    inference_config=inference_config,
                    source_metadata={},
                    quiet=True,
                ),
                workspace=workspace,
                device=device,
                num_workers=declared.training.num_dataloader_workers,
            )
            # Snapshot before the workspace unlinks its memory-mapped backing.
            object_out_array = numpy.squeeze(numpy.asarray(result.complex_canvas))
            canvas_anchor = dict(result.canvas_anchor)

        object_in = input_.product.object_
        object_out = Object(
            array=object_out_array,
            layer_spacing_m=object_in.layer_spacing_m,
            pixel_geometry=object_in.get_pixel_geometry(),
            center=_canvas_center(
                object_in.get_geometry(), canvas_anchor, object_out_array.shape[-2:]
            ),
        )

        losses: Sequence[LossValue] = []
        product = Product(
            metadata=input_.product.metadata,
            probe_positions=input_.product.probe_positions,
            probes=input_.product.probes,
            object_=object_out,
            losses=losses,
        )

        yield ReconstructOutput(product=product, progress=1, progress_goal=1)

    def train(
        self, input_directory: Path, output_directory: Path, *, options: BaseModel
    ) -> Iterator[TrainOutput]:
        """Run one training session and yield its final output, naming the saved bundle.

        Hands the translated settings mapping to the library's public
        dataset-to-bundle training door, on the devices this process can see.
        That door takes one archive, so several in the input directory are first
        merged into one object bank in a private directory under the output
        directory -- private so that ranks of a distributed launch never share it
        -- and removed afterwards. A supervised bank carries ground-truth patches;
        an unsupervised one needs none.
        """
        declared = _declared(options)
        settings = _build_settings(self._mode, declared)

        from ptycho_torch.execution_request import ExecutionRequest
        from ptycho_torch.train import train

        # Under a distributed launcher this process is one rank of LOCAL_WORLD_SIZE
        # on its node, and Lightning's TorchElastic environment insists the device
        # count say so. Otherwise 'auto' resolves to the number of devices this
        # process can see, so no manual clamping is needed.
        local_world_size = os.environ.get('LOCAL_WORLD_SIZE')
        execution_values: dict[str, Any] = {
            'accelerator': str(declared.training.accelerator),
            'devices': 'auto' if local_world_size is None else int(local_world_size),
            'num_workers': declared.training.num_dataloader_workers,
            'enable_progress_bar': False,
        }

        archives = find_ptychopinn_training_archives(input_directory)
        merge_directory: Path | None = None

        try:
            if len(archives) == 1:
                dataset = archives[0]
            else:
                output_directory.mkdir(parents=True, exist_ok=True)
                merge_directory = Path(tempfile.mkdtemp(prefix='.merged-', dir=output_directory))
                is_supervised = settings.get('mode') == 'Supervised'
                dataset = merge_ptychopinn_training_archives(
                    archives,
                    merge_directory / 'train_data',
                    truth_patches=_compute_truth_patches if is_supervised else None,
                )
                logger.info(f'Merged {len(archives)} training archives into one object bank.')

            bundle_path = train(
                dataset,
                output_directory,
                settings,
                execution_config=ExecutionRequest(
                    values=execution_values,
                    explicit_fields=frozenset(execution_values),
                ),
            )
        finally:
            if merge_directory is not None:
                shutil.rmtree(merge_directory, ignore_errors=True)

        training_loss, validation_loss = _read_epoch_losses(output_directory)

        # Training has finished by now, so report it as complete even when the log
        # could not be read -- a run that produced a bundle but no loss curve is
        # still a finished run, and zero progress would read as a failure.
        progress = len(training_loss) or int(settings.get('epochs', 0))

        yield TrainOutput(
            training_loss=training_loss,
            validation_loss=validation_loss,
            progress=progress,
            progress_goal=progress,
            trained_model=Path(bundle_path),
        )
