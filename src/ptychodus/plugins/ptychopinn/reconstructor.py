"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoPINN.

Zero backend imports: no ``ptycho_torch``, no lightning, no torch, and so no
GPU context. The declared options are translated here into a plain mapping of
the backend's documented resolver names, and the child turns that mapping into
resolved configuration. See :mod:`._subprocess` for the child entry points.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Final

from ptychodus.api.io import save_ptychopinn_training_data
from pydantic import BaseModel

from ptychodus.api.reconstruct import ReconstructInput, SubprocessReconstructor

from .options import PtychoPINNOptions, default_options

__all__ = [
    'build_reconstructor',
]

logger = logging.getLogger(__name__)

MODEL_FILE_NAME: Final[str] = 'wts.h5.zip'


# 'None' in a settings combo box means "no auxiliary loss", which the backend
# spells as the absent value rather than a string.
_NO_AUXILIARY_LOSS = 'none'


def _optional_auxiliary_loss(value: str) -> str | None:
    return None if value.casefold() == _NO_AUXILIARY_LOSS else value


def _declared(options: BaseModel) -> PtychoPINNOptions:
    """Narrow the options a call was given to what this backend declares.

    Checked rather than asserted: the dispatching layer validates the type
    against what ``get_default_options`` returns, so a mismatch here means
    those two have drifted apart, and saying so beats an attribute error
    several frames later.
    """
    if not isinstance(options, PtychoPINNOptions):
        raise TypeError(f'Expected PtychoPINNOptions, got {type(options).__name__}.')

    return options


def _build_settings(model_training_mode: str, options: PtychoPINNOptions) -> dict[str, Any]:
    """Translate the declared options into the backend's resolver vocabulary.

    The result is a flat mapping of documented setting names to values. Names
    absent from it keep the backend's own defaults, which is deliberate in two
    cases:

    ``nphotons`` is omitted. Ptychodus exports diffraction patterns that are
    already photon counts and declares them as such, so the source and the
    bundle target agree and nothing needs rescaling. Supplying ``nphotons``
    would mark the photon count as explicitly chosen and trigger an
    amplitude-to-counts conversion of data that is already counts, silently
    changing its scale.

    ``loss_function`` and ``nll`` are omitted because the backend derives both
    from ``torch_loss_mode``; supplying them directly is rejected.

    Enum members are rendered as their values, because the mapping is pickled
    to a child that reads it as plain configuration.
    """
    data = options.data
    model = options.model
    training = options.training
    inference = options.inference

    object_big = model.object_big
    gridsize = data.gridsize

    # A group holds gridsize**2 members, so the neighbour search must return at
    # least that many candidates minus the centre itself. The backend raises
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
    """Resolve the recorded model path to a directory the child can load from.

    Accepts the bundle directory itself, a ``wts.h5.zip`` inside a bundle
    directory, or an outer zip archive to unpack into a fresh tempdir. Runs
    parent-side because it is pure zipfile/tempfile work with no GPU
    involvement.
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
    """Archive the loaded bundle directory (or copy the bundle .zip) to ``dest``.

    The child records either a bundle directory (right after training) or a
    ``wts.h5.zip`` file (right after inference load). This function normalises
    to a zip archive at ``dest``.
    """
    if loaded_from.is_dir():
        archive_stem = str(dest.with_suffix(''))
        logger.debug(f'Archiving bundle {loaded_from!r} -> {dest!r}')
        shutil.make_archive(archive_stem, 'zip', root_dir=loaded_from)
        return

    if loaded_from.name == MODEL_FILE_NAME:
        # wts.h5.zip inside a bundle dir -- archive the parent dir so the
        # saved artifact is a self-contained bundle.
        archive_stem = str(dest.with_suffix(''))
        shutil.make_archive(archive_stem, 'zip', root_dir=loaded_from.parent)
        return

    if loaded_from.suffix == '.zip':
        shutil.copyfile(loaded_from, dest)
        return

    raise RuntimeError(f'Cannot save PtychoPINN model: unrecognized source path {loaded_from!r}.')


def build_reconstructor(model_training_mode: str) -> SubprocessReconstructor:
    """Build a :class:`SubprocessReconstructor` for one PtychoPINN mode.

    ``model_training_mode`` is 'Unsupervised' or 'Supervised' and becomes the
    reconstructor's display name plus the mode string the backend resolves.
    """
    # Source model path -> extracted bundle directory, so repeated reconstructs
    # against the same model unpack the archive only once.
    bundle_dir_cache: dict[Path, Path] = dict()

    def resolve_model_path(loaded_model_path: Path | None) -> Path:
        """The bundle directory the child loads from, unpacked at most once.

        A bundle arrives as an archive the user picked; the child wants the
        directory inside it. Unpacking here rather than child-side means
        repeated reconstructs against one model pay for it once, and a
        reconstruct with nothing loaded says so before a process is spawned.
        """
        if loaded_model_path is None:
            raise RuntimeError('Cannot reconstruct: no PtychoPINN model has been loaded.')

        try:
            return bundle_dir_cache[loaded_model_path]
        except KeyError:
            bundle_dir = _extract_bundle_dir(loaded_model_path)
            bundle_dir_cache[loaded_model_path] = bundle_dir
            return bundle_dir

    def dump_reconstruct_options(
        parameters: ReconstructInput, options: BaseModel
    ) -> dict[str, Any]:
        declared = _declared(options)
        return {
            'patch_weighting': str(declared.inference.patch_weighting_method),
            'inference_batch_size': declared.inference.batch_size,
            'accelerator': str(declared.training.accelerator),
            'num_workers': declared.training.num_dataloader_workers,
        }

    def dump_train_options(options: BaseModel) -> dict[str, Any]:
        declared = _declared(options)
        return {
            'settings': _build_settings(model_training_mode, declared),
            'accelerator': str(declared.training.accelerator),
            'num_workers': declared.training.num_dataloader_workers,
        }

    def export_training_data(
        file_path: Path, parameters: ReconstructInput, options: BaseModel
    ) -> None:
        save_ptychopinn_training_data(file_path, parameters)

    return SubprocessReconstructor(
        name=model_training_mode,
        default_options=default_options,
        dump_reconstruct_options=dump_reconstruct_options,
        resolve_model_path=resolve_model_path,
        is_trainable=True,
        dump_train_options=dump_train_options,
        model_file_filter='Zipped Archive (*.zip)',
        model_file_extension='.zip',
        training_data_file_filter='NumPy Zipped Archive (*.npz)',
        export_training_data=export_training_data,
        save_model=_save_model_bundle,
    )
