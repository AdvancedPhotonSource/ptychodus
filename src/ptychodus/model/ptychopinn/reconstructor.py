"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoPINN.

Zero backend imports: no ``ptycho_torch``, no lightning, no torch, and so no
GPU context. Settings are translated here into a plain mapping of the
backend's documented resolver names, and the child turns that mapping into
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

from ._payload import ReconstructPayload, TrainPayload
from .settings import (
    PtychoPINNDataSettings,
    PtychoPINNInferenceSettings,
    PtychoPINNModelSettings,
    PtychoPINNTrainingSettings,
)

__all__ = [
    'build_reconstructor',
]

logger = logging.getLogger(__name__)

MODEL_FILE_NAME: Final[str] = 'wts.h5.zip'

_RECONSTRUCT_ENTRY = 'ptychodus.model.ptychopinn._subprocess:run_reconstruct'
_TRAIN_ENTRY = 'ptychodus.model.ptychopinn._subprocess:run_train'

# 'None' in a settings combo box means "no auxiliary loss", which the backend
# spells as the absent value rather than a string.
_NO_AUXILIARY_LOSS = 'none'


def _optional_auxiliary_loss(value: str) -> str | None:
    return None if value.casefold() == _NO_AUXILIARY_LOSS else value


def _build_settings(
    model_training_mode: str,
    data_settings: PtychoPINNDataSettings,
    model_settings: PtychoPINNModelSettings,
    training_settings: PtychoPINNTrainingSettings,
    inference_settings: PtychoPINNInferenceSettings,
) -> dict[str, Any]:
    """Translate ptychodus settings into the backend's resolver vocabulary.

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
    """
    object_big = model_settings.object_big.get_value()
    gridsize = data_settings.gridsize.get_value()

    # A group holds gridsize**2 members, so the neighbour search must return at
    # least that many candidates minus the centre itself. The backend raises
    # rather than shrinking the group, so clamp here where the setting is read.
    group_size = gridsize * gridsize
    neighbor_count = max(data_settings.neighbor_count.get_value(), group_size - 1)

    gradient_clip_val = training_settings.gradient_clip_val.get_value()
    loss_function = model_settings.loss_function.get_value()

    return {
        # -- data ------------------------------------------------------------
        'N': data_settings.model_size.get_value(),
        'gridsize': gridsize,
        'neighbor_count': neighbor_count,
        'group_padding_step': data_settings.group_padding_step.get_value(),
        'normalize': data_settings.data_normalization_mode.get_value(),
        'probe_normalize': data_settings.normalize_probe.get_value(),
        'probe_scale': data_settings.probe_scale.get_value(),
        'data_scaling': data_settings.data_scaling_method.get_value(),
        'phase_subtraction': data_settings.subtract_mean_phase.get_value(),
        'x_bounds': (
            data_settings.x_lower_bound.get_value(),
            data_settings.x_upper_bound.get_value(),
        ),
        'y_bounds': (
            data_settings.y_lower_bound.get_value(),
            data_settings.y_upper_bound.get_value(),
        ),
        # -- model -----------------------------------------------------------
        'mode': model_training_mode,
        # The object policy is two fields that must agree; only these two
        # pairings are accepted, and the deprecated single-boolean spelling is
        # derived from them rather than supplied alongside.
        'object_layout': 'grouped_patches' if object_big else 'single_patch',
        'training_canvas': 'relative_overlap' if object_big else 'independent',
        'probe_big': model_settings.probe_big.get_value(),
        'amp_activation': model_settings.amplitude_activation_function.get_value(),
        'cbam_encoder': model_settings.cbam_encoder.get_value(),
        'use_shared_decoder': model_settings.use_shared_decoder.get_value(),
        'intensity_scale_trainable': model_settings.intensity_scale_trainable.get_value(),
        'intensity_scale': model_settings.intensity_scale.get_value(),
        'max_position_jitter': model_settings.max_position_jitter.get_value(),
        'num_datasets': model_settings.num_datasets.get_value(),
        'amp_loss': _optional_auxiliary_loss(model_settings.auxiliary_amplitude_loss.get_value()),
        'phase_loss': _optional_auxiliary_loss(model_settings.auxiliary_phase_loss.get_value()),
        'amp_loss_coeff': model_settings.auxiliary_amplitude_loss_coeff.get_value(),
        'phase_loss_coeff': model_settings.auxiliary_phase_loss_coeff.get_value(),
        'n_filters_scale': model_settings.num_filters_scale.get_value(),
        'eca_decoder': model_settings.eca_decoder.get_value(),
        'batch_norm': model_settings.use_batch_normalization.get_value(),
        'edge_pad': model_settings.edge_pad.get_value(),
        'decoder_last_c_outer_fraction': (model_settings.decoder_last_c_outer_fraction.get_value()),
        'cbam_bottleneck': model_settings.cbam_bottleneck.get_value(),
        'cbam_decoder': model_settings.cbam_decoder.get_value(),
        'spatial_decoder': model_settings.spatial_decoder.get_value(),
        'decoder_spatial_kernel': model_settings.decoder_spatial_kernel.get_value(),
        'eca_encoder': model_settings.eca_encoder.get_value(),
        'offset': model_settings.offset.get_value(),
        # -- training --------------------------------------------------------
        'epochs': training_settings.epochs.get_value(),
        'batch_size': training_settings.batch_size.get_value(),
        'learning_rate': training_settings.learning_rate.get_value(),
        'training_groups': training_settings.training_groups.get_value(),
        'accum_steps': training_settings.gradient_accumulation_steps.get_value(),
        'epochs_fine_tune': training_settings.epochs_finetune.get_value(),
        'fine_tune_gamma': training_settings.finetune_gamma.get_value(),
        'gradient_clip_val': gradient_clip_val if gradient_clip_val > 0.0 else None,
        'scheduler': training_settings.learning_rate_scheduler.get_value(),
        'lr_warmup_epochs': training_settings.learning_rate_warmup_epochs.get_value(),
        'lr_min_ratio': training_settings.minimum_learning_rate_ratio.get_value(),
        'notes': training_settings.notes.get_value(),
        'model_name': training_settings.model_name.get_value(),
        'torch_loss_mode': loss_function.casefold(),
        'stage_1_epochs': training_settings.stage_1_epochs.get_value(),
        'stage_2_epochs': training_settings.stage_2_epochs.get_value(),
        'stage_3_epochs': training_settings.stage_3_epochs.get_value(),
        'stage_3_lr_factor': training_settings.stage_3_lr_factor.get_value(),
        'physics_weight_schedule': training_settings.physics_weight_schedule.get_value(),
        # -- inference -------------------------------------------------------
        'middle_trim': inference_settings.middle_trim.get_value(),
        'inference_batch_size': inference_settings.batch_size.get_value(),
        'experiment_number': inference_settings.experiment_number.get_value(),
        'pad_eval': inference_settings.pad_eval.get_value(),
        'window': inference_settings.window.get_value(),
        'patch_weighting': inference_settings.patch_weighting_method.get_value(),
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


def build_reconstructor(
    model_training_mode: str,
    data_settings: PtychoPINNDataSettings,
    model_settings: PtychoPINNModelSettings,
    inference_settings: PtychoPINNInferenceSettings,
    training_settings: PtychoPINNTrainingSettings,
) -> SubprocessReconstructor:
    """Build a :class:`SubprocessReconstructor` for one PtychoPINN mode.

    ``model_training_mode`` is 'Unsupervised' or 'Supervised' and becomes the
    reconstructor's display name plus the mode string the backend resolves.
    """
    # Source model path -> extracted bundle directory, so repeated reconstructs
    # against the same model unpack the archive only once.
    bundle_dir_cache: dict[Path, Path] = dict()

    def build_reconstruct_payload(
        parameters: ReconstructInput, loaded_model_path: Path | None, options: BaseModel
    ) -> ReconstructPayload:
        if loaded_model_path is None:
            raise RuntimeError('Cannot reconstruct: no PtychoPINN model has been loaded.')

        try:
            bundle_dir = bundle_dir_cache[loaded_model_path]
        except KeyError:
            bundle_dir = _extract_bundle_dir(loaded_model_path)
            bundle_dir_cache[loaded_model_path] = bundle_dir

        return ReconstructPayload(
            model_training_mode=model_training_mode,
            model_bundle_dir=bundle_dir,
            reconstruct_input=parameters,
            patch_weighting=inference_settings.patch_weighting_method.get_value(),
            inference_batch_size=inference_settings.batch_size.get_value(),
            accelerator=training_settings.accelerator.get_value(),
            num_workers=training_settings.num_dataloader_workers.get_value(),
        )

    def build_train_payload(
        input_path: Path, output_path: Path, options: BaseModel
    ) -> TrainPayload:
        return TrainPayload(
            settings=_build_settings(
                model_training_mode,
                data_settings,
                model_settings,
                training_settings,
                inference_settings,
            ),
            input_path=input_path,
            output_path=output_path,
            accelerator=training_settings.accelerator.get_value(),
            num_workers=training_settings.num_dataloader_workers.get_value(),
        )

    def export_training_data(
        file_path: Path, parameters: ReconstructInput, options: BaseModel
    ) -> None:
        save_ptychopinn_training_data(file_path, parameters)

    return SubprocessReconstructor(
        name=model_training_mode,
        reconstruct_entry_point=_RECONSTRUCT_ENTRY,
        build_reconstruct_payload=build_reconstruct_payload,
        is_trainable=True,
        train_entry_point=_TRAIN_ENTRY,
        build_train_payload=build_train_payload,
        model_file_filter='Zipped Archive (*.zip)',
        model_file_extension='.zip',
        training_data_file_filter='NumPy Zipped Archive (*.npz)',
        export_training_data=export_training_data,
        save_model=_save_model_bundle,
    )
