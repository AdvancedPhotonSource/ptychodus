"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoFM.

Zero torch imports. All GPU work runs inside a spawned child; see
:mod:`._subprocess` for the child entry points.

PtychoFM's own ``config.yaml`` is a nested dict of scalars, so ``_build_config``
produces a plain :class:`dict` from the ptychodus settings groups -- pickleable
without pulling any ptycho_fm module in. The child feeds that dict to
``resolve_inference_model`` and to the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy

from ptychodus.api.preprocess.diffraction import zero_bad_pixels
from ptychodus.api.reconstruct import ReconstructInput

from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import ReconstructPayload, TrainPayload
from .enums import COUPLED_DECODER_MODES
from .settings import (
    PtychoFMDataSettings,
    PtychoFMInferenceSettings,
    PtychoFMModelSettings,
    PtychoFMTrainingSettings,
)

__all__ = [
    'build_reconstructor',
]


_RECONSTRUCT_ENTRY = 'ptychodus.model.ptycho_fm._subprocess:run_reconstruct'
_TRAIN_ENTRY = 'ptychodus.model.ptycho_fm._subprocess:run_train'

# Losses evaluated on the detector plane. The supervised objective applies the
# criterion to object amplitude and phase patches instead, where a probe
# envelope, a radial detector coordinate and a total photon flux are undefined.
_DIFFRACTION_SPACE_LOSSES = frozenset({'total_flux', 'probe_aware', 'q_dependent'})


def _build_config(
    name: str,
    data_settings: PtychoFMDataSettings,
    model_settings: PtychoFMModelSettings,
    training_settings: PtychoFMTrainingSettings,
    inference_settings: PtychoFMInferenceSettings,
) -> dict[str, Any]:
    """Translate ptychodus settings into a nested ``ptycho_fm`` config dict.

    Mirrors the top-level shape of ``ptycho_fm/config.yaml``: ``data``,
    ``model`` (with ``encoder`` / ``decoder`` / ``init`` sub-sections) and
    ``training`` (with ``weighted_loss``, ``probe_aware_loss`` and
    ``q_dependent_loss`` sub-sections). The child fills in any missing paths
    (``data_path`` / model save path) from the training payload at call time.

    ``coupled_decoder_mode`` selects the model class, so it is part of the
    architecture a checkpoint was trained under rather than a free parameter:
    a checkpoint loads only into the mode that produced it.

    Losses are translated, not validated here: this runs on the inference path
    too, where the training loss is irrelevant.

    The ``inference`` section is ptychodus-local: ``ptycho_fm``'s own YAML has
    no such section, and ``pad`` has no home in it at all. Only ``model`` is
    read back by ptycho_fm itself (via ``resolve_inference_model``).

    ``name`` selects the training objective: 'Supervised' compares the decoded
    amplitude and phase against the object patch, anything else compares the
    modelled diffraction against the measurement.
    """
    max_files = data_settings.max_files.get_value()
    data_config: dict[str, Any] = {
        'scale': data_settings.scale.get_value(),
        'default_normalization': data_settings.default_normalization.get_value(),
        'cache_object': data_settings.cache_object.get_value(),
        'max_probe_modes': data_settings.max_probe_modes.get_value(),
        'max_OPR_modes': data_settings.max_opr_modes.get_value(),
        'cache_memory_budget_mb': data_settings.cache_memory_budget_mb.get_value(),
        'train_split': data_settings.train_split.get_value(),
        'random_seed': data_settings.random_seed.get_value(),
        # 0 in settings means "no cap" (null in the YAML).
        'max_files': max_files if max_files > 0 else None,
        'num_workers': data_settings.num_workers.get_value(),
        'prefetch_factor': data_settings.prefetch_factor.get_value(),
    }

    coupled_decoder_mode = model_settings.coupled_decoder_mode.get_value()

    if coupled_decoder_mode not in COUPLED_DECODER_MODES:
        raise ValueError(
            f'Unknown coupled_decoder_mode: {coupled_decoder_mode!r}. '
            f'Expected one of {sorted(COUPLED_DECODER_MODES)}.'
        )

    model_config: dict[str, Any] = {
        'encoder_type': model_settings.encoder_type.get_value(),
        # Emitted even when uncoupled: ptycho_fm reads absent and None alike,
        # and a stable dict shape is what the config-builder test pins.
        'coupled_decoder_mode': None if coupled_decoder_mode == 'none' else coupled_decoder_mode,
        'encoder': {
            'img_size': model_settings.img_size.get_value(),
            'patch_size': model_settings.patch_size.get_value(),
            'in_channels': 1,
            'embed_dim': model_settings.embed_dim.get_value(),
            'depth': model_settings.depth.get_value(),
            'num_heads': model_settings.num_heads.get_value(),
            'mlp_ratio': model_settings.mlp_ratio.get_value(),
            'use_cls_token': model_settings.use_cls_token.get_value(),
            'dropout': model_settings.dropout.get_value(),
            'attn_dropout': model_settings.attn_dropout.get_value(),
            'timm_model_name': model_settings.timm_model_name.get_value(),
        },
        'decoder': {
            'base_channels': model_settings.decoder_base_channels.get_value(),
            'latent_dim': model_settings.decoder_latent_dim.get_value(),
            'num_stages': model_settings.decoder_num_stages.get_value(),
            'use_batchnorm': model_settings.decoder_use_batchnorm.get_value(),
            'dropout': model_settings.decoder_dropout.get_value(),
        },
        'init': {
            'enabled': model_settings.init_enabled.get_value(),
            'method': model_settings.init_method.get_value(),
        },
    }

    training_config: dict[str, Any] = {
        'mode': 'supervised' if name == 'Supervised' else 'unsupervised',
        'batch_size': training_settings.batch_size.get_value(),
        'learning_rate': training_settings.learning_rate.get_value(),
        'epochs': training_settings.epochs.get_value(),
        'loss_function': training_settings.loss_function.get_value(),
        'weighted_loss': {
            'loss_type': training_settings.weighted_loss_type.get_value(),
            'threshold': training_settings.weighted_loss_threshold.get_value(),
            'alpha': training_settings.weighted_loss_alpha.get_value(),
        },
        'probe_aware_loss': {
            'loss_type': training_settings.probe_aware_loss_type.get_value(),
            'threshold': training_settings.probe_aware_loss_threshold.get_value(),
            'alpha': training_settings.probe_aware_loss_alpha.get_value(),
            'envelope_floor': training_settings.probe_aware_loss_envelope_floor.get_value(),
        },
        'q_dependent_loss': {
            'loss_type': training_settings.q_dependent_loss_type.get_value(),
            'alpha': training_settings.q_dependent_loss_alpha.get_value(),
            'q_beta': training_settings.q_dependent_loss_q_beta.get_value(),
            'q_floor': training_settings.q_dependent_loss_q_floor.get_value(),
            'threshold': training_settings.q_dependent_loss_threshold.get_value(),
            'log_error': training_settings.q_dependent_loss_log_error.get_value(),
            'normalize_q': training_settings.q_dependent_loss_normalize_q.get_value(),
        },
        'save_epoch_models': training_settings.save_epoch_models.get_value(),
    }

    inference_config: dict[str, Any] = {
        'central_crop': inference_settings.central_crop.get_value(),
        'pad': inference_settings.pad.get_value(),
        'batch_size': inference_settings.batch_size.get_value(),
    }

    return {
        'data': data_config,
        'model': model_config,
        'training': training_config,
        'inference': inference_config,
    }


def _verify_spatial_size(shape: tuple[int, ...], img_size: int, what: str) -> None:
    """Raise unless a trailing ``(height, width)`` is exactly ``img_size`` square."""
    if tuple(shape) != (img_size, img_size):
        height, width = shape
        raise ValueError(
            f'{what} are {height}x{width}, but PtychoFM requires {img_size}x{img_size} '
            'to match the encoder input. Adjust the pattern crop or the model image size.'
        )


def _verify_loss_supports_mode(name: str, loss_function: str) -> None:
    """Raise unless ``loss_function`` is meaningful for the training objective.

    The diffraction-space losses do not fail on object patches, they quietly
    compute something else: a log of a signed phase clamps half the patch to the
    stability floor, and a probe envelope the caller never supplies degrades to
    a plain masked error.
    """
    if name == 'Supervised' and loss_function in _DIFFRACTION_SPACE_LOSSES:
        raise ValueError(
            f'The {loss_function!r} loss compares diffraction amplitudes, but supervised '
            'training compares the decoded object against the ground truth patch. Select '
            'an object-space loss, or train unsupervised.'
        )


def build_reconstructor(
    name: str,
    data_settings: PtychoFMDataSettings,
    model_settings: PtychoFMModelSettings,
    inference_settings: PtychoFMInferenceSettings,
    training_settings: PtychoFMTrainingSettings,
) -> SubprocessReconstructor:
    """Build a :class:`SubprocessReconstructor` for one PtychoFM mode.

    ``name`` is 'Unsupervised' or 'Supervised' and becomes the reconstructor's
    display name.
    """

    def build_reconstruct_payload(
        parameters: ReconstructInput, loaded_model_path: Path | None
    ) -> ReconstructPayload:
        return ReconstructPayload(
            name=name,
            config=_build_config(
                name, data_settings, model_settings, training_settings, inference_settings
            ),
            model_path=loaded_model_path,
            reconstruct_input=parameters,
        )

    def build_train_payload(input_path: Path, output_path: Path) -> TrainPayload:
        _verify_loss_supports_mode(name, training_settings.loss_function.get_value())
        return TrainPayload(
            name=name,
            config=_build_config(
                name, data_settings, model_settings, training_settings, inference_settings
            ),
            input_path=input_path,
            output_path=output_path,
        )

    def export_training_data(file_path: Path, parameters: ReconstructInput) -> None:
        # ptycho_fm.CombinedDataset expects one directory per scan holding
        # two HDF5 files sharing a common stem:
        #     <stem>_dp.hdf5    dataset 'dp'    (N, H, W) float
        #     <stem>_para.hdf5  dataset 'object'              (1, H, W) complex
        #                                       'probe'                (1, N_modes, H, W) complex
        #                                       'probe_position_x_m'   (N,) float64
        #                                       'probe_position_y_m'   (N,) float64
        #                       'object' carries a 'pixel_height_m' attr; the
        #                       loader auto-detects meters vs pixels from range.
        # Patterns and probes are stored at their native spatial size and the
        # object patch is extracted at the pattern size, so both must already
        # match the encoder input; nothing downstream resizes them.
        # file_path is treated as a stem, so the picked filename itself is
        # never created; the two derived files land next to it.
        stem = file_path.stem
        dp_path = file_path.with_name(f'{stem}_dp.hdf5')
        para_path = file_path.with_name(f'{stem}_para.hdf5')

        dp = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)
        obj = parameters.product.object_
        object_layer = obj.get_layer(0)
        pixel_geometry = obj.get_pixel_geometry()
        probe_array = parameters.product.probes.get_probe_no_opr().get_array()

        img_size = model_settings.img_size.get_value()
        _verify_spatial_size(dp.shape[-2:], img_size, 'Diffraction patterns')
        _verify_spatial_size(probe_array.shape[-2:], img_size, 'Probe')

        pos_x_m: list[float] = []
        pos_y_m: list[float] = []
        for point in parameters.product.probe_positions:
            pos_x_m.append(point.x_m)
            pos_y_m.append(point.y_m)

        with h5py.File(dp_path, 'w') as h5_dp:
            h5_dp.create_dataset('dp', data=dp)

        with h5py.File(para_path, 'w') as h5_para:
            obj_ds = h5_para.create_dataset('object', data=object_layer[numpy.newaxis, :, :])
            obj_ds.attrs['pixel_height_m'] = pixel_geometry.height_m
            obj_ds.attrs['pixel_width_m'] = pixel_geometry.width_m
            h5_para.create_dataset('probe', data=probe_array[numpy.newaxis, :, :, :])
            h5_para.create_dataset(
                'probe_position_x_m', data=numpy.asarray(pos_x_m, dtype=numpy.float64)
            )
            h5_para.create_dataset(
                'probe_position_y_m', data=numpy.asarray(pos_y_m, dtype=numpy.float64)
            )

    return SubprocessReconstructor(
        name=name,
        reconstruct_entry_point=_RECONSTRUCT_ENTRY,
        progress_goal_fn=lambda: training_settings.epochs.get_value(),
        build_reconstruct_payload=build_reconstruct_payload,
        is_trainable=True,
        train_entry_point=_TRAIN_ENTRY,
        build_train_payload=build_train_payload,
        model_file_filter='PyTorch Checkpoint (*.pth *.pt)',
        model_file_extension='.pth',
        training_data_file_filter='PtychoFM Training Pair (*.hdf5)',
        export_training_data=export_training_data,
    )
