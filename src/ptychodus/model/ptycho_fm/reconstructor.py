"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoFM.

Zero torch imports. All GPU work runs inside a spawned child; see
:mod:`._subprocess` for the child entry points.

PtychoFM's own ``config.yaml`` is a nested dict of scalars, so ``_build_config``
produces a plain :class:`dict` from the ptychodus settings groups -- pickleable
without pulling any ptycho_vit module in. The child feeds that dict directly
into ``PtychoViT(config)`` and the training loop.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ptychodus.api.io import save_ptychopinn_training_data
from ptychodus.api.reconstruct import ReconstructInput

from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import ReconstructPayload, TrainPayload
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


def _build_config(
    data_settings: PtychoFMDataSettings,
    model_settings: PtychoFMModelSettings,
    training_settings: PtychoFMTrainingSettings,
    inference_settings: PtychoFMInferenceSettings,
) -> dict[str, Any]:
    """Translate ptychodus settings into a nested ``ptycho_vit`` config dict.

    Mirrors the top-level shape of ``ptycho_vit/config.yaml``: ``data``,
    ``model`` (with ``encoder`` / ``decoder`` / ``init`` sub-sections),
    ``training`` (with ``weighted_loss`` sub-section), and ``inference``. The
    child fills in any missing paths (``data_path`` / model save path) from
    the training payload at call time.
    """
    max_files = data_settings.max_files.get_value()
    data_config: dict[str, Any] = {
        'scale': data_settings.scale.get_value(),
        'default_normalization': data_settings.default_normalization.get_value(),
        'packed': data_settings.packed.get_value(),
        'cache_object': data_settings.cache_object.get_value(),
        'max_probe_modes': data_settings.max_probe_modes.get_value(),
        'target_size': data_settings.target_size.get_value(),
        'train_split': data_settings.train_split.get_value(),
        'random_seed': data_settings.random_seed.get_value(),
        'sharding_strategy': data_settings.sharding_strategy.get_value(),
        # 0 in settings means "no cap" (null in the YAML).
        'max_files': max_files if max_files > 0 else None,
        'num_workers': data_settings.num_workers.get_value(),
        'prefetch_factor': data_settings.prefetch_factor.get_value(),
        'use_cuda_prefetcher': data_settings.use_cuda_prefetcher.get_value(),
    }

    model_config: dict[str, Any] = {
        'encoder_type': model_settings.encoder_type.get_value(),
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
        'mode': training_settings.mode.get_value(),
        'batch_size': training_settings.batch_size.get_value(),
        'learning_rate': training_settings.learning_rate.get_value(),
        'epochs': training_settings.epochs.get_value(),
        'loss_function': training_settings.loss_function.get_value(),
        'weighted_loss': {
            'loss_type': training_settings.weighted_loss_type.get_value(),
            'threshold': training_settings.weighted_loss_threshold.get_value(),
            'alpha': training_settings.weighted_loss_alpha.get_value(),
        },
        'validation_plot_freq': training_settings.validation_plot_freq.get_value(),
        'checkpoint_freq': training_settings.checkpoint_freq.get_value(),
        'save_epoch_models': training_settings.save_epoch_models.get_value(),
        'resume_from_checkpoint': training_settings.resume_from_checkpoint.get_value(),
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
                data_settings, model_settings, training_settings, inference_settings
            ),
            model_path=loaded_model_path,
            reconstruct_input=parameters,
        )

    def build_train_payload(input_path: Path, output_path: Path) -> TrainPayload:
        return TrainPayload(
            name=name,
            config=_build_config(
                data_settings, model_settings, training_settings, inference_settings
            ),
            input_path=input_path,
            output_path=output_path,
        )

    def export_training_data(file_path: Path, parameters: ReconstructInput) -> None:
        # PtychoFM's PtychographyDataset expects the ptycho-torch NPZ layout
        # (multi-mode probeGuess as (N_modes, H, W)); reuse the shared helper.
        save_ptychopinn_training_data(file_path, parameters, multimodal_probe=True)

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
        training_data_file_filter='NumPy Zipped Archive (*.npz)',
        export_training_data=export_training_data,
    )
