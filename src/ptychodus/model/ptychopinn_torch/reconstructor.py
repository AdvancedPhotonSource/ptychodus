"""Parent-side factory that builds a :class:`SubprocessReconstructor` for PtychoPINN-Torch.

Zero lightning imports, and no GPU context is ever acquired here. All GPU work
runs inside a spawned child; see :mod:`._subprocess` for the child entry
points.

:func:`_build_configs` does import ``ptycho_torch.config_params`` (and
transitively torch) so the parent can hand the child finished config objects
instead of an INI blob. It is called only from ``build_train_payload``, so that
import happens on the first training run -- never at composition-root time, and
never at all for inference. See :mod:`._payload` for why this is allowed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ptychodus.api.io import save_ptychopinn_training_data
from ptychodus.api.reconstruct import ReconstructInput

from ..processing.subprocess_reconstructor import SubprocessReconstructor
from ._payload import ReconstructPayload, TrainPayload
from .settings import (
    PtychoPINNTorchDataSettings,
    PtychoPINNTorchInferenceSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
)

__all__ = [
    'build_reconstructor',
]


_RECONSTRUCT_ENTRY = 'ptychodus.model.ptychopinn_torch._subprocess:run_reconstruct'
_TRAIN_ENTRY = 'ptychodus.model.ptychopinn_torch._subprocess:run_train'


def _build_configs(
    model_training_mode: str,
    data_settings: PtychoPINNTorchDataSettings,
    model_settings: PtychoPINNTorchModelSettings,
    training_settings: PtychoPINNTorchTrainingSettings,
    inference_settings: PtychoPINNTorchInferenceSettings,
) -> tuple[Any, Any, Any, Any, Any]:
    """Translate ptychodus settings into the five ptycho_torch config objects.

    Returns ``(data, model, training, inference, datagen)``. The child
    assembles them into a ``ConfigManager``; that step needs the GPU-side
    ``ptycho_torch.api`` package and stays child-side.
    """
    from ptycho_torch.config_params import (
        DataConfig,
        DatagenConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )

    grid_size = (
        data_settings.grid_size_y.get_value(),
        data_settings.grid_size_x.get_value(),
    )
    x_bounds = (
        data_settings.x_lower_bound.get_value(),
        data_settings.x_upper_bound.get_value(),
    )
    y_bounds = (
        data_settings.y_lower_bound.get_value(),
        data_settings.y_upper_bound.get_value(),
    )
    data_config = DataConfig(
        N=data_settings.model_size.get_value(),
        C=data_settings.num_channels.get_value(),
        normalize=data_settings.data_normalization_mode.get_value(),
        neighbor_function=data_settings.neighbor_lookup_method.get_value(),
        scan_pattern=data_settings.scan_pattern.get_value(),
        probe_normalize=data_settings.normalize_probe.get_value(),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        min_neighbor_distance=data_settings.min_neighbor_distance.get_value(),
        max_neighbor_distance=data_settings.max_neighbor_distance.get_value(),
        K_quadrant=data_settings.num_nearest_neighbors_for_quadrant_lookup.get_value(),
        n_subsample=data_settings.coordinate_subsampling_factor.get_value(),
        probe_scale=data_settings.probe_scale.get_value(),
        K=data_settings.num_nearest_neighbors_for_lookup.get_value(),
        grid_size=grid_size,
        probe_ramp_removal=data_settings.probe_ramp_removal.get_value(),
        data_scaling=data_settings.data_scaling_method.get_value(),
        phase_subtraction=data_settings.subtract_mean_phase.get_value(),
    )

    amp_loss = model_settings.auxiliary_amplitude_loss.get_value()
    phase_loss = model_settings.auxiliary_phase_loss.get_value()

    model_config = ModelConfig(
        mode=model_training_mode,
        object_big=model_settings.object_big.get_value(),
        probe_big=model_settings.probe_big.get_value(),
        loss_function=model_settings.loss_function.get_value(),
        amp_activation=model_settings.amplitude_activation_function.get_value(),
        cbam_encoder=model_settings.cbam_encoder.get_value(),
        decoder_last_amp_channels=data_settings.num_channels.get_value(),
        use_shared_decoder=model_settings.use_shared_decoder.get_value(),
        intensity_scale_trainable=model_settings.intensity_scale_trainable.get_value(),
        intensity_scale=model_settings.intensity_scale.get_value(),
        max_position_jitter=model_settings.max_position_jitter.get_value(),
        num_datasets=model_settings.num_datasets.get_value(),
        C_model=data_settings.num_channels.get_value(),
        C_forward=data_settings.num_channels.get_value(),
        amp_loss=None if amp_loss.casefold() == 'none' else amp_loss,
        phase_loss=None if phase_loss.casefold() == 'none' else phase_loss,
        amp_loss_coeff=model_settings.auxiliary_amplitude_loss_coeff.get_value(),
        phase_loss_coeff=model_settings.auxiliary_phase_loss_coeff.get_value(),
        n_filters_scale=model_settings.num_filters_scale.get_value(),
        probe_mask=None,
        eca_decoder=model_settings.eca_decoder.get_value(),
        batch_norm=model_settings.use_batch_normalization.get_value(),
        edge_pad=model_settings.edge_pad.get_value(),
        decoder_last_c_outer_fraction=model_settings.decoder_last_c_outer_fraction.get_value(),
        cbam_bottleneck=model_settings.cbam_bottleneck.get_value(),
        cbam_decoder=model_settings.cbam_decoder.get_value(),
        spatial_decoder=model_settings.spatial_decoder.get_value(),
        decoder_spatial_kernel=model_settings.decoder_spatial_kernel.get_value(),
        eca_encoder=model_settings.eca_encoder.get_value(),
        offset=model_settings.offset.get_value(),
        probe_reference_coeff=model_settings.probe_reference_loss_coeff.get_value(),
        amplitude_variance_loss=model_settings.amplitude_variance_loss.get_value(),
        amplitude_variance_coeff=model_settings.amplitude_variance_coeff.get_value(),
    )

    gradient_clip_val = training_settings.gradient_clip_val.get_value()
    training_config = TrainingConfig(
        epochs=training_settings.epochs.get_value(),
        batch_size=training_settings.batch_size.get_value(),
        learning_rate=training_settings.learning_rate.get_value(),
        n_devices=training_settings.n_devices.get_value(),
        num_workers=training_settings.num_dataloader_workers.get_value(),
        accum_steps=training_settings.gradient_accumulation_steps.get_value(),
        epochs_fine_tune=training_settings.epochs_finetune.get_value(),
        fine_tune_gamma=training_settings.finetune_gamma.get_value(),
        gradient_clip_val=gradient_clip_val if gradient_clip_val > 0.0 else None,
        nll=training_settings.use_negative_log_likelihood_loss.get_value(),
        device=training_settings.device.get_value(),
        strategy=training_settings.distributed_strategy.get_value(),
        framework='Lightning',
        orchestrator='Lightning',
        scheduler=training_settings.learning_rate_scheduler.get_value(),
        warmup_epochs=training_settings.learning_rate_warmup_epochs.get_value(),
        min_lr_ratio=training_settings.minimum_learning_rate_ratio.get_value(),
        notes=training_settings.notes.get_value(),
        model_name=training_settings.model_name.get_value(),
        enable_staged_finetuning=training_settings.enable_staged_finetuning.get_value(),
        finetune_stage1_epochs=training_settings.finetune_stage1_epochs.get_value(),
        finetune_stage2_epochs=training_settings.finetune_stage2_epochs.get_value(),
        finetune_stage3_epochs=training_settings.finetune_stage3_epochs.get_value(),
        finetune_stage1_lr_decoder=training_settings.finetune_stage1_lr_decoder.get_value(),
        finetune_stage2_lr_encoder_top=training_settings.finetune_stage2_lr_encoder_top.get_value(),
        finetune_stage2_lr_decoder=training_settings.finetune_stage2_lr_decoder.get_value(),
        finetune_stage2_lr_phase_head=training_settings.finetune_stage2_lr_phase_head.get_value(),
        finetune_stage3_lr_encoder_bottom=(
            training_settings.finetune_stage3_lr_encoder_bottom.get_value()
        ),
        finetune_stage3_lr_encoder_top=training_settings.finetune_stage3_lr_encoder_top.get_value(),
        finetune_stage3_lr_decoder=training_settings.finetune_stage3_lr_decoder.get_value(),
        finetune_stage3_lr_phase_head=training_settings.finetune_stage3_lr_phase_head.get_value(),
        finetune_skip_stage3=training_settings.finetune_skip_stage3.get_value(),
        finetune_early_stop_patience=training_settings.finetune_early_stop_patience.get_value(),
        finetune_val_split=training_settings.finetune_validation_split.get_value(),
    )

    inference_config = InferenceConfig(
        batch_size=inference_settings.batch_size.get_value(),
        middle_trim=inference_settings.middle_trim.get_value(),
        experiment_number=inference_settings.experiment_number.get_value(),
        patch_weighting=inference_settings.patch_weighting_method.get_value(),
        pad_eval=inference_settings.pad_eval.get_value(),
        window=inference_settings.window.get_value(),
    )

    return data_config, model_config, training_config, inference_config, DatagenConfig()


def build_reconstructor(
    model_training_mode: str,
    data_settings: PtychoPINNTorchDataSettings,
    model_settings: PtychoPINNTorchModelSettings,
    inference_settings: PtychoPINNTorchInferenceSettings,
    training_settings: PtychoPINNTorchTrainingSettings,
) -> SubprocessReconstructor:
    """Build a :class:`SubprocessReconstructor` for one PtychoPINN-Torch mode.

    ``model_training_mode`` is 'Unsupervised' or 'Supervised' and becomes the
    reconstructor's display name plus the config-manager mode string used
    inside the child.
    """

    def build_reconstruct_payload(
        parameters: ReconstructInput, loaded_model_path: Path | None
    ) -> ReconstructPayload:
        return ReconstructPayload(
            model_training_mode=model_training_mode,
            model_path=loaded_model_path,
            reconstruct_input=parameters,
        )

    def build_train_payload(input_path: Path, output_path: Path) -> TrainPayload:
        data_config, model_config, training_config, inference_config, datagen_config = (
            _build_configs(
                model_training_mode,
                data_settings,
                model_settings,
                training_settings,
                inference_settings,
            )
        )
        return TrainPayload(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            inference_config=inference_config,
            datagen_config=datagen_config,
            input_path=input_path,
            output_path=output_path,
            visible_gpu_indices=training_settings.visible_gpu_indices.get_value(),
        )

    def export_training_data(file_path: Path, parameters: ReconstructInput) -> None:
        save_ptychopinn_training_data(file_path, parameters, multimodal_probe=True)

    return SubprocessReconstructor(
        name=model_training_mode,
        reconstruct_entry_point=_RECONSTRUCT_ENTRY,
        progress_goal_fn=lambda: training_settings.epochs.get_value(),
        build_reconstruct_payload=build_reconstruct_payload,
        is_trainable=True,
        train_entry_point=_TRAIN_ENTRY,
        build_train_payload=build_train_payload,
        model_file_filter='PyTorch Lightning Checkpoint Files (*.ckpt)',
        model_file_extension='.ckpt',
        training_data_file_filter='NumPy Zipped Archive (*.npz)',
        export_training_data=export_training_data,
    )
