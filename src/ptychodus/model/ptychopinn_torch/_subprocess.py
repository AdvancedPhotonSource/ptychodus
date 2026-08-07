"""Child-side subprocess entry points for the PtychoPINN-Torch backend.

This module runs INSIDE a spawned subprocess. It is the only place in the
ptychodus tree that is allowed to import torch / lightning / ptycho_torch.
The parent-side ptychodus process never imports this module.

Two entry points are exposed:

- :func:`run_reconstruct` -- load a checkpoint, run one inference pass,
  stream back a single :class:`ReconstructOutput`.
- :func:`run_train` -- run one Lightning training session (which may itself
  fan out to ``n_devices`` DDP ranks via ``strategy='ddp_spawn'``), save the
  best checkpoint, and stream back the final :class:`TrainOutput` plus the
  saved-checkpoint path.

Both entry points rehydrate the ptychodus settings groups from the INI
string carried in the payload, so the child sees exactly the parameter
values the parent had at call time.
"""

from __future__ import annotations

import logging
import os
import pickle
from collections.abc import Sequence
from multiprocessing.queues import Queue
from pathlib import Path
from typing import Any

import numpy

from ptychodus.api.diffraction import zero_bad_pixels
from ptychodus.api.object import Object
from ptychodus.api.product import LossValue, Product
from ptychodus.api.reconstructor import ReconstructOutput, TrainOutput
from ptychodus.api.settings import SettingsRegistry

from ..processing._subprocess_protocol import load_settings_registry_from_string
from ..processing.subprocess_reconstructor import (
    TAG_MODEL_SAVED,
    TAG_OUTPUT,
    TAG_TRAIN_OUTPUT,
)
from ._payload import ReconstructPayload, TrainPayload
from .settings import (
    PtychoPINNTorchDataSettings,
    PtychoPINNTorchInferenceSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
)

logger = logging.getLogger(__name__)


def _rehydrate_settings(
    settings_ini: str,
) -> tuple[
    PtychoPINNTorchDataSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
    PtychoPINNTorchInferenceSettings,
]:
    registry = SettingsRegistry()
    data_s = PtychoPINNTorchDataSettings(registry)
    model_s = PtychoPINNTorchModelSettings(registry)
    training_s = PtychoPINNTorchTrainingSettings(registry)
    inference_s = PtychoPINNTorchInferenceSettings(registry)
    load_settings_registry_from_string(registry, settings_ini)
    return data_s, model_s, training_s, inference_s


def _create_config_manager(
    model_training_mode: str,
    data_s: PtychoPINNTorchDataSettings,
    model_s: PtychoPINNTorchModelSettings,
    training_s: PtychoPINNTorchTrainingSettings,
    inference_s: PtychoPINNTorchInferenceSettings,
    *,
    override_n_devices: int | None = None,
    override_strategy: str | None = None,
) -> Any:
    """Build the ptycho_torch ConfigManager from ptychodus settings.

    Runs inside the child (this function imports ptycho_torch types).
    """
    from ptycho_torch.api.base_api import ConfigManager
    from ptycho_torch.config_params import (
        DataConfig,
        DatagenConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )

    grid_size = (
        data_s.grid_size_y.get_value(),
        data_s.grid_size_x.get_value(),
    )
    x_bounds = (
        data_s.x_lower_bound.get_value(),
        data_s.x_upper_bound.get_value(),
    )
    y_bounds = (
        data_s.y_lower_bound.get_value(),
        data_s.y_upper_bound.get_value(),
    )
    data_config = DataConfig(
        N=data_s.model_size.get_value(),
        C=data_s.num_channels.get_value(),
        normalize=data_s.data_normalization_mode.get_value(),
        neighbor_function=data_s.neighbor_lookup_method.get_value(),
        scan_pattern=data_s.scan_pattern.get_value(),
        probe_normalize=data_s.normalize_probe.get_value(),
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        min_neighbor_distance=data_s.min_neighbor_distance.get_value(),
        max_neighbor_distance=data_s.max_neighbor_distance.get_value(),
        K_quadrant=data_s.num_nearest_neighbors_for_quadrant_lookup.get_value(),
        n_subsample=data_s.coordinate_subsampling_factor.get_value(),
        probe_scale=data_s.probe_scale.get_value(),
        K=data_s.num_nearest_neighbors_for_lookup.get_value(),
        grid_size=grid_size,
        probe_ramp_removal=data_s.probe_ramp_removal.get_value(),
        data_scaling=data_s.data_scaling_method.get_value(),
        phase_subtraction=data_s.subtract_mean_phase.get_value(),
    )

    amp_loss = model_s.auxiliary_amplitude_loss.get_value()
    phase_loss = model_s.auxiliary_phase_loss.get_value()

    model_config = ModelConfig(
        mode=model_training_mode,
        object_big=model_s.object_big.get_value(),
        probe_big=model_s.probe_big.get_value(),
        loss_function=model_s.loss_function.get_value(),
        amp_activation=model_s.amplitude_activation_function.get_value(),
        cbam_encoder=model_s.cbam_encoder.get_value(),
        decoder_last_amp_channels=data_s.num_channels.get_value(),
        use_shared_decoder=model_s.use_shared_decoder.get_value(),
        intensity_scale_trainable=model_s.intensity_scale_trainable.get_value(),
        intensity_scale=model_s.intensity_scale.get_value(),
        max_position_jitter=model_s.max_position_jitter.get_value(),
        num_datasets=model_s.num_datasets.get_value(),
        C_model=data_s.num_channels.get_value(),
        C_forward=data_s.num_channels.get_value(),
        amp_loss=None if amp_loss.casefold() == 'none' else amp_loss,
        phase_loss=None if phase_loss.casefold() == 'none' else phase_loss,
        amp_loss_coeff=model_s.auxiliary_amplitude_loss_coeff.get_value(),
        phase_loss_coeff=model_s.auxiliary_phase_loss_coeff.get_value(),
        n_filters_scale=model_s.num_filters_scale.get_value(),
        probe_mask=None,
        eca_decoder=model_s.eca_decoder.get_value(),
        batch_norm=model_s.use_batch_normalization.get_value(),
        edge_pad=model_s.edge_pad.get_value(),
        decoder_last_c_outer_fraction=model_s.decoder_last_c_outer_fraction.get_value(),
        cbam_bottleneck=model_s.cbam_bottleneck.get_value(),
        cbam_decoder=model_s.cbam_decoder.get_value(),
        spatial_decoder=model_s.spatial_decoder.get_value(),
        decoder_spatial_kernel=model_s.decoder_spatial_kernel.get_value(),
        eca_encoder=model_s.eca_encoder.get_value(),
        offset=model_s.offset.get_value(),
        probe_reference_coeff=model_s.probe_reference_loss_coeff.get_value(),
        amplitude_variance_loss=model_s.amplitude_variance_loss.get_value(),
        amplitude_variance_coeff=model_s.amplitude_variance_coeff.get_value(),
    )
    gradient_clip_val = training_s.gradient_clip_val.get_value()
    n_devices = (
        training_s.n_devices.get_value() if override_n_devices is None else override_n_devices
    )
    strategy = (
        training_s.distributed_strategy.get_value()
        if override_strategy is None
        else override_strategy
    )
    training_config = TrainingConfig(
        epochs=training_s.epochs.get_value(),
        batch_size=training_s.batch_size.get_value(),
        learning_rate=training_s.learning_rate.get_value(),
        n_devices=n_devices,
        num_workers=training_s.num_dataloader_workers.get_value(),
        accum_steps=training_s.gradient_accumulation_steps.get_value(),
        epochs_fine_tune=training_s.epochs_finetune.get_value(),
        fine_tune_gamma=training_s.finetune_gamma.get_value(),
        gradient_clip_val=gradient_clip_val if gradient_clip_val > 0.0 else None,
        nll=training_s.use_negative_log_likelihood_loss.get_value(),
        device=training_s.device.get_value(),
        strategy=strategy,
        framework='Lightning',
        orchestrator='Lightning',
        scheduler=training_s.learning_rate_scheduler.get_value(),
        warmup_epochs=training_s.learning_rate_warmup_epochs.get_value(),
        min_lr_ratio=training_s.minimum_learning_rate_ratio.get_value(),
        notes=training_s.notes.get_value(),
        model_name=training_s.model_name.get_value(),
        enable_staged_finetuning=training_s.enable_staged_finetuning.get_value(),
        finetune_stage1_epochs=training_s.finetune_stage1_epochs.get_value(),
        finetune_stage2_epochs=training_s.finetune_stage2_epochs.get_value(),
        finetune_stage3_epochs=training_s.finetune_stage3_epochs.get_value(),
        finetune_stage1_lr_decoder=training_s.finetune_stage1_lr_decoder.get_value(),
        finetune_stage2_lr_encoder_top=training_s.finetune_stage2_lr_encoder_top.get_value(),
        finetune_stage2_lr_decoder=training_s.finetune_stage2_lr_decoder.get_value(),
        finetune_stage2_lr_phase_head=training_s.finetune_stage2_lr_phase_head.get_value(),
        finetune_stage3_lr_encoder_bottom=training_s.finetune_stage3_lr_encoder_bottom.get_value(),
        finetune_stage3_lr_encoder_top=training_s.finetune_stage3_lr_encoder_top.get_value(),
        finetune_stage3_lr_decoder=training_s.finetune_stage3_lr_decoder.get_value(),
        finetune_stage3_lr_phase_head=training_s.finetune_stage3_lr_phase_head.get_value(),
        finetune_skip_stage3=training_s.finetune_skip_stage3.get_value(),
        finetune_early_stop_patience=training_s.finetune_early_stop_patience.get_value(),
        finetune_val_split=training_s.finetune_validation_split.get_value(),
    )
    inference_config = InferenceConfig(
        batch_size=inference_s.batch_size.get_value(),
        middle_trim=inference_s.middle_trim.get_value(),
        experiment_number=inference_s.experiment_number.get_value(),
        patch_weighting=inference_s.patch_weighting_method.get_value(),
        pad_eval=inference_s.pad_eval.get_value(),
        window=inference_s.window.get_value(),
    )
    datagen_config = DatagenConfig()

    return ConfigManager(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        inference_config=inference_config,
        datagen_config=datagen_config,
    )


def _load_ptycho_model(model_path: Path) -> tuple[Any, Any]:
    """Load a PtychoModel from a .ckpt file. Returns (model, config_manager)."""
    from ptycho_torch.api.base_api import ConfigManager, PtychoModel
    from ptycho_torch.model import PtychoPINN_Lightning

    data_config, model_config, training_config, inference_config = (
        PtychoModel._extract_configs_from_checkpoint(str(model_path))
    )
    if any(c is None for c in (data_config, model_config, training_config, inference_config)):
        raise ValueError(
            f'Checkpoint at {model_path} is missing one or more saved configs '
            f'(data/model/training/inference).'
        )

    ptycho_model = PtychoModel(
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )
    ptycho_model.model = PtychoPINN_Lightning.load_from_checkpoint(
        model_path,
        model_config=model_config,
        data_config=data_config,
        training_config=training_config,
        inference_config=inference_config,
    )

    config_manager = ConfigManager.from_loaded_model(ptycho_model)
    config_manager.validate_arch_compatibility(ptycho_model)
    return ptycho_model, config_manager


def run_reconstruct(payload: ReconstructPayload, queue: 'Queue[Any]') -> None:
    """Child entry point for one inference pass. Streams a single ReconstructOutput."""
    if payload.model_path is None:
        raise RuntimeError('Cannot reconstruct: no model checkpoint has been loaded.')

    from ptycho_torch.api.base_api import InferenceEngine, PtychoDataLoader

    ptycho_model, config_manager = _load_ptycho_model(payload.model_path)

    if config_manager.model_config.mode != payload.model_training_mode:
        logger.warning(
            'Loaded checkpoint mode %r does not match reconstructor mode %r; '
            'predictions may be inconsistent.',
            config_manager.model_config.mode,
            payload.model_training_mode,
        )

    inference_engine = InferenceEngine(config_manager=config_manager, ptycho_model=ptycho_model)

    parameters = payload.reconstruct_input
    object_geometry = parameters.product.object_.get_geometry()
    positions_px: list[float] = list()

    for position in parameters.product.probe_positions:
        object_point = object_geometry.map_coordinates_probe_to_object(position)
        positions_px.append(object_point.coordinate_y_px)
        positions_px.append(object_point.coordinate_x_px)

    diff_patterns = zero_bad_pixels(parameters.diffraction_patterns, parameters.bad_pixels)
    data_loader = PtychoDataLoader.from_np(
        diff_patterns=diff_patterns,
        probe=parameters.product.probes.get_probe_no_opr().get_array(),
        positions=numpy.reshape(positions_px, (-1, 2)),
        config_manager=config_manager,
    )
    object_out_array = numpy.asarray(inference_engine.predict_and_stitch(data_loader))

    object_in = parameters.product.object_
    object_out = Object(
        array=object_out_array,
        layer_spacing_m=object_in.layer_spacing_m,
        pixel_geometry=object_in.get_pixel_geometry(),
        center=object_in.get_center(),
    )

    losses: Sequence[LossValue] = []
    product = Product(
        metadata=parameters.product.metadata,
        probe_positions=parameters.product.probe_positions,
        probes=parameters.product.probes,
        object_=object_out,
        losses=losses,
    )

    queue.put((TAG_OUTPUT, pickle.dumps(ReconstructOutput(product=product, progress=1))))


def _apply_visible_devices(visible_gpu_indices: str) -> None:
    """Set CUDA_VISIBLE_DEVICES before torch is imported anywhere in this process."""
    trimmed = visible_gpu_indices.strip()
    if trimmed:
        os.environ['CUDA_VISIBLE_DEVICES'] = trimmed


def _clamp_n_devices(requested: int) -> int:
    """Return min(requested, torch.cuda.device_count()); warn on clamp."""
    import torch

    available = torch.cuda.device_count()
    if available == 0:
        logger.warning('No CUDA devices available; training will fall back to CPU.')
        return 1
    if requested > available:
        logger.warning(
            'Requested n_devices=%d but only %d CUDA device(s) available; clamping.',
            requested,
            available,
        )
        return available
    return requested


def run_train(payload: TrainPayload, queue: 'Queue[Any]') -> None:
    """Child entry point for one training session.

    Sets CUDA_VISIBLE_DEVICES before any torch import, clamps ``n_devices`` to
    what's actually visible, then runs the Lightning training loop with the
    configured DDP strategy.
    """
    _apply_visible_devices(payload.visible_gpu_indices)

    from lightning.pytorch.callbacks import Callback
    from ptycho_torch.api.base_api import (
        DataloaderFormats,
        PtychoDataLoader,
        PtychoModel,
        Trainer,
    )
    from ptycho_torch.lightning_utils import find_best_checkpoint
    from ptycho_torch.model import PtychoPINN_Lightning

    n_devices = _clamp_n_devices(payload.n_devices)

    data_s, model_s, training_s, inference_s = _rehydrate_settings(payload.settings_ini)
    config_manager = _create_config_manager(
        payload.model_training_mode,
        data_s,
        model_s,
        training_s,
        inference_s,
        override_n_devices=n_devices,
        override_strategy=payload.distributed_strategy,
    )

    class _LossCollectorCallback(Callback):
        """Collects per-epoch train and validation losses from the Lightning Trainer."""

        def __init__(self, train_metric_name: str, val_metric_name: str) -> None:
            super().__init__()
            self._train_metric_name = train_metric_name
            self._val_metric_name = val_metric_name
            self.training_loss: list[LossValue] = []
            self.validation_loss: list[LossValue] = []
            self.epochs_completed = 0

        def on_train_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
            value = trainer.callback_metrics.get(self._train_metric_name)
            if value is not None:
                self.training_loss.append(
                    LossValue(epoch=trainer.current_epoch, value=float(value))
                )
            self.epochs_completed = trainer.current_epoch + 1

        def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
            if trainer.sanity_checking:
                return
            value = trainer.callback_metrics.get(self._val_metric_name)
            if value is not None:
                self.validation_loss.append(
                    LossValue(epoch=trainer.current_epoch, value=float(value))
                )

    data_loader = PtychoDataLoader(
        data_dir=payload.input_path,
        config_manager=config_manager,
        data_format=DataloaderFormats('lightning_only_module'),
        output_dir=payload.output_path,
    )
    model = PtychoModel._new_model(model=PtychoPINN_Lightning, config_manager=config_manager)
    trainer = Trainer._from_lightning(
        model=model,
        dataloader=data_loader,
        orchestration='lightning',
        config_manager=config_manager,
    )

    loss_collector = _LossCollectorCallback(
        train_metric_name=model.model.loss_name,
        val_metric_name=model.model.val_loss_name,
    )
    trainer._trainer.callbacks.append(loss_collector)

    trainer.train(
        orchestration='lightning',
        experiment_name='',
    )
    # PtychoDataLoader appends a `run_<timestamp>` segment to output_dir,
    # and the checkpoint callback writes there — not at output_path.
    run_dir = Path(data_loader.output_dir)
    checkpoint_path = find_best_checkpoint(run_dir)

    if checkpoint_path is None:
        raise FileNotFoundError(f'No checkpoints found in {run_dir} after training.')

    queue.put(
        (
            TAG_TRAIN_OUTPUT,
            pickle.dumps(
                TrainOutput(
                    training_loss=loss_collector.training_loss,
                    validation_loss=loss_collector.validation_loss,
                    progress=loss_collector.epochs_completed,
                )
            ),
        )
    )
    queue.put((TAG_MODEL_SAVED, str(checkpoint_path)))
