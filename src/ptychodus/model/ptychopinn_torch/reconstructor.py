from __future__ import annotations
from collections.abc import Iterator, Sequence
from pathlib import Path
import logging
import shutil

import numpy

from lightning.pytorch.callbacks import Callback

from ptycho_torch.api.base_api import (
    ConfigManager,
    DataloaderFormats,
    InferenceEngine,
    PtychoDataLoader,
    PtychoModel,
    Trainer,
)
from ptycho_torch.config_params import (
    DataConfig,
    DatagenConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.lightning_utils import find_best_checkpoint
from ptycho_torch.model import PtychoPINN_Lightning

from ptychodus.api.diffraction import zero_bad_pixels
from ptychodus.api.io import save_ptychopinn_training_data
from ptychodus.api.object import Object
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import (
    LossValue,
    ReconstructInput,
    ReconstructOutput,
    TrainOutput,
    TrainableReconstructor,
)

from .settings import (
    PtychoPINNTorchDataSettings,
    PtychoPINNTorchInferenceSettings,
    PtychoPINNTorchModelSettings,
    PtychoPINNTorchTrainingSettings,
)

__all__ = [
    'PtychoPINNTorchTrainableReconstructor',
]

logger = logging.getLogger(__name__)


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
            self.training_loss.append(LossValue(epoch=trainer.current_epoch, value=float(value)))
        self.epochs_completed = trainer.current_epoch + 1

    def on_validation_epoch_end(self, trainer, pl_module) -> None:  # noqa: ANN001
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self._val_metric_name)
        if value is not None:
            self.validation_loss.append(LossValue(epoch=trainer.current_epoch, value=float(value)))


class PtychoPINNTorchTrainableReconstructor(TrainableReconstructor):
    def __init__(
        self,
        model_training_mode: str,
        data_settings: PtychoPINNTorchDataSettings,
        model_settings: PtychoPINNTorchModelSettings,
        inference_settings: PtychoPINNTorchInferenceSettings,
        training_settings: PtychoPINNTorchTrainingSettings,
        *,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__()
        self._model_training_mode = model_training_mode
        self._data_settings = data_settings
        self._model_settings = model_settings
        self._inference_settings = inference_settings
        self._training_settings = training_settings
        self._is_developer_mode_enabled = is_developer_mode_enabled

        self._inference_engine: InferenceEngine | None = None
        self._inference_config_manager: ConfigManager | None = None
        self._loaded_from: Path | None = None

    def _create_config_from_settings(self) -> ConfigManager:
        grid_size = (
            self._data_settings.grid_size_y.get_value(),
            self._data_settings.grid_size_x.get_value(),
        )
        x_bounds = (
            self._data_settings.x_lower_bound.get_value(),
            self._data_settings.x_upper_bound.get_value(),
        )
        y_bounds = (
            self._data_settings.y_lower_bound.get_value(),
            self._data_settings.y_upper_bound.get_value(),
        )
        data_config = DataConfig(
            N=self._data_settings.model_size.get_value(),
            C=self._data_settings.num_channels.get_value(),
            normalize=self._data_settings.data_normalization_mode.get_value(),
            neighbor_function=self._data_settings.neighbor_lookup_method.get_value(),
            scan_pattern=self._data_settings.scan_pattern.get_value(),
            probe_normalize=self._data_settings.normalize_probe.get_value(),
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            min_neighbor_distance=self._data_settings.min_neighbor_distance.get_value(),
            max_neighbor_distance=self._data_settings.max_neighbor_distance.get_value(),
            K_quadrant=self._data_settings.num_nearest_neighbors_for_quadrant_lookup.get_value(),
            nphotons=self._data_settings.num_photons.get_value(),  # TODO get from product
            n_subsample=self._data_settings.coordinate_subsampling_factor.get_value(),
            probe_scale=self._data_settings.probe_scale.get_value(),
            K=self._data_settings.num_nearest_neighbors_for_lookup.get_value(),
            grid_size=grid_size,
            probe_ramp_removal=self._data_settings.probe_ramp_removal.get_value(),
            data_scaling=self._data_settings.data_scaling_method.get_value(),
            phase_subtraction=self._data_settings.subtract_mean_phase.get_value(),
        )

        amp_loss = self._model_settings.auxiliary_amplitude_loss.get_value()
        phase_loss = self._model_settings.auxiliary_phase_loss.get_value()

        model_config = ModelConfig(
            mode=self._model_training_mode,
            object_big=self._model_settings.object_big.get_value(),
            probe_big=self._model_settings.probe_big.get_value(),
            loss_function=self._model_settings.loss_function.get_value(),
            amp_activation=self._model_settings.amplitude_activation_function.get_value(),
            cbam_encoder=self._model_settings.cbam_encoder.get_value(),
            decoder_last_amp_channels=self._data_settings.num_channels.get_value(),
            use_shared_decoder=self._model_settings.use_shared_decoder.get_value(),
            intensity_scale_trainable=self._model_settings.intensity_scale_trainable.get_value(),
            intensity_scale=self._model_settings.intensity_scale.get_value(),
            max_position_jitter=self._model_settings.max_position_jitter.get_value(),
            num_datasets=self._model_settings.num_datasets.get_value(),
            C_model=self._data_settings.num_channels.get_value(),
            C_forward=self._data_settings.num_channels.get_value(),
            amp_loss=None if amp_loss.casefold() == 'none' else amp_loss,
            phase_loss=None if phase_loss.casefold() == 'none' else phase_loss,
            amp_loss_coeff=self._model_settings.auxiliary_amplitude_loss_coeff.get_value(),
            phase_loss_coeff=self._model_settings.auxiliary_phase_loss_coeff.get_value(),
            n_filters_scale=self._model_settings.num_filters_scale.get_value(),
            probe_mask=None,
            eca_decoder=self._model_settings.eca_decoder.get_value(),
            batch_norm=self._model_settings.use_batch_normalization.get_value(),
            edge_pad=self._model_settings.edge_pad.get_value(),
            decoder_last_c_outer_fraction=self._model_settings.decoder_last_c_outer_fraction.get_value(),
            cbam_bottleneck=self._model_settings.cbam_bottleneck.get_value(),
            cbam_decoder=self._model_settings.cbam_decoder.get_value(),
            spatial_decoder=self._model_settings.spatial_decoder.get_value(),
            decoder_spatial_kernel=self._model_settings.decoder_spatial_kernel.get_value(),
            eca_encoder=self._model_settings.eca_encoder.get_value(),
            offset=self._model_settings.offset.get_value(),
            probe_reference_coeff=self._model_settings.probe_reference_loss_coeff.get_value(),
            amplitude_variance_loss=self._model_settings.amplitude_variance_loss.get_value(),
            amplitude_variance_coeff=self._model_settings.amplitude_variance_coeff.get_value(),
        )
        gradient_clip_val = self._training_settings.gradient_clip_val.get_value()
        training_config = TrainingConfig(
            epochs=self._training_settings.epochs.get_value(),
            batch_size=self._training_settings.batch_size.get_value(),
            learning_rate=self._training_settings.learning_rate.get_value(),
            n_devices=self._training_settings.num_devices.get_value(),
            num_workers=self._training_settings.num_dataloader_workers.get_value(),
            accum_steps=self._training_settings.gradient_accumulation_steps.get_value(),
            epochs_fine_tune=self._training_settings.epochs_finetune.get_value(),
            fine_tune_gamma=self._training_settings.finetune_gamma.get_value(),
            gradient_clip_val=gradient_clip_val if gradient_clip_val > 0.0 else None,
            experiment_name=self._training_settings.experiment_name.get_value(),
            nll=self._training_settings.use_negative_log_likelihood_loss.get_value(),
            device=self._training_settings.device.get_value(),
            strategy='ddp_spawn',
            framework='Lightning',
            orchestrator='Lightning',
            scheduler=self._training_settings.learning_rate_scheduler.get_value(),
            warmup_epochs=self._training_settings.learning_rate_warmup_epochs.get_value(),
            min_lr_ratio=self._training_settings.minimum_learning_rate_ratio.get_value(),
            notes=self._training_settings.notes.get_value(),
            model_name=self._training_settings.model_name.get_value(),
            enable_staged_finetuning=self._training_settings.enable_staged_finetuning.get_value(),
            finetune_stage1_epochs=self._training_settings.finetune_stage1_epochs.get_value(),
            finetune_stage2_epochs=self._training_settings.finetune_stage2_epochs.get_value(),
            finetune_stage3_epochs=self._training_settings.finetune_stage3_epochs.get_value(),
            finetune_stage1_lr_decoder=self._training_settings.finetune_stage1_lr_decoder.get_value(),
            finetune_stage2_lr_encoder_top=self._training_settings.finetune_stage2_lr_encoder_top.get_value(),
            finetune_stage2_lr_decoder=self._training_settings.finetune_stage2_lr_decoder.get_value(),
            finetune_stage2_lr_phase_head=self._training_settings.finetune_stage2_lr_phase_head.get_value(),
            finetune_stage3_lr_encoder_bottom=self._training_settings.finetune_stage3_lr_encoder_bottom.get_value(),
            finetune_stage3_lr_encoder_top=self._training_settings.finetune_stage3_lr_encoder_top.get_value(),
            finetune_stage3_lr_decoder=self._training_settings.finetune_stage3_lr_decoder.get_value(),
            finetune_stage3_lr_phase_head=self._training_settings.finetune_stage3_lr_phase_head.get_value(),
            finetune_skip_stage3=self._training_settings.finetune_skip_stage3.get_value(),
            finetune_early_stop_patience=self._training_settings.finetune_early_stop_patience.get_value(),
            finetune_val_split=self._training_settings.finetune_validation_split.get_value(),
        )
        inference_config = InferenceConfig(
            batch_size=self._inference_settings.batch_size.get_value(),
            middle_trim=self._inference_settings.middle_trim.get_value(),
            experiment_number=self._inference_settings.experiment_number.get_value(),
            patch_weighting=self._inference_settings.patch_weighting_method.get_value(),
            pad_eval=self._inference_settings.pad_eval.get_value(),
            window=self._inference_settings.window.get_value(),
        )
        datagen_config = DatagenConfig()

        return ConfigManager(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            inference_config=inference_config,
            datagen_config=datagen_config,
        )

    def _sync_config_to_settings(self, config_manager: ConfigManager) -> None:
        """Update ptychodus settings to match the configs from the loaded checkpoint.

        Mirrors the field mapping in `_create_config_from_settings` in reverse so
        that what the user sees in the UI accurately reflects what the loaded
        model was trained with.
        """
        data_config = config_manager.data_config
        model_config = config_manager.model_config
        training_config = config_manager.training_config
        inference_config = config_manager.inference_config

        if model_config.mode != self._model_training_mode:
            logger.warning(
                'Loaded checkpoint mode %r does not match reconstructor mode %r; '
                'predictions may be inconsistent.',
                model_config.mode,
                self._model_training_mode,
            )

        d = self._data_settings
        d.model_size.set_value(data_config.N)
        d.num_channels.set_value(data_config.C)
        d.data_normalization_mode.set_value(data_config.normalize)
        d.neighbor_lookup_method.set_value(data_config.neighbor_function)
        d.scan_pattern.set_value(data_config.scan_pattern)
        d.normalize_probe.set_value(data_config.probe_normalize)
        d.x_lower_bound.set_value(data_config.x_bounds[0])
        d.x_upper_bound.set_value(data_config.x_bounds[1])
        d.y_lower_bound.set_value(data_config.y_bounds[0])
        d.y_upper_bound.set_value(data_config.y_bounds[1])
        d.min_neighbor_distance.set_value(data_config.min_neighbor_distance)
        d.max_neighbor_distance.set_value(data_config.max_neighbor_distance)
        d.num_nearest_neighbors_for_quadrant_lookup.set_value(data_config.K_quadrant)
        d.num_photons.set_value(data_config.nphotons)
        d.coordinate_subsampling_factor.set_value(data_config.n_subsample)
        d.probe_scale.set_value(data_config.probe_scale)
        d.num_nearest_neighbors_for_lookup.set_value(data_config.K)
        d.grid_size_y.set_value(data_config.grid_size[0])
        d.grid_size_x.set_value(data_config.grid_size[1])
        d.probe_ramp_removal.set_value(data_config.probe_ramp_removal)
        d.data_scaling_method.set_value(data_config.data_scaling)
        d.subtract_mean_phase.set_value(data_config.phase_subtraction)

        m = self._model_settings
        m.object_big.set_value(model_config.object_big)
        m.probe_big.set_value(model_config.probe_big)
        m.loss_function.set_value(model_config.loss_function)
        m.amplitude_activation_function.set_value(model_config.amp_activation)
        m.cbam_encoder.set_value(model_config.cbam_encoder)
        m.use_shared_decoder.set_value(model_config.use_shared_decoder)
        m.intensity_scale_trainable.set_value(model_config.intensity_scale_trainable)
        m.intensity_scale.set_value(model_config.intensity_scale)
        m.max_position_jitter.set_value(model_config.max_position_jitter)
        m.num_datasets.set_value(model_config.num_datasets)
        m.auxiliary_amplitude_loss.set_value(
            'None' if model_config.amp_loss is None else model_config.amp_loss
        )
        m.auxiliary_phase_loss.set_value(
            'None' if model_config.phase_loss is None else model_config.phase_loss
        )
        m.auxiliary_amplitude_loss_coeff.set_value(model_config.amp_loss_coeff)
        m.auxiliary_phase_loss_coeff.set_value(model_config.phase_loss_coeff)
        m.num_filters_scale.set_value(model_config.n_filters_scale)
        m.eca_decoder.set_value(model_config.eca_decoder)
        m.use_batch_normalization.set_value(model_config.batch_norm)
        m.edge_pad.set_value(model_config.edge_pad)
        m.decoder_last_c_outer_fraction.set_value(model_config.decoder_last_c_outer_fraction)
        m.cbam_bottleneck.set_value(model_config.cbam_bottleneck)
        m.cbam_decoder.set_value(model_config.cbam_decoder)
        m.spatial_decoder.set_value(model_config.spatial_decoder)
        m.decoder_spatial_kernel.set_value(model_config.decoder_spatial_kernel)
        m.eca_encoder.set_value(model_config.eca_encoder)
        m.offset.set_value(model_config.offset)
        m.probe_reference_loss_coeff.set_value(model_config.probe_reference_coeff)
        m.amplitude_variance_loss.set_value(model_config.amplitude_variance_loss)
        m.amplitude_variance_coeff.set_value(model_config.amplitude_variance_coeff)

        t = self._training_settings
        t.epochs.set_value(training_config.epochs)
        t.batch_size.set_value(training_config.batch_size)
        t.learning_rate.set_value(training_config.learning_rate)
        # n_devices may be 'auto' (str) or an int; the settings field is int-only,
        # so warn rather than silently leaving the UI out of sync.
        if isinstance(training_config.n_devices, int):
            t.num_devices.set_value(training_config.n_devices)
        else:
            logger.warning(
                'Loaded checkpoint n_devices=%r is non-integer; UI num_devices left at %d.',
                training_config.n_devices,
                t.num_devices.get_value(),
            )
        t.num_dataloader_workers.set_value(training_config.num_workers)
        t.gradient_accumulation_steps.set_value(training_config.accum_steps)
        t.epochs_finetune.set_value(training_config.epochs_fine_tune)
        t.finetune_gamma.set_value(training_config.fine_tune_gamma)
        t.gradient_clip_val.set_value(
            training_config.gradient_clip_val
            if training_config.gradient_clip_val is not None
            else 0.0
        )
        t.experiment_name.set_value(training_config.experiment_name)
        t.use_negative_log_likelihood_loss.set_value(training_config.nll)
        t.device.set_value(training_config.device)
        t.learning_rate_scheduler.set_value(training_config.scheduler)
        t.learning_rate_warmup_epochs.set_value(training_config.warmup_epochs)
        t.minimum_learning_rate_ratio.set_value(training_config.min_lr_ratio)
        t.notes.set_value(training_config.notes)
        t.model_name.set_value(training_config.model_name)
        t.enable_staged_finetuning.set_value(training_config.enable_staged_finetuning)
        t.finetune_stage1_epochs.set_value(training_config.finetune_stage1_epochs)
        t.finetune_stage2_epochs.set_value(training_config.finetune_stage2_epochs)
        t.finetune_stage3_epochs.set_value(training_config.finetune_stage3_epochs)
        t.finetune_stage1_lr_decoder.set_value(training_config.finetune_stage1_lr_decoder)
        t.finetune_stage2_lr_encoder_top.set_value(training_config.finetune_stage2_lr_encoder_top)
        t.finetune_stage2_lr_decoder.set_value(training_config.finetune_stage2_lr_decoder)
        t.finetune_stage2_lr_phase_head.set_value(training_config.finetune_stage2_lr_phase_head)
        t.finetune_stage3_lr_encoder_bottom.set_value(
            training_config.finetune_stage3_lr_encoder_bottom
        )
        t.finetune_stage3_lr_encoder_top.set_value(training_config.finetune_stage3_lr_encoder_top)
        t.finetune_stage3_lr_decoder.set_value(training_config.finetune_stage3_lr_decoder)
        t.finetune_stage3_lr_phase_head.set_value(training_config.finetune_stage3_lr_phase_head)
        t.finetune_skip_stage3.set_value(training_config.finetune_skip_stage3)
        t.finetune_early_stop_patience.set_value(training_config.finetune_early_stop_patience)
        t.finetune_validation_split.set_value(training_config.finetune_val_split)

        i = self._inference_settings
        i.batch_size.set_value(inference_config.batch_size)
        i.middle_trim.set_value(inference_config.middle_trim)
        i.experiment_number.set_value(inference_config.experiment_number)
        i.patch_weighting_method.set_value(inference_config.patch_weighting)
        i.pad_eval.set_value(inference_config.pad_eval)
        i.window.set_value(inference_config.window)

    @property
    def name(self) -> str:
        return self._model_training_mode

    def get_progress_goal(self) -> int:
        # Reflects the main training fit only; finetuning (epochs_fine_tune,
        # staged finetuning) runs on separate Lightning Trainer instances we
        # don't observe.
        return self._training_settings.epochs.get_value()

    def reconstruct(self, parameters: ReconstructInput) -> Iterator[ReconstructOutput]:
        if self._inference_engine is None or self._inference_config_manager is None:
            raise RuntimeError('Model must be loaded before reconstruction.')

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
            config_manager=self._inference_config_manager,
        )
        object_out_array = numpy.asarray(self._inference_engine.predict_and_stitch(data_loader))

        # predict_and_stitch returns a 2D (H, W) array. Object accepts 2- or
        # 3-D; pass it through and let Object validate against layer_spacing_m
        # rather than blanket-squeezing (which would also collapse legitimate
        # singleton spatial axes).
        object_in = parameters.product.object_
        object_out = Object(
            array=object_out_array,
            layer_spacing_m=object_in.layer_spacing_m,
            pixel_geometry=object_in.get_pixel_geometry(),
            center=object_in.get_center(),
        )

        # TODO: Fourier error
        losses: Sequence[LossValue] = []
        product = Product(
            metadata=parameters.product.metadata,
            probe_positions=parameters.product.probe_positions,
            probes=parameters.product.probes,
            object_=object_out,
            losses=losses,
        )

        yield ReconstructOutput(product)

    def is_model_loaded(self) -> bool:
        return self._inference_engine is not None

    def get_model_file_filter(self) -> str:
        return 'PyTorch Lightning Checkpoint Files (*.ckpt)'

    def load_model_from_file(self, file_path: Path) -> None:
        # The checkpoint is the authority for architecture-critical configs.
        # Read the saved hyperparameters directly from the .ckpt rather than
        # rebuilding configs from current settings (which may have drifted).
        data_config, model_config, training_config, inference_config = (
            PtychoModel._extract_configs_from_checkpoint(str(file_path))
        )
        if any(c is None for c in (data_config, model_config, training_config, inference_config)):
            raise ValueError(
                f'Checkpoint at {file_path} is missing one or more saved configs '
                f'(data/model/training/inference).'
            )

        ptycho_model = PtychoModel(
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
        )
        ptycho_model.model = PtychoPINN_Lightning.load_from_checkpoint(
            file_path,
            model_config=model_config,
            data_config=data_config,
            training_config=training_config,
            inference_config=inference_config,
        )

        # Build the canonical ConfigManager from the loaded model (frozen-mode),
        # then audit field-by-field to catch any silent drift.
        config_manager = ConfigManager.from_loaded_model(ptycho_model)
        config_manager.validate_arch_compatibility(ptycho_model)

        # Mirror the loaded configs back into ptychodus settings so the UI stays
        # consistent with what the model was actually trained with.
        self._sync_config_to_settings(config_manager)

        self._inference_engine = InferenceEngine(
            config_manager=config_manager, ptycho_model=ptycho_model
        )
        # Cache the canonical ConfigManager for reuse in reconstruct(); InferenceEngine
        # only stores the four config dataclasses, not the manager itself.
        self._inference_config_manager = config_manager
        self._loaded_from = file_path

    def get_model_file_extension(self) -> str:
        return '.ckpt'

    def save_model(self, file_path: Path) -> None:
        if self._inference_engine is None:
            raise RuntimeError('Cannot save PtychoPINN_Torch model: model is not loaded.')
        if self._loaded_from is None or not self._loaded_from.is_file():
            raise RuntimeError(
                'Cannot save PtychoPINN_Torch model: no source checkpoint to copy from. '
                'Train or load a model first.'
            )
        logger.debug(f'Copying loaded checkpoint "{self._loaded_from}" -> "{file_path}"')
        shutil.copyfile(self._loaded_from, file_path)

    def get_training_data_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        save_ptychopinn_training_data(file_path, parameters, multimodal_probe=True)

    def train(self, input_path: Path, output_path: Path) -> Iterator[TrainOutput]:
        config_manager = self._create_config_from_settings()
        data_loader = PtychoDataLoader(
            data_dir=input_path,
            config_manager=config_manager,
            data_format=DataloaderFormats('lightning_only_module'),
            output_dir=output_path,
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
            experiment_name=self._training_settings.experiment_name.get_value(),
        )
        # PtychoDataLoader appends a `run_<timestamp>` segment to output_dir,
        # and the checkpoint callback writes there — not at output_path.
        run_dir = Path(data_loader.output_dir)
        checkpoint_path = find_best_checkpoint(run_dir)

        if checkpoint_path is None:
            raise FileNotFoundError(f'No checkpoints found in {run_dir} after training.')
        else:
            self.load_model_from_file(checkpoint_path)

        yield TrainOutput(
            training_loss=loss_collector.training_loss,
            validation_loss=loss_collector.validation_loss,
            progress=loss_collector.epochs_completed,
        )
