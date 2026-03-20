from __future__ import annotations
from collections.abc import Iterator, Sequence
from datetime import datetime
from pathlib import Path
import logging

import numpy

from ptycho_torch.api.base_api import (
    ConfigManager,
    DataConfig,
    DatagenConfig,
    DataloaderFormats,
    InferenceConfig,
    InferenceEngine,
    ModelConfig,
    PtychoDataLoader,
    PtychoModel,
    Trainer,
    TrainingConfig,
)
from ptycho_torch.model import PtychoPINN_Lightning

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


class PtychoPINNTorchTrainableReconstructor(TrainableReconstructor):
    def __init__(
        self,
        name: str,
        data_settings: PtychoPINNTorchDataSettings,
        model_settings: PtychoPINNTorchModelSettings,
        inference_settings: PtychoPINNTorchInferenceSettings,
        training_settings: PtychoPINNTorchTrainingSettings,
        *,
        is_developer_mode_enabled: bool,
    ) -> None:
        super().__init__()
        self._name = name
        self._data_settings = data_settings
        self._model_settings = model_settings
        self._inference_settings = inference_settings
        self._training_settings = training_settings
        self._is_developer_mode_enabled = is_developer_mode_enabled

        self._model: PtychoModel | None = None

    def _create_config_manager(self) -> ConfigManager:
        grid_size = (
            self._data_settings.grid_size_x.get_value(),
            self._data_settings.grid_size_y.get_value(),
        )  # FIXME verify order
        x_bounds = (
            self._data_settings.x_lower_bound.get_value(),
            self._data_settings.x_upper_bound.get_value(),
        )
        y_bounds = (
            self._data_settings.y_lower_bound.get_value(),
            self._data_settings.y_upper_bound.get_value(),
        )
        data_config = DataConfig(
            nphotons=self._data_settings.nphotons.get_value(),
            N=self._data_settings.N.get_value(),
            C=self._data_settings.C.get_value(),
            K=self._data_settings.K.get_value(),
            K_quadrant=self._data_settings.K_quadrant.get_value(),
            n_subsample=self._data_settings.n_subsample.get_value(),
            subsample_seed=None,
            grid_size=grid_size,
            neighbor_function=self._data_settings.neighbor_function.get_value(),
            min_neighbor_distance=self._data_settings.min_neighbor_distance.get_value(),
            max_neighbor_distance=self._data_settings.max_neighbor_distance.get_value(),
            scan_pattern=self._data_settings.scan_pattern.get_value(),
            normalize=self._data_settings.normalize.get_value(),
            probe_scale=self._data_settings.probe_scale.get_value(),
            probe_normalize=self._data_settings.probe_normalize.get_value(),
            probe_ramp_removal=self._data_settings.probe_ramp_removal.get_value(),
            data_scaling=self._data_settings.data_scaling.get_value(),
            phase_subtraction=self._data_settings.phase_subtraction.get_value(),
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        model_config = ModelConfig(
            mode=self._model_settings.mode.get_value(),
            intensity_scale_trainable=self._model_settings.intensity_scale_trainable.get_value(),
            intensity_scale=self._model_settings.intensity_scale.get_value(),
            max_position_jitter=self._model_settings.max_position_jitter.get_value(),
            num_datasets=self._model_settings.num_datasets.get_value(),
            C_model=self._model_settings.C_model.get_value(),
            n_filters_scale=self._model_settings.n_filters_scale.get_value(),
            amp_activation=self._model_settings.amp_activation.get_value(),
            batch_norm=self._model_settings.batch_norm.get_value(),
            probe_mask=None,  # FIXME
            edge_pad=self._model_settings.edge_pad.get_value(),
            decoder_last_c_outer_fraction=self._model_settings.decoder_last_c_outer_fraction.get_value(),
            decoder_last_amp_channels=self._model_settings.decoder_last_amp_channels.get_value(),
            use_shared_decoder=self._model_settings.use_shared_decoder.get_value(),
            eca_encoder=self._model_settings.eca_encoder.get_value(),
            cbam_encoder=self._model_settings.cbam_encoder.get_value(),
            cbam_bottleneck=self._model_settings.cbam_bottleneck.get_value(),
            cbam_decoder=self._model_settings.cbam_decoder.get_value(),
            eca_decoder=self._model_settings.eca_decoder.get_value(),
            spatial_decoder=self._model_settings.spatial_decoder.get_value(),
            decoder_spatial_kernel=self._model_settings.decoder_spatial_kernel.get_value(),
            object_big=self._model_settings.object_big.get_value(),
            probe_big=self._model_settings.probe_big.get_value(),
            offset=self._model_settings.offset.get_value(),
            C_forward=self._model_settings.C_forward.get_value(),
            pad_object=self._model_settings.pad_object.get_value(),
            gaussian_smoothing_sigma=self._model_settings.gaussian_smoothing_sigma.get_value(),
            loss_function=self._model_settings.loss_function.get_value(),
            amp_loss=self._model_settings.amp_loss.get_value(),
            phase_loss=self._model_settings.phase_loss.get_value(),
            amp_loss_coeff=self._model_settings.amp_loss_coeff.get_value(),
            phase_loss_coeff=self._model_settings.phase_loss_coeff.get_value(),
            probe_reference_coeff=self._model_settings.probe_reference_coeff.get_value(),
        )
        training_config = TrainingConfig(
            training_directories=[''],  # FIXME
            nll=self._training_settings.nll.get_value(),
            device=self._training_settings.device.get_value(),
            strategy=self._training_settings.strategy.get_value(),
            n_devices=self._training_settings.n_devices.get_value(),
            framework=self._training_settings.framework.get_value(),
            orchestrator=self._training_settings.orchestrator.get_value(),
            learning_rate=self._training_settings.learning_rate.get_value(),
            epochs=self._training_settings.epochs.get_value(),
            batch_size=self._training_settings.batch_size.get_value(),
            epochs_fine_tune=self._training_settings.epochs_fine_tune.get_value(),
            fine_tune_gamma=self._training_settings.fine_tune_gamma.get_value(),
            scheduler=self._training_settings.scheduler.get_value(),
            num_workers=self._training_settings.num_workers.get_value(),
            accum_steps=self._training_settings.accum_steps.get_value(),
            gradient_clip_val=self._training_settings.gradient_clip_val.get_value(),
            warmup_epochs=self._training_settings.lr_warmup_epochs.get_value(),
            min_lr_ratio=self._training_settings.min_lr_ratio.get_value(),
            stage_1_epochs=self._training_settings.stage_1_epochs.get_value(),
            stage_2_epochs=self._training_settings.stage_2_epochs.get_value(),
            stage_3_epochs=self._training_settings.stage_3_epochs.get_value(),
            physics_weight_schedule=self._training_settings.physics_weight_schedule.get_value(),
            stage_3_lr_factor=self._training_settings.stage_3_lr_factor.get_value(),
            torch_loss_mode=self._training_settings.torch_loss_mode.get_value(),
            experiment_name=self._training_settings.experiment_name.get_value(),
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
            finetune_val_split=self._training_settings.finetune_val_split.get_value(),
            output_dir=self._training_settings.output_dir.get_value(),
            train_data_file='',  # FIXME
            test_data_file='',  # FIXME
            n_groups=self._training_settings.n_groups.get_value(),
        )
        inference_config = InferenceConfig(
            middle_trim=self._inference_settings.middle_trim.get_value(),
            batch_size=self._inference_settings.batch_size.get_value(),
            experiment_number=self._inference_settings.experiment_number.get_value(),
            pad_eval=self._inference_settings.pad_eval.get_value(),
            window=self._inference_settings.window.get_value(),
            patch_weighting=self._inference_settings.patch_weighting.get_value(),
        )
        datagen_config = DatagenConfig()

        return ConfigManager(
            data_config=data_config,
            model_config=model_config,
            training_config=training_config,
            inference_config=inference_config,
            datagen_config=datagen_config,
        )

    @property
    def name(self) -> str:
        return self._name

    def get_progress_goal(self) -> int:
        return 0

    def reconstruct(self, parameters: ReconstructInput) -> Iterator[ReconstructOutput]:
        if self._model is None:
            raise RuntimeError('Model must be loaded before reconstruction.')

        config_manager = self._create_config_manager()  # FIXME reuse config manager
        ptycho_data_dir = Path()  # FIXME
        tensordict_dataloader = PtychoDataLoader(
            data_dir=ptycho_data_dir, config_manager=config_manager, data_format='tensordict'
        )
        ptycho_inference = InferenceEngine(config_manager=config_manager, ptycho_model=self._model)
        object_out_array = ptycho_inference.predict_and_stitch(tensordict_dataloader)

        object_in = parameters.product.object_
        object_out = Object(
            array=numpy.squeeze(object_out_array),
            layer_spacing_m=object_in.layer_spacing_m,
            pixel_geometry=object_in.get_pixel_geometry(),
            center=object_in.get_center(),
        )
        losses: Sequence[LossValue] = list()

        product = Product(
            metadata=parameters.product.metadata,
            probe_positions=parameters.product.probe_positions,
            probes=parameters.product.probes,
            object_=object_out,
            losses=losses,
        )

        yield ReconstructOutput(product)

    def is_model_loaded(self):
        return self._model is not None

    def get_model_file_filter(self) -> str:
        return 'PyTorch Lightning Checkpoint Files (*.ckpt)'

    def load_model_from_file(self, file_path: Path) -> None:
        json_base_path = Path()  # FIXME
        self._model = PtychoModel._load(
            config_manager=self._create_config_manager(),
            strategy='lightning',
            run_path=json_base_path,  # FIXME
            model_class=PtychoPINN_Lightning,
        )

    def get_training_data_file_filter(self) -> str:
        return 'NumPy Zipped Archive (*.npz)'

    def export_training_data(self, file_path: Path, parameters: ReconstructInput) -> None:
        # TODO extract & share with ptychopinn
        object_geometry = parameters.product.object_.get_geometry()
        position_x_px: list[float] = list()
        position_y_px: list[float] = list()

        for scan_point in parameters.product.probe_positions:
            object_point = object_geometry.map_coordinates_probe_to_object(scan_point)
            position_x_px.append(object_point.coordinate_x_px)
            position_y_px.append(object_point.coordinate_y_px)

        xcoords = numpy.array(position_x_px)
        ycoords = numpy.array(position_y_px)

        numpy.savez(
            file_path,
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=parameters.diffraction_patterns,
            probeGuess=parameters.product.probes.get_probe_no_opr().get_incoherent_mode(0),
            # assume that all patches are from the same object
            objectGuess=parameters.product.object_.get_layer(0),
            scan_index=numpy.zeros(len(parameters.product.probe_positions), dtype=int),
        )

    def train(self, input_path: Path, output_path: Path) -> Iterator[TrainOutput]:
        config_manager = self._create_config_manager()  # FIXME reuse config manager?
        timestamp = datetime.now()  # FIXME

        lightning_dataloader = PtychoDataLoader(
            data_dir=input_path,
            config_manager=config_manager,
            data_format=DataloaderFormats('lightning_only_module'),
            output_dir=output_path,
            timestamp=timestamp,  # FIXME
        )

        new_ptycho_model = PtychoModel._new_model(
            model=PtychoPINN_Lightning, config_manager=config_manager
        )
        lightning_trainer = Trainer._from_lightning(
            model=new_ptycho_model,
            dataloader=lightning_dataloader,
            orchestration='lightning',
            config_manager=config_manager,
        )
        output_dir = lightning_trainer.train(orchestration='lightning', experiment_name='test_run')

        new_destination = output_path / 'new_ptycho_model'  # FIXME file suffix
        new_ptycho_model.save(
            path=new_destination, source_run_path=output_dir, strategy='lightning'
        )
        self._model = new_ptycho_model

        yield TrainOutput()  # TODO yield losses & progress
