from PyQt5.QtWidgets import QWidget

from ..model.ptychopinn_torch.core import PtychoPINNTorchReconstructorLibrary
from .data import FileDialogFactory
from .parameters import ParameterViewBuilder
from .processing import ReconstructorViewControllerFactory


class PtychoPINNTorchViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(
        self,
        model: PtychoPINNTorchReconstructorLibrary,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._model = model
        self._file_dialog_factory = file_dialog_factory

    @property
    def name(self) -> str:
        return 'PtychoPINN-Torch'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        builder = ParameterViewBuilder(self._file_dialog_factory)
        enumerators = self._model.enumerators

        # Data
        data_group = 'Data'
        data_settings = self._model.data_settings
        builder.add_integer_line_edit(data_settings.model_size, 'Model Size:', group=data_group)
        builder.add_integer_line_edit(data_settings.num_channels, 'Channels:', group=data_group)
        builder.add_combo_box(
            data_settings.data_normalization_mode,
            enumerators.get_data_normalization_modes(),
            'Normalize:',
            group=data_group,
        )
        builder.add_combo_box(
            data_settings.neighbor_lookup_method,
            enumerators.get_neighbor_lookup_methods(),
            'Neighbor Function:',
            group=data_group,
        )
        builder.add_combo_box(
            data_settings.scan_pattern,
            enumerators.get_scan_patterns(),
            'Scan Pattern:',
            group=data_group,
        )
        builder.add_check_box(data_settings.normalize_probe, 'Normalize Probe', group=data_group)
        builder.add_decimal_line_edit(
            data_settings.x_lower_bound, 'X Lower Bound:', group=data_group
        )
        builder.add_decimal_line_edit(
            data_settings.x_upper_bound, 'X Upper Bound:', group=data_group
        )
        builder.add_decimal_line_edit(
            data_settings.y_lower_bound, 'Y Lower Bound:', group=data_group
        )
        builder.add_decimal_line_edit(
            data_settings.y_upper_bound, 'Y Upper Bound:', group=data_group
        )
        builder.add_decimal_line_edit(
            data_settings.min_neighbor_distance, 'Min Neighbor Distance [px]:', group=data_group
        )
        builder.add_decimal_line_edit(
            data_settings.max_neighbor_distance, 'Max Neighbor Distance [px]:', group=data_group
        )
        builder.add_integer_line_edit(
            data_settings.num_nearest_neighbors_for_quadrant_lookup,
            'Nearest Neighbors for Quadrant Lookup:',
            group=data_group,
        )

        # Data Advanced
        builder.add_integer_line_edit(
            data_settings.coordinate_subsampling_factor,
            'Coordinate Subsampling Factor:',
            group=data_group,
        )
        builder.add_decimal_line_edit(data_settings.probe_scale, 'Probe Scale:', group=data_group)
        builder.add_integer_line_edit(
            data_settings.num_nearest_neighbors_for_lookup,
            'Nearest Neighbors for Lookup:',
            group=data_group,
        )
        builder.add_integer_line_edit(data_settings.grid_size_x, 'Grid Size X:', group=data_group)
        builder.add_integer_line_edit(data_settings.grid_size_y, 'Grid Size Y:', group=data_group)
        builder.add_check_box(
            data_settings.probe_ramp_removal,
            'Probe Ramp Removal',
            group=data_group,
        )
        builder.add_combo_box(
            data_settings.data_scaling_method,
            enumerators.get_data_scaling_methods(),
            'Data Scaling:',
            group=data_group,
        )
        builder.add_check_box(
            data_settings.subtract_mean_phase,
            'Subtract Phase',
            tool_tip='Only useful for supervised training dataset.',
            group=data_group,
        )

        # Model
        model_group = 'Model'
        model_settings = self._model.model_settings
        builder.add_check_box(model_settings.object_big, 'Object Big', group=model_group)
        builder.add_check_box(model_settings.probe_big, 'Probe Big', group=model_group)
        builder.add_combo_box(
            model_settings.loss_function,
            enumerators.get_loss_functions(),
            'Loss Function:',
            group=model_group,
        )
        builder.add_combo_box(
            model_settings.amplitude_activation_function,
            enumerators.get_amplitude_activation_functions(),
            'Amplitude Activation:',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.cbam_encoder,
            'CBAM Encoder',
            tool_tip='Whether Convolutional Block Attention Module (CBAM) is turned on for encoder.',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.use_shared_decoder,
            'Shared Decoder',
            group=model_group,
        )

        # Model Advanced
        builder.add_check_box(
            model_settings.intensity_scale_trainable, 'Intensity Scale Trainable', group=model_group
        )
        builder.add_decimal_line_edit(
            model_settings.intensity_scale, 'Intensity Scale:', group=model_group
        )
        builder.add_integer_line_edit(
            model_settings.max_position_jitter,
            'Max Position Jitter:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.num_datasets,
            'Datasets:',
            group=model_group,
        )
        builder.add_combo_box(
            model_settings.auxiliary_amplitude_loss,
            enumerators.get_auxiliary_loss_functions(),
            'Auxiliary Amplitude Loss Function:',
            group=model_group,
        )
        builder.add_decimal_line_edit(
            model_settings.auxiliary_amplitude_loss_coeff,
            'Auxiliary Amplitude Loss Coefficient:',
            group=model_group,
        )
        builder.add_combo_box(
            model_settings.auxiliary_phase_loss,
            enumerators.get_auxiliary_loss_functions(),
            'Auxiliary Phase Loss Function:',
            group=model_group,
        )
        builder.add_decimal_line_edit(
            model_settings.auxiliary_phase_loss_coeff,
            'Auxiliary Phase Loss Coefficient:',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.num_filters_scale,
            'Num Filters Scale:',
            tool_tip='Shrinking factor for channels in network layers.',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.eca_decoder,
            'ECA Decoder',
            tool_tip='Whether Efficient Channel Attention (ECA) is turned on for decoder.',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.use_batch_normalization,
            'Use Batch Normalization',
            tool_tip='Whether to use batch normalization',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.edge_pad,
            'Edge Padding:',
            tool_tip='For padding the decoder_last reconstruction.',
            group=model_group,
        )
        builder.add_decimal_slider(
            model_settings.decoder_last_c_outer_fraction,
            'Decoder Last C Outer Fraction:',
            tool_tip='Amount of channels going to higher frequency components in decoder_last',
            group=model_group,
        )
        builder.add_check_box(model_settings.cbam_bottleneck, 'CBAM Bottleneck', group=model_group)
        builder.add_check_box(
            model_settings.cbam_decoder,
            'CBAM Decoder',
            tool_tip='Whether Convolutional Block Attention Module (CBAM) is turned on for decoder.',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.spatial_decoder,
            'Decoder Spatial Attention',
            tool_tip='Whether Spatial Attention is turned on for decoder.',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.decoder_spatial_kernel,
            'Decoder Spatial Atttention Kernel:',
            tool_tip='Spatial attention kernel for decoder.',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.eca_encoder,
            'ECA Encoder',
            tool_tip='Whether Efficient Channel Attention (ECA) is turned on for encoder.',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.offset,
            'Offset:',
            tool_tip='Offset parameter for nearest neighbor patches.',
            group=model_group,
        )
        builder.add_decimal_line_edit(
            model_settings.probe_reference_loss_coeff,
            'Probe Reference Loss Coefficient:',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.amplitude_variance_loss,
            'Amplitude Variance Loss',
            tool_tip='Penalize spatial variance of complex modulus to encourage uniform amplitude.',
            group=model_group,
        )
        builder.add_decimal_line_edit(
            model_settings.amplitude_variance_coeff,
            'Amplitude Variance Coefficient:',
            group=model_group,
        )

        # Inference
        inference_group = 'Inference'
        inference_settings = self._model.inference_settings

        builder.add_integer_line_edit(
            inference_settings.batch_size,
            'Batch Size:',
            tool_tip='Batch size for reconstruction.',
            group=inference_group,
        )
        builder.add_integer_line_edit(
            inference_settings.middle_trim, 'Middle Trim:', group=inference_group
        )
        builder.add_integer_line_edit(
            inference_settings.experiment_number, 'Experiment Number:', group=inference_group
        )
        builder.add_combo_box(
            inference_settings.patch_weighting_method,
            enumerators.get_patch_weighting_methods(),
            'Patch Weighting Method:',
            group=inference_group,
        )

        # Inference Advanced
        builder.add_check_box(
            inference_settings.pad_eval,
            'Pad Evaluation Edges',
            tool_tip='Pads the evaluation edges, enforced during training for Nyquist frequency.',
            group=inference_group,
        )
        builder.add_integer_line_edit(
            inference_settings.window,
            'Window Padding:',
            tool_tip='Window padding around reconstruction due to edge errors.',
            group=inference_group,
        )

        # Training
        training_group = 'Training'
        training_settings = self._model.training_settings
        builder.add_integer_line_edit(
            training_settings.epochs, 'Default Epochs:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.batch_size, 'Batch Size:', group=training_group
        )
        builder.add_decimal_line_edit(
            training_settings.learning_rate, 'Learning Rate:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.num_dataloader_workers,
            'Dataloader Workers:',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.gradient_accumulation_steps,
            'Gradient Accumulation Steps:',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.epochs_finetune, 'Fine Tune Epochs:', group=training_group
        )
        builder.add_decimal_slider(
            training_settings.finetune_gamma,
            'Fine Tune Gamma:',
            tool_tip='Scales base learning rate for fine-tuning.',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.gradient_clip_val, 'Gradient Clip Value:', group=training_group
        )

        # Training Advanced
        builder.add_check_box(
            training_settings.use_negative_log_likelihood_loss,
            'Use Negative Log Likelihood Loss',
            group=training_group,
        )
        builder.add_line_edit(
            training_settings.device,
            'Device:',
            tool_tip='Device to train on ("cuda", "cpu", etc.)',
            group=training_group,
        )  # TODO improve
        builder.add_integer_line_edit(
            training_settings.n_devices,
            'Number of GPUs:',
            tool_tip=(
                'Number of CUDA devices to use for training. Lightning fans out to one '
                'process per device via the configured distributed strategy.'
            ),
            group=training_group,
        )
        builder.add_line_edit(
            training_settings.distributed_strategy,
            'Distributed Strategy:',
            tool_tip=(
                "Lightning distributed strategy. Recommended: 'ddp_spawn' for "
                "single-node multi-GPU. 'ddp' requires torchrun and is not wired in "
                "phase 1. 'auto' lets Lightning decide."
            ),
            group=training_group,
        )
        builder.add_line_edit(
            training_settings.visible_gpu_indices,
            'Visible GPU Indices:',
            tool_tip=(
                'Comma-separated CUDA device indices exposed to the training '
                'subprocess (via CUDA_VISIBLE_DEVICES). Leave blank to inherit '
                "ptychodus's environment."
            ),
            group=training_group,
        )
        builder.add_combo_box(
            training_settings.learning_rate_scheduler,
            enumerators.get_learning_rate_schedulers(),
            'Learning Rate Scheduler:',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.learning_rate_warmup_epochs,
            'Warmup Epochs:',
            tool_tip='warmup epochs for WarmupCosine scheduler',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.minimum_learning_rate_ratio,
            'Min Learning Rate Ratio:',
            tool_tip='Minimum learning rate ratio for WarmupCoside scheduler',
            group=training_group,
        )
        builder.add_line_edit(training_settings.notes, 'MLflow Notes:', group=training_group)
        builder.add_line_edit(
            training_settings.model_name, 'MLflow Model Name:', group=training_group
        )

        builder.add_check_box(
            training_settings.enable_staged_finetuning,
            'Enable Staged Fine Tuning',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.finetune_stage1_epochs,
            'Fine Tune Stage 1 Epochs:',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.finetune_stage2_epochs,
            'Fine Tune Stage 2 Epochs:',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.finetune_stage3_epochs,
            'Fine Tune Stage 3 Epochs:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage1_lr_decoder,
            'Fine Tune Stage 1 LR Decoder:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage2_lr_encoder_top,
            'Fine Tune Stage 2 LR Encoder Top:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage2_lr_decoder,
            'Fine Tune Stage 2 LR Decoder:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage2_lr_phase_head,
            'Fine Tune Stage 2 LR Phase Head:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage3_lr_encoder_bottom,
            'Fine Tune Stage 3 LR Encoder Bottom:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage3_lr_encoder_top,
            'Fine Tune Stage 3 LR Encoder Top:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage3_lr_decoder,
            'Fine Tune Stage 3 LR Decoder:',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.finetune_stage3_lr_phase_head,
            'Fine Tune Stage 3 LR Phase Head:',
            group=training_group,
        )
        builder.add_check_box(
            training_settings.finetune_skip_stage3,
            'Skip Fine Tune Stage 3',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.finetune_early_stop_patience,
            'Fine Tune Early Stop Patience:',
            group=training_group,
        )
        builder.add_decimal_slider(
            training_settings.finetune_validation_split,
            'Fine Tune Validation Split:',
            group=training_group,
        )
        return builder.build_widget()
