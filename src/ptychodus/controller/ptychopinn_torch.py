from PyQt5.QtWidgets import QWidget

from ..model.ptychopinn_torch.core import PtychoPINNTorchReconstructorLibrary
from .data import FileDialogFactory
from .parametric import ParameterViewBuilder
from .processing import ReconstructorViewControllerFactory


class PtychoPINNTorchViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(
        self, model: PtychoPINNTorchReconstructorLibrary, file_dialog_factory: FileDialogFactory
    ) -> None:
        super().__init__()
        self._model = model
        self._file_dialog_factory = file_dialog_factory

    @property
    def name(self) -> str:
        return 'PtychoPINN-Torch'

    def create_view_controller(self, reconstructor_name: str) -> QWidget:
        builder = ParameterViewBuilder(self._file_dialog_factory)

        data_group = 'Data'
        data_settings = self._model.data_settings
        builder.add_decimal_line_edit(
            data_settings.nphotons, 'Number of Photons:', group=data_group
        )
        builder.add_integer_line_edit(
            data_settings.N, 'Diffraction Pattern Size:', group=data_group
        )
        builder.add_integer_line_edit(data_settings.C, 'Channels:', group=data_group)
        builder.add_integer_line_edit(
            data_settings.K, 'Nearest Neighbors for Lookup:', group=data_group
        )
        builder.add_integer_line_edit(
            data_settings.K_quadrant, 'Nearest Neighbors for Quadrant Lookup:', group=data_group
        )
        builder.add_integer_line_edit(
            data_settings.n_subsample, 'Coordinate Subsampling Factor:', group=data_group
        )
        builder.add_integer_line_edit(
            data_settings.grid_size_x, 'Scan Grid Size X:', group=data_group
        )
        builder.add_integer_line_edit(
            data_settings.grid_size_y, 'Scan Grid Size Y:', group=data_group
        )
        builder.add_line_edit(
            data_settings.neighbor_function, 'Neighbor Function:', group=data_group
        )  # FIXME make enum
        builder.add_decimal_line_edit(
            data_settings.min_neighbor_distance, 'Min Neighbor Distance [px]:', group=data_group
        )
        builder.add_decimal_line_edit(
            data_settings.max_neighbor_distance, 'Max Neighbor Distance [px]:', group=data_group
        )
        builder.add_line_edit(
            data_settings.scan_pattern,
            'Scan Pattern:',
            tool_tip='Used for four quadrant neighbor function.',
            group=data_group,
        )  # FIXME make enum
        builder.add_line_edit(
            data_settings.normalize, 'Normalize:', group=data_group
        )  # FIXME make enum
        builder.add_decimal_line_edit(data_settings.probe_scale, 'Probe Scale:', group=data_group)
        builder.add_check_box(data_settings.probe_normalize, 'Normalize Probe', group=data_group)
        builder.add_line_edit(
            data_settings.data_scaling, 'Data Scaling:', group=data_group
        )  # FIXME make enum
        builder.add_check_box(
            data_settings.phase_subtraction,
            'Subtract Phase',
            tool_tip='Only useful for supervised training dataset.',
            group=data_group,
        )
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

        model_group = 'Model'
        model_settings = self._model.model_settings
        builder.add_line_edit(
            model_settings.mode, 'Training Mode:', group=model_group
        )  # FIXME make enum
        builder.add_line_edit(
            model_settings.architecture, 'Architecture:', group=model_group
        )  # FIXME make enum
        builder.add_integer_line_edit(model_settings.fno_modes, 'FNO Modes:', group=model_group)
        builder.add_integer_line_edit(model_settings.fno_width, 'FNO Width:', group=model_group)
        builder.add_integer_line_edit(model_settings.fno_blocks, 'FNO Blocks:', group=model_group)
        builder.add_integer_line_edit(
            model_settings.fno_cnn_blocks, 'FNO CNN Blocks:', group=model_group
        )
        builder.add_line_edit(
            model_settings.fno_input_transform, 'FNO Input Transform:', group=model_group
        )  # FIXME make enum
        builder.add_integer_line_edit(
            model_settings.max_hidden_channels, 'Max Hidden Channels:', group=model_group
        )
        builder.add_integer_line_edit(
            model_settings.resnet_width, 'ResNet Width:', group=model_group
        )
        builder.add_line_edit(
            model_settings.generator_output_mode, 'Generator Output Mode:', group=model_group
        )  # FIXME make enum
        builder.add_check_box(
            model_settings.intensity_scale_trainable, 'Intensity Scale Trainable', group=model_group
        )
        builder.add_decimal_line_edit(
            model_settings.intensity_scale, 'Intensity Scale:', group=model_group
        )
        builder.add_integer_line_edit(
            model_settings.max_position_jitter,
            'Max Position Jitter:',
            tool_tip='Deprecated in Torch path (padded size ignores this)',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.num_datasets,
            'Num Datasets:',
            tool_tip='Number of unique datasets being trained.',
            group=model_group,
        )
        builder.add_integer_line_edit(model_settings.C_model, 'C Model:', group=model_group)
        builder.add_integer_line_edit(
            model_settings.n_filters_scale,
            'Num Filters Scale:',
            tool_tip='Shrinking factor for channels in network layers.',
            group=model_group,
        )
        builder.add_line_edit(
            model_settings.amp_activation,
            'Amplitude Activation:',
            tool_tip='Activation function for amplitude part',
            group=model_group,
        )  # FIXNE make enum
        builder.add_check_box(
            model_settings.batch_norm,
            'Batch Normalization',
            tool_tip='Whether to use batch normalization',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.edge_pad,
            'Edge Padding:',
            tool_tip='For padding the decoder_last reconstruction.',
            group=model_group,
        )
        builder.add_decimal_line_edit(
            model_settings.decoder_last_c_outer_fraction,
            'Decoder Last C Outer Fraction:',
            tool_tip='Amount of channels going to higher frequency components in decoder_last',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.decoder_last_amp_channels,
            'Decoder Last Amp Channels:',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.eca_encoder,
            'ECA Encoder',
            tool_tip='Whether Efficient Channel Attention (ECA) is turned on for encoder.',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.cbam_encoder,
            'CBAM Encoder',
            tool_tip='Whether Convolutional Block Attention Module (CBAM) is turned on for encoder.',
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
            model_settings.eca_decoder,
            'ECA Decoder',
            tool_tip='Whether Efficient Channel Attention (ECA) is turned on for decoder.',
            group=model_group,
        )
        builder.add_check_box(
            model_settings.spatial_decoder,
            'Spatial Attention Decoder',
            tool_tip='Whether Spatial Attention is turned on for decoder.',
            group=model_group,
        )
        builder.add_integer_line_edit(
            model_settings.decoder_spatial_kernel,
            'Spatial Atttention Kernel:',
            tool_tip='Spatial attention kernel for decoder.',
            group=model_group,
        )
        builder.add_check_box(model_settings.object_big, 'Object Big', group=model_group)
        builder.add_check_box(model_settings.probe_big, 'Probe Big', group=model_group)
        builder.add_integer_line_edit(
            model_settings.offset,
            'Offset:',
            tool_tip='Offset parameter for nearest neighbor patches.',
            group=model_group,
        )
        builder.add_integer_line_edit(model_settings.C_forward, 'Num Channels:', group=model_group)
        builder.add_check_box(
            model_settings.pad_object,
            'Pad Object',
            tool_tip='Pad object during forward model.',
            group=model_group,
        )
        builder.add_decimal_line_edit(
            model_settings.gaussian_smoothing_sigma,
            'Gaussian Smoothing Sigma:',
            tool_tip='Gaussian smoothing sigma for probe.',
            group=model_group,
        )
        builder.add_line_edit(
            model_settings.loss_function, 'Loss Function:', group=model_group
        )  # FIXME make enum
        builder.add_line_edit(
            model_settings.amp_loss, 'Amplitude Loss Function:', group=model_group
        )  # FIXME make enum
        builder.add_decimal_line_edit(
            model_settings.amp_loss_coeff, 'Amplitude Loss Coefficient:', group=model_group
        )
        builder.add_line_edit(
            model_settings.phase_loss, 'Phase Loss Function:', group=model_group
        )  # FIXME make enum
        builder.add_decimal_line_edit(
            model_settings.phase_loss_coeff, 'Phase Loss Coefficient:', group=model_group
        )

        inference_group = 'Inference'
        inference_settings = self._model.inference_settings
        builder.add_integer_line_edit(
            inference_settings.middle_trim, 'Middle Trim:', group=inference_group
        )
        builder.add_integer_line_edit(
            inference_settings.batch_size,
            'Batch Size:',
            tool_tip='Batch size for reconstruction.',
            group=inference_group,
        )
        builder.add_integer_line_edit(
            inference_settings.experiment_number, 'Experiment Number:', group=inference_group
        )
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
        builder.add_check_box(
            inference_settings.log_patch_stats,
            'Log Patch Statistics',
            tool_tip='Emit patch statistics during training & inference.',
            group=inference_group,
        )
        builder.add_integer_line_edit(
            inference_settings.patch_stats_limit,
            'Patch Statistics Limit:',
            tool_tip='Maximum number of batches to log.',
            group=inference_group,
        )

        training_group = 'Training'
        training_settings = self._model.training_settings
        builder.add_check_box(
            training_settings.nll, 'Use Negative Log Likelihood Loss', group=training_group
        )
        builder.add_line_edit(
            training_settings.device,
            'Device:',
            tool_tip='Device to train on ("cuda", "cpu", etc.)',
            group=training_group,
        )  # FIXME improve
        builder.add_integer_line_edit(
            training_settings.n_devices,
            'Num. Devices:',
            tool_tip='Number of devices to train on',
            group=training_group,
        )  # FIXME improve
        builder.add_line_edit(
            training_settings.strategy, 'Distributed Training Strategy:', group=training_group
        )  # FIXME make enum
        builder.add_line_edit(
            training_settings.framework, 'Training Framework:', group=training_group
        )  # FIXME make enum
        builder.add_line_edit(
            training_settings.orchestrator, 'Training Orchestrator:', group=training_group
        )  # FIXME make enum
        builder.add_decimal_line_edit(
            training_settings.learning_rate, 'Learning Rate:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.epochs, 'Default Epochs:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.batch_size, 'Batch Size:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.epochs_fine_tune, 'Fine Tune Epochs:', group=training_group
        )
        builder.add_decimal_line_edit(
            training_settings.fine_tune_gamma,
            'Fine Tune Gamma:',
            tool_tip='Scales base learning rate for fine-tuning.',
            group=training_group,
        )
        builder.add_line_edit(
            training_settings.scheduler, 'Scheduler:', group=training_group
        )  # FIXME make enum

        builder.add_integer_line_edit(
            training_settings.lr_warmup_epochs,
            'Warmup Epochs:',
            tool_tip='Number of warmup epochs for WarmupCosine scheduler',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.min_lr_ratio,
            'Min Learning Rate Ratio:',
            tool_tip='Minimum learning rate ratio for WarmupCoside scheduler',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.plateau_factor, 'Plateau Factor:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.plateau_patience, 'Plateau Patience:', group=training_group
        )
        builder.add_decimal_line_edit(
            training_settings.plateau_min_lr, 'Plateau Min Learning Rate:', group=training_group
        )
        builder.add_decimal_line_edit(
            training_settings.plateau_threshold, 'Plateau Threshold:', group=training_group
        )

        builder.add_integer_line_edit(
            training_settings.num_workers, 'Num Dataloader Workers:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.accum_steps, 'Batch Size Accumulation Steps:', group=training_group
        )
        builder.add_decimal_line_edit(
            training_settings.gradient_clip_val, 'Gradient Clip Value:', group=training_group
        )
        builder.add_line_edit(
            training_settings.gradient_clip_algorithm,
            'Gradient Clip Algorithm:',
            group=training_group,
        )  # FIXME make enum

        builder.add_line_edit(training_settings.optimizer, 'Optimizer:', group=training_group)
        builder.add_decimal_line_edit(training_settings.momentum, 'Momentum:', group=training_group)
        builder.add_decimal_line_edit(
            training_settings.weight_decay,
            'Weight Decay:',
            tool_tip='L2 Penalty',
            group=training_group,
        )
        builder.add_decimal_line_edit(
            training_settings.adam_beta1, 'Adam/AdamW beta1:', group=training_group
        )
        builder.add_decimal_line_edit(
            training_settings.adam_beta2, 'Adam/AdamW beta2:', group=training_group
        )
        builder.add_check_box(
            training_settings.log_grad_norm, 'Log Grad Norm', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.grad_norm_log_freq, 'Grad Norm Log Freq:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.batch_size, 'Batch Size:', group=training_group
        )

        builder.add_integer_line_edit(
            training_settings.stage_1_epochs,
            'Stage 1 Epochs:',
            tool_tip='Will be set to total epochs if not specified.',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.stage_2_epochs,
            'Stage 2 Epochs:',
            tool_tip='Weighted transition (0 = disabled)',
            group=training_group,
        )
        builder.add_integer_line_edit(
            training_settings.stage_3_epochs,
            'Stage 3 Epochs:',
            tool_tip='Physics only (0 = disabled)',
            group=training_group,
        )
        builder.add_line_edit(
            training_settings.physics_weight_schedule,
            'Physics Weight Schedule:',
            group=training_group,
        )  # FIXME make enum
        builder.add_decimal_line_edit(
            training_settings.stage_3_lr_factor,
            'Stage 3 Learning Rate Reduction:',
            group=training_group,
        )

        builder.add_line_edit(
            training_settings.torch_loss_mode, 'Torch Loss Mode:', group=training_group
        )  # FIXME make enum

        builder.add_line_edit(
            training_settings.experiment_name, 'MLflow Experiment Name:', group=training_group
        )
        builder.add_line_edit(training_settings.notes, 'MLflow Notes:', group=training_group)
        builder.add_line_edit(
            training_settings.model_name, 'MLflow Model Name:', group=training_group
        )

        builder.add_line_edit(
            training_settings.output_dir, 'Lightning Output Directory:', group=training_group
        )
        builder.add_integer_line_edit(
            training_settings.n_groups, 'Number of Grouped Samples:', group=training_group
        )

        return builder.build_widget()
