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
        # FIXME BEGIN
        # # Framework
        # # Training framework. Most of work don in PT was done in lightning
        # framework = string_parameter('framework', 'Lightning')
        # orchestrator = string_parameter('orchestrator', 'Mlflow')
        #
        # # Add other training-specific parameters here as needed
        # learning_rate = real_parameter('learning_rate', 1e-3)
        # # Default epochs number, will be overridden if multi-stage training is active at all
        # epochs = integer_parameter('epochs', 50)
        # batch_size = integer_parameter('batch_size', 16)
        # # Default 0 fine-tune means no fine-tuning
        # epochs_fine_tune = integer_parameter('epochs_fine_tune', 0)
        # # Scales base LR for fine-tuning
        # fine_tune_gamma = real_parameter('fine_tune_gamma', 0.1)
        # scheduler = string_parameter('scheduler', 'Default')
        # # Number of warmup epochs for WarmupCosine scheduler
        # lr_warmup_epochs = integer_parameter('lr_warmup_epochs', 0)
        # # Minimum LR ratio for WarmupCosine scheduler (eta_min = base_lr * ratio)
        # lr_min_ratio = real_parameter('lr_min_ratio', 0.1)
        # plateau_factor = real_parameter('plateau_factor', 0.5)
        # plateau_patience = integer_parameter('plateau_patience', 2)
        # plateau_min_lr = real_parameter('plateau_min_lr', 1e-4)
        # plateau_threshold = real_parameter('plateau_threshold', 0.0)
        # # Dataloader workers
        # num_workers = integer_parameter('num_workers', 4)
        # # Batch size accumulation, manually implemented for DDP
        # accum_steps = integer_parameter('accum_steps', 1)
        # gradient_clip_val = real_parameter('gradient_clip_val', 0.0)
        # # Gradient clipping algorithm: 'norm', 'value', or 'agc'
        # gradient_clip_algorithm = string_parameter(
        #     'gradient_clip_algorithm', 'norm'
        # )
        # # Optimizer algorithm: 'adam', 'adamw', or 'sgd'
        # optimizer = string_parameter('optimizer', 'adam')
        # # SGD momentum (ignored for Adam/AdamW)
        # momentum = real_parameter('momentum', 0.9)
        # # Weight decay (L2 penalty)
        # weight_decay = real_parameter('weight_decay', 0.0)
        # # Adam/AdamW beta1
        # adam_beta1 = real_parameter('adam_beta1', 0.9)
        # # Adam/AdamW beta2
        # adam_beta2 = real_parameter('adam_beta2', 0.999)
        # log_grad_norm = boolean_parameter('log_grad_norm', False)
        # grad_norm_log_freq = integer_parameter('grad_norm_log_freq', 1)
        # # batch_size = integer_parameter('batch_size', 32)
        #
        # # Meta learning: Schedulers etc.
        # # Will be set to total epochs if not specified
        # stage_1_epochs = integer_parameter('stage_1_epochs', 0)
        # # Weighted transition (0 = disabled)
        # stage_2_epochs = integer_parameter('stage_2_epochs', 0)
        # # Physics only (0 = disabled)
        # stage_3_epochs = integer_parameter('stage_3_epochs', 0)
        # # 'linear', 'cosine', 'exponential'
        # physics_weight_schedule = string_parameter(
        #     'physics_weight_schedule', 'cosine'
        # )
        #
        # # Multi-stage learning rate parameters
        # # LR reduction for stage 3 (physics)
        # stage_3_lr_factor = real_parameter('stage_3_lr_factor', 0.1)
        #
        # # Backend-specific loss selection
        # torch_loss_mode = string_parameter('torch_loss_mode', 'poisson')
        #
        # # MLFlow config
        # experiment_name = string_parameter(
        #     'experiment_name', 'Synthetic_Runs'
        # )
        # notes = string_parameter('notes', '')
        # model_name = string_parameter('model_name', 'PtychoPINNv2')
        #
        # # Lightning specific configs
        # output_dir = string_parameter('output_dir', 'lightning_outputs')
        #
        # # Number of grouped samples
        # n_groups = integer_parameter('n_groups', 0)
        # FIXME END

        return builder.build_widget()
