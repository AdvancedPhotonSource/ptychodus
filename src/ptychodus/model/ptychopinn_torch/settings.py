from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNTorchDataSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchData')
        self._group.add_observer(self)

        self.nphotons = self._group.create_real_parameter('nphotons', 1e5, minimum=0)
        # General sizing parameters
        # Size of the diffraction patterns/object patch
        self.N = self._group.create_integer_parameter('N', 64, minimum=0)
        # Number of channels
        self.C = self._group.create_integer_parameter('C', 4, minimum=0)
        # Number of nearest neighbors for lookup
        self.K = self._group.create_integer_parameter('K', 6, minimum=0)
        # Grid parameters specifically for overlap constraint
        # Number of nearest neighbors for quadrant lookup
        self.K_quadrant = self._group.create_integer_parameter('K_quadrant', 30, minimum=0)
        # Subsampling factor for coordinates (if applicable)
        self.n_subsample = self._group.create_integer_parameter('n_subsample', 7, minimum=0)
        # Grid size for scanning positions
        self.grid_size_x = self._group.create_integer_parameter('grid_size_x', 2, minimum=0)
        self.grid_size_y = self._group.create_integer_parameter('grid_size_y', 2, minimum=0)
        self.neighbor_function = self._group.create_string_parameter('neighbor_function', 'Nearest')
        self.min_neighbor_distance = self._group.create_real_parameter(
            'min_neighbor_distance', 0.0, minimum=0.0
        )
        self.max_neighbor_distance = self._group.create_real_parameter(
            'max_neighbor_distance', 3.0, minimum=0.0
        )
        # Scan pattern, used for 4_quadrant neighbor function
        self.scan_pattern = self._group.create_string_parameter('scan_pattern', 'Isotropic')

        # Miscellaneous
        # Whether to normalize the data
        self.normalize = self._group.create_string_parameter('normalize', 'Batch')
        self.probe_scale = self._group.create_real_parameter('probe_scale', 4.0)
        self.probe_normalize = self._group.create_boolean_parameter('probe_normalize', True)
        self.probe_ramp_removal = self._group.create_boolean_parameter('probe_ramp_removal', False)
        self.data_scaling = self._group.create_string_parameter('data_scaling', 'Parseval')
        # Only useful for supervised training dataset
        self.phase_subtraction = self._group.create_boolean_parameter('phase_subtraction', True)

        # Bounding parameters for scan positions
        self.x_lower_bound = self._group.create_real_parameter('x_lower_bound', 0.1)
        self.x_upper_bound = self._group.create_real_parameter('x_upper_bound', 0.9)
        self.y_lower_bound = self._group.create_real_parameter('y_lower_bound', 0.1)
        self.y_upper_bound = self._group.create_real_parameter('y_upper_bound', 0.9)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTorchModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchModel')
        self._group.add_observer(self)

        # Mode Category
        # Training mode, affects all aspects of model
        self.mode = self._group.create_string_parameter('mode', 'Unsupervised')
        # Generator architecture selection
        self.architecture = self._group.create_string_parameter('architecture', 'cnn')
        self.fno_modes = self._group.create_integer_parameter('fno_modes', 12)
        self.fno_width = self._group.create_integer_parameter('fno_width', 32)
        self.fno_blocks = self._group.create_integer_parameter('fno_blocks', 4)
        self.fno_cnn_blocks = self._group.create_integer_parameter('fno_cnn_blocks', 2)
        self.fno_input_transform = self._group.create_string_parameter(
            'fno_input_transform', 'none'
        )
        self.max_hidden_channels = self._group.create_integer_parameter('max_hidden_channels', 0)
        self.resnet_width = self._group.create_integer_parameter('resnet_width', 0)
        self.generator_output_mode = self._group.create_string_parameter(
            'generator_output_mode', 'real_imag'
        )

        # Intensity Parameters
        self.intensity_scale_trainable = self._group.create_boolean_parameter(
            'intensity_scale_trainable', False
        )
        # General intensity scale guess
        self.intensity_scale = self._group.create_real_parameter('intensity_scale', 10000.0)
        # Deprecated in Torch path (padded size ignores this)
        self.max_position_jitter = self._group.create_integer_parameter('max_position_jitter', 10)
        # Number of unique datasets being trained. For instantiating fitting constants
        self.num_datasets = self._group.create_integer_parameter('num_datasets', 1)

        # Model architecture parameters
        self.C_model = self._group.create_integer_parameter('C_model', 4)
        # Shrinking factor for channels in network layers
        self.n_filters_scale = self._group.create_integer_parameter('n_filters_scale', 2)
        # Activation function for amplitude part
        self.amp_activation = self._group.create_string_parameter('amp_activation', 'silu')
        # Whether to use batch normalization
        self.batch_norm = self._group.create_boolean_parameter('batch_norm', False)

        # Module-specific
        # For padding the decoder_last reconstruction
        self.edge_pad = self._group.create_integer_parameter('edge_pad', 10)
        # Amount of channels going to higher frequency components in decoder_last
        self.decoder_last_c_outer_fraction = self._group.create_real_parameter(
            'decoder_last_c_outer_fraction', 0.125
        )
        self.decoder_last_amp_channels = self._group.create_integer_parameter(
            'decoder_last_amp_channels', 1
        )
        self.use_shared_decoder = self._group.create_boolean_parameter('use_shared_decoder', False)

        # Attention
        self.eca_encoder = self._group.create_boolean_parameter('eca_encoder', False)
        # Whether CBAM module is turned on for encoder
        self.cbam_encoder = self._group.create_boolean_parameter('cbam_encoder', True)
        # CBAM bottleneck
        self.cbam_bottleneck = self._group.create_boolean_parameter('cbam_bottleneck', False)
        # CBAM for decoder
        self.cbam_decoder = self._group.create_boolean_parameter('cbam_decoder', False)
        # ECA for decoder
        self.eca_decoder = self._group.create_boolean_parameter('eca_decoder', False)
        # Spatial attention for decoder
        self.spatial_decoder = self._group.create_boolean_parameter('spatial_decoder', False)
        # Spatial attention kernel for decoder
        self.decoder_spatial_kernel = self._group.create_integer_parameter(
            'decoder_spatial_kernel', 7
        )

        # Forward model parameters
        # True if object requires patch reassembly
        self.object_big = self._group.create_boolean_parameter('object_big', True)
        # True if probe requires patch reassembly
        self.probe_big = self._group.create_boolean_parameter('probe_big', True)
        # Offset parameter (for nearest neighbor patches)
        self.offset = self._group.create_integer_parameter('offset', 6)
        # Number of channels
        self.C_forward = self._group.create_integer_parameter('C_forward', 4)

        # Spec-mandated defaults (align with TensorFlow backend)
        # Pad object during forward model
        self.pad_object = self._group.create_boolean_parameter('pad_object', True)
        # Gaussian smoothing sigma for probe
        self.gaussian_smoothing_sigma = self._group.create_real_parameter(
            'gaussian_smoothing_sigma', 0.0
        )

        # Loss
        # Loss function to use ('MAE', 'MSE', etc.)
        self.loss_function = self._group.create_string_parameter('loss_function', 'Poisson')
        self.amp_loss = self._group.create_string_parameter('amp_loss', 'None')
        self.phase_loss = self._group.create_string_parameter('phase_loss', 'None')
        self.amp_loss_coeff = self._group.create_real_parameter('amp_loss_coeff', 1.0)
        self.phase_loss_coeff = self._group.create_real_parameter('phase_loss_coeff', 1.0)
        # Probe reference loss coefficient (0.0 = disabled)
        self.probe_reference_coeff = self._group.create_real_parameter(
            'probe_reference_coeff', 0.0, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTorchTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchTraining')
        self._group.add_observer(self)

        # Device/Loss
        # Use Negative Log Likelihood loss component
        self.nll = self._group.create_boolean_parameter('nll', True)
        # Device to train on ('cuda', 'cpu')
        self.device = self._group.create_string_parameter('device', 'cuda')
        # Strategy for distributed training (e.g., 'ddp', None)
        self.strategy = self._group.create_string_parameter('strategy', 'ddp')
        # Number of devices you're training on
        self.n_devices = self._group.create_integer_parameter('n_devices', 1)

        # Framework
        # Training framework. Most of work don in PT was done in lightning
        self.framework = self._group.create_string_parameter('framework', 'Lightning')
        self.orchestrator = self._group.create_string_parameter('orchestrator', 'Mlflow')

        # Add other training-specific parameters here as needed
        self.learning_rate = self._group.create_real_parameter('learning_rate', 1e-3)
        # Default epochs number, will be overridden if multi-stage training is active at all
        self.epochs = self._group.create_integer_parameter('epochs', 50)
        self.batch_size = self._group.create_integer_parameter('batch_size', 16)
        # Default 0 fine-tune means no fine-tuning
        self.epochs_fine_tune = self._group.create_integer_parameter('epochs_fine_tune', 0)
        # Scales base LR for fine-tuning
        self.fine_tune_gamma = self._group.create_real_parameter('fine_tune_gamma', 0.1)
        self.scheduler = self._group.create_string_parameter('scheduler', 'Default')
        # Number of warmup epochs for WarmupCosine scheduler
        self.lr_warmup_epochs = self._group.create_integer_parameter('lr_warmup_epochs', 0)
        # Minimum LR ratio for WarmupCosine scheduler (eta_min = base_lr * ratio)
        self.min_lr_ratio = self._group.create_real_parameter('min_lr_ratio', 0.1)
        self.plateau_factor = self._group.create_real_parameter('plateau_factor', 0.5)
        self.plateau_patience = self._group.create_integer_parameter('plateau_patience', 2)
        self.plateau_min_lr = self._group.create_real_parameter('plateau_min_lr', 1e-4)
        self.plateau_threshold = self._group.create_real_parameter('plateau_threshold', 0.0)
        # Dataloader workers
        self.num_workers = self._group.create_integer_parameter('num_workers', 4)
        # Batch size accumulation, manually implemented for DDP
        self.accum_steps = self._group.create_integer_parameter('accum_steps', 1)
        self.gradient_clip_val = self._group.create_real_parameter('gradient_clip_val', 0.0)
        # Gradient clipping algorithm: 'norm', 'value', or 'agc'
        self.gradient_clip_algorithm = self._group.create_string_parameter(
            'gradient_clip_algorithm', 'norm'
        )
        # Optimizer algorithm: 'adam', 'adamw', or 'sgd'
        self.optimizer = self._group.create_string_parameter('optimizer', 'adam')
        # SGD momentum (ignored for Adam/AdamW)
        self.momentum = self._group.create_real_parameter('momentum', 0.9)
        # Weight decay (L2 penalty)
        self.weight_decay = self._group.create_real_parameter('weight_decay', 0.0)
        # Adam/AdamW beta1
        self.adam_beta1 = self._group.create_real_parameter('adam_beta1', 0.9)
        # Adam/AdamW beta2
        self.adam_beta2 = self._group.create_real_parameter('adam_beta2', 0.999)
        self.log_grad_norm = self._group.create_boolean_parameter('log_grad_norm', False)
        self.grad_norm_log_freq = self._group.create_integer_parameter('grad_norm_log_freq', 1)
        # self.batch_size = self._group.create_integer_parameter('batch_size', 32)

        # Meta learning: Schedulers etc.
        # Will be set to total epochs if not specified
        self.stage_1_epochs = self._group.create_integer_parameter('stage_1_epochs', 0)
        # Weighted transition (0 = disabled)
        self.stage_2_epochs = self._group.create_integer_parameter('stage_2_epochs', 0)
        # Physics only (0 = disabled)
        self.stage_3_epochs = self._group.create_integer_parameter('stage_3_epochs', 0)
        # 'linear', 'cosine', 'exponential'
        self.physics_weight_schedule = self._group.create_string_parameter(
            'physics_weight_schedule', 'cosine'
        )

        # Multi-stage learning rate parameters
        # LR reduction for stage 3 (physics)
        self.stage_3_lr_factor = self._group.create_real_parameter('stage_3_lr_factor', 0.1)

        # Backend-specific loss selection
        self.torch_loss_mode = self._group.create_string_parameter('torch_loss_mode', 'poisson')

        # MLFlow config
        self.experiment_name = self._group.create_string_parameter(
            'experiment_name', 'Synthetic_Runs'
        )
        self.notes = self._group.create_string_parameter('notes', '')
        self.model_name = self._group.create_string_parameter('model_name', 'PtychoPINNv2')

        # Beta configs... fine tuning on experiments
        # Fine-tuning configuration
        self.enable_staged_finetuning = self._group.create_boolean_parameter(
            'enable_staged_finetuning', False
        )  # Master switch for synthetic→experimental transfer

        # Stage durations (epochs)
        self.finetune_stage1_epochs = self._group.create_integer_parameter(
            'finetune_stage1_epochs', 7
        )  # Decoder-only
        self.finetune_stage2_epochs = self._group.create_integer_parameter(
            'finetune_stage2_epochs', 7
        )  # Partial encoder + decoder
        self.finetune_stage3_epochs = self._group.create_integer_parameter(
            'finetune_stage3_epochs', 5
        )  # Full network (optional)

        # Learning rate multipliers (relative to base_lr)
        # Stage 1: Decoder-only
        self.finetune_stage1_lr_decoder = self._group.create_real_parameter(
            'finetune_stage1_lr_decoder', 0.1
        )

        # Stage 2: Partial encoder + decoder with discriminative LR
        self.finetune_stage2_lr_encoder_top = self._group.create_real_parameter(
            'finetune_stage2_lr_encoder_top', 0.01
        )
        self.finetune_stage2_lr_decoder = self._group.create_real_parameter(
            'finetune_stage2_lr_decoder', 0.05
        )
        self.finetune_stage2_lr_phase_head = self._group.create_real_parameter(
            'finetune_stage2_lr_phase_head', 0.1
        )

        # Stage 3: Full network with very conservative LR
        self.finetune_stage3_lr_encoder_bottom = self._group.create_real_parameter(
            'finetune_stage3_lr_encoder_bottom', 0.005
        )
        self.finetune_stage3_lr_encoder_top = self._group.create_real_parameter(
            'finetune_stage3_lr_encoder_top', 0.01
        )
        self.finetune_stage3_lr_decoder = self._group.create_real_parameter(
            'finetune_stage3_lr_decoder', 0.02
        )
        self.finetune_stage3_lr_phase_head = self._group.create_real_parameter(
            'finetune_stage3_lr_phase_head', 0.05
        )

        # Stage control
        self.finetune_skip_stage3 = self._group.create_boolean_parameter(
            'finetune_skip_stage3', True
        )  # Most cases won't need Stage 3
        self.finetune_early_stop_patience = self._group.create_integer_parameter(
            'finetune_early_stop_patience', 3
        )  # Per-stage early stopping

        # Validation split for fine-tuning (small experimental dataset)
        self.finetune_val_split = self._group.create_real_parameter(
            'finetune_val_split', 0.05
        )  # 5% validation split

        # Lightning specific configs
        self.output_dir = self._group.create_string_parameter('output_dir', 'lightning_outputs')

        # Number of grouped samples
        self.n_groups = self._group.create_integer_parameter('n_groups', 0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTorchInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchInference')
        self._group.add_observer(self)

        self.middle_trim = self._group.create_integer_parameter('middle_trim', 32)
        # Batch size for reconstruction. Lower this due to GPU memory bandwidth
        self.batch_size = self._group.create_integer_parameter('batch_size', 1000)
        # Experiment number for inference
        self.experiment_number = self._group.create_integer_parameter('experiment_number', 0)
        # Pads the evaluation edges, enforced during training for Nyquist frequency. Can turn off for eval
        self.pad_eval = self._group.create_boolean_parameter('pad_eval', True)
        # Window padding around reconstruction due to edge errors
        self.window = self._group.create_integer_parameter('window', 20)
        self.patch_weighting = self._group.create_string_parameter('patch_weighting', 'probe')
        # Emit patch stats during training/inference
        self.log_patch_stats = self._group.create_boolean_parameter('log_patch_stats', False)
        # Max number of batches to log
        self.patch_stats_limit = self._group.create_integer_parameter('patch_stats_limit', 0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
