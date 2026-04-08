from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNTorchDataSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchData')
        self._group.add_observer(self)

        # FIXME see ptychopinn settings and reconstructor for handling of diffraction_pattern_size_px, batch_size, nphotons, etc.

        self.num_channels = self._group.create_integer_parameter('num_channels', 4, minimum=1)
        self.data_normalization_mode = self._group.create_string_parameter(
            'data_normalization_mode', 'Batch'
        )
        self.neighbor_lookup_method = self._group.create_string_parameter(
            'neighbor_lookup_method', 'Nearest'
        )
        self.scan_pattern = self._group.create_string_parameter('scan_pattern', 'Isotropic')
        self.normalize_probe = self._group.create_boolean_parameter('normalize_probe', True)

        # Bounding parameters for probe positions
        self.x_lower_bound = self._group.create_real_parameter(
            'x_lower_bound', 0.1, minimum=0.0, maximum=1.0
        )
        self.x_upper_bound = self._group.create_real_parameter(
            'x_upper_bound', 0.9, minimum=0.0, maximum=1.0
        )
        self.y_lower_bound = self._group.create_real_parameter(
            'y_lower_bound', 0.1, minimum=0.0, maximum=1.0
        )
        self.y_upper_bound = self._group.create_real_parameter(
            'y_upper_bound', 0.9, minimum=0.0, maximum=1.0
        )

        self.min_neighbor_distance = self._group.create_real_parameter(
            'min_neighbor_distance', 0.0, minimum=0.0
        )
        self.max_neighbor_distance = self._group.create_real_parameter(
            'max_neighbor_distance', 3.0, minimum=0.0
        )

        self.num_nearest_neighbors_for_quadrant_lookup = self._group.create_integer_parameter(
            'num_nearest_neighbors_for_quadrant_lookup', 30, minimum=0
        )

        # Advanced
        self.num_photons = self._group.create_real_parameter('num_photons', 1e5, minimum=0.0)
        self.coordinate_subsampling_factor = self._group.create_integer_parameter(
            'coordinate_subsampling_factor', 7, minimum=1
        )
        self.probe_scale = self._group.create_real_parameter('probe_scale', 4.0, minimum=0.0)
        self.num_nearest_neighbors_for_lookup = self._group.create_integer_parameter(
            'num_nearest_neighbors_for_lookup', 6, minimum=0
        )
        self.grid_size_x = self._group.create_integer_parameter('grid_size_x', 2, minimum=0)
        self.grid_size_y = self._group.create_integer_parameter('grid_size_y', 2, minimum=0)
        self.probe_ramp_removal = self._group.create_boolean_parameter('probe_ramp_removal', False)
        self.data_scaling_method = self._group.create_string_parameter(
            'data_scaling_method', 'Parseval'
        )
        # subtract mean phase is only used for supervised training
        self.subtract_mean_phase = self._group.create_boolean_parameter('subtract_mean_phase', True)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTorchModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchModel')
        self._group.add_observer(self)

        self.object_big = self._group.create_boolean_parameter('object_big', True)
        self.probe_big = self._group.create_boolean_parameter('probe_big', True)
        self.loss_function = self._group.create_string_parameter('loss_function', 'Poisson')
        self.amplitude_activation_function = self._group.create_string_parameter(
            'amplitude_activation_function', 'silu'
        )
        self.cbam_encoder = self._group.create_boolean_parameter('cbam_encoder', True)
        self.use_shared_decoder = self._group.create_boolean_parameter('use_shared_decoder', False)

        # Advanced
        self.intensity_scale_trainable = self._group.create_boolean_parameter(
            'intensity_scale_trainable', False
        )
        self.intensity_scale = self._group.create_real_parameter(
            'intensity_scale', 10000.0, minimum=0.0
        )
        self.max_position_jitter = self._group.create_integer_parameter(
            'max_position_jitter', 10, minimum=0
        )
        self.num_datasets = self._group.create_integer_parameter('num_datasets', 1, minimum=1)
        self.auxiliary_amplitude_loss = self._group.create_string_parameter(
            'auxiliary_amplitude_loss', 'None'
        )
        self.auxiliary_phase_loss = self._group.create_string_parameter(
            'auxiliary_phase_loss', 'None'
        )
        self.auxiliary_amplitude_loss_coeff = self._group.create_real_parameter(
            'auxiliary_amplitude_loss_coeff', 1.0, minimum=0.0
        )
        self.auxiliary_phase_loss_coeff = self._group.create_real_parameter(
            'auxiliary_phase_loss_coeff', 1.0, minimum=0.0
        )
        self.num_filters_scale = self._group.create_integer_parameter(
            'num_filters_scale', 2, minimum=1
        )
        self.eca_decoder = self._group.create_boolean_parameter('eca_decoder', False)
        self.use_batch_normalization = self._group.create_boolean_parameter(
            'use_batch_normalization', False
        )
        self.edge_pad = self._group.create_integer_parameter('edge_pad', 10, minimum=0)
        self.decoder_last_c_outer_fraction = self._group.create_real_parameter(
            'decoder_last_c_outer_fraction', 0.125, minimum=0.0, maximum=1.0
        )
        self.cbam_bottleneck = self._group.create_boolean_parameter('cbam_bottleneck', False)
        self.cbam_decoder = self._group.create_boolean_parameter('cbam_decoder', False)
        self.spatial_decoder = self._group.create_boolean_parameter('spatial_decoder', False)
        self.decoder_spatial_kernel = self._group.create_integer_parameter(
            'decoder_spatial_kernel', 7, minimum=1
        )
        self.eca_encoder = self._group.create_boolean_parameter('eca_encoder', False)
        self.offset = self._group.create_integer_parameter('offset', 6, minimum=0)
        self.pad_object = self._group.create_boolean_parameter('pad_object', True)
        self.probe_gaussian_smoothing_sigma = self._group.create_real_parameter(
            'probe_gaussian_smoothing_sigma', 0.0
        )
        self.probe_reference_loss_coeff = self._group.create_real_parameter(
            'probe_reference_loss_coeff', 0.0, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTorchTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchTraining')
        self._group.add_observer(self)

        self.epochs = self._group.create_integer_parameter('epochs', 50, minimum=1)
        self.batch_size = self._group.create_integer_parameter('batch_size', 16, minimum=1)
        self.learning_rate = self._group.create_real_parameter('learning_rate', 1e-3, minimum=0.0)
        self.num_devices = self._group.create_integer_parameter('num_devices', 1, minimum=0)
        self.num_dataloader_workers = self._group.create_integer_parameter(
            'num_dataloader_workers', 4, minimum=0
        )
        self.gradient_accumulation_steps = self._group.create_integer_parameter(
            'gradient_accumulation_steps', 1, minimum=1
        )
        self.epochs_finetune = self._group.create_integer_parameter('epochs_finetune', 0, minimum=0)
        self.finetune_gamma = self._group.create_real_parameter(
            'finetune_gamma', 0.1, minimum=0.0, maximum=1.0
        )
        self.gradient_clip_val = self._group.create_real_parameter(
            'gradient_clip_val', 0.0, minimum=0.0
        )
        self.experiment_name = self._group.create_string_parameter(
            'experiment_name', 'Synthetic_Runs'
        )

        self.use_negative_log_likelihood_loss = self._group.create_boolean_parameter(
            'use_negative_log_likelihood_loss', True
        )
        self.device = self._group.create_string_parameter('device', 'cuda')
        self.learning_rate_scheduler = self._group.create_string_parameter(
            'learning_rate_scheduler', 'Default'
        )
        self.learning_rate_warmup_epochs = self._group.create_integer_parameter(
            'learning_rate_warmup_epochs', 0, minimum=0
        )
        self.minimum_learning_rate_ratio = self._group.create_real_parameter(
            'minimum_learning_rate_ratio', 0.1, minimum=0.0
        )
        self.physics_weight_schedule = self._group.create_string_parameter(
            'physics_weight_schedule', 'cosine'
        )
        self.torch_loss_mode = self._group.create_string_parameter('torch_loss_mode', 'poisson')
        self.notes = self._group.create_string_parameter('notes', '')
        self.model_name = self._group.create_string_parameter('model_name', 'PtychoPINNv2')

        self.enable_staged_finetuning = self._group.create_boolean_parameter(
            'enable_staged_finetuning', False
        )
        self.finetune_stage1_epochs = self._group.create_integer_parameter(
            'finetune_stage1_epochs', 7, minimum=1
        )
        self.finetune_stage2_epochs = self._group.create_integer_parameter(
            'finetune_stage2_epochs', 7, minimum=1
        )
        self.finetune_stage3_epochs = self._group.create_integer_parameter(
            'finetune_stage3_epochs', 5, minimum=1
        )
        self.finetune_stage1_lr_decoder = self._group.create_real_parameter(
            'finetune_stage1_lr_decoder', 0.1, minimum=0.0
        )
        self.finetune_stage2_lr_encoder_top = self._group.create_real_parameter(
            'finetune_stage2_lr_encoder_top', 0.01, minimum=0.0
        )
        self.finetune_stage2_lr_decoder = self._group.create_real_parameter(
            'finetune_stage2_lr_decoder', 0.05, minimum=0.0
        )
        self.finetune_stage2_lr_phase_head = self._group.create_real_parameter(
            'finetune_stage2_lr_phase_head', 0.1, minimum=0.0
        )
        self.finetune_stage3_lr_encoder_bottom = self._group.create_real_parameter(
            'finetune_stage3_lr_encoder_bottom', 0.005, minimum=0.0
        )
        self.finetune_stage3_lr_encoder_top = self._group.create_real_parameter(
            'finetune_stage3_lr_encoder_top', 0.01, minimum=0.0
        )
        self.finetune_stage3_lr_decoder = self._group.create_real_parameter(
            'finetune_stage3_lr_decoder', 0.02, minimum=0.0
        )
        self.finetune_stage3_lr_phase_head = self._group.create_real_parameter(
            'finetune_stage3_lr_phase_head', 0.05, minimum=0.0
        )
        self.finetune_skip_stage3 = self._group.create_boolean_parameter(
            'finetune_skip_stage3', True
        )
        self.finetune_early_stop_patience = self._group.create_integer_parameter(
            'finetune_early_stop_patience', 3, minimum=1
        )
        self.finetune_validation_split = self._group.create_real_parameter(
            'finetune_val_split', 0.05, minimum=0.0, maximum=1.0
        )  # 5% validation split

        self.num_grouped_samples = self._group.create_integer_parameter('num_grouped_samples', 0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTorchInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTorchInference')
        self._group.add_observer(self)

        self.batch_size = self._group.create_integer_parameter('batch_size', 1000, minimum=1)
        self.middle_trim = self._group.create_integer_parameter('middle_trim', 32, minimum=0)
        self.experiment_number = self._group.create_integer_parameter(
            'experiment_number', 0, minimum=0
        )
        self.patch_weighting_method = self._group.create_string_parameter(
            'patch_weighting_method', 'probe'
        )

        # Advanced
        self.pad_eval = self._group.create_boolean_parameter('pad_eval', True)
        self.window = self._group.create_integer_parameter('window', 20, minimum=0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
