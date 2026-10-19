from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNDataSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNData')
        self._group.add_observer(self)

        self.model_size = self._group.create_integer_parameter('model_size', 128, minimum=1)
        # Square group side length. The channel count is gridsize**2, derived
        # downstream rather than configured.
        self.gridsize = self._group.create_integer_parameter('gridsize', 2, minimum=1)
        self.neighbor_count = self._group.create_integer_parameter('neighbor_count', 6, minimum=1)
        # Sizes the grouped object canvas; no grouping planner reads it.
        self.group_padding_step = self._group.create_real_parameter(
            'group_padding_step', 3.0, minimum=0.0
        )
        self.data_normalization_mode = self._group.create_string_parameter(
            'data_normalization_mode', 'Batch'
        )
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

        # Advanced
        self.probe_scale = self._group.create_real_parameter('probe_scale', 4.0, minimum=0.0)
        self.data_scaling_method = self._group.create_string_parameter(
            'data_scaling_method', 'Parseval'
        )
        # subtract mean phase is only used for supervised training
        self.subtract_mean_phase = self._group.create_boolean_parameter('subtract_mean_phase', True)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNModel')
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

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTraining')
        self._group.add_observer(self)

        self.epochs = self._group.create_integer_parameter('epochs', 50, minimum=1)
        self.batch_size = self._group.create_integer_parameter('batch_size', 16, minimum=1)
        self.learning_rate = self._group.create_real_parameter('learning_rate', 1e-3, minimum=0.0)
        # Number of grouped samples drawn from the dataset. Grouping is
        # centered-nearest and never oversamples, so a value larger than the
        # number of candidate rows is rejected upstream.
        self.training_groups = self._group.create_integer_parameter(
            'training_groups', 512, minimum=1
        )
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

        self.accelerator = self._group.create_string_parameter('accelerator', 'auto')
        self.learning_rate_scheduler = self._group.create_string_parameter(
            'learning_rate_scheduler', 'Default'
        )
        self.learning_rate_warmup_epochs = self._group.create_integer_parameter(
            'learning_rate_warmup_epochs', 0, minimum=0
        )
        self.minimum_learning_rate_ratio = self._group.create_real_parameter(
            'minimum_learning_rate_ratio', 0.1, minimum=0.0
        )
        self.notes = self._group.create_string_parameter('notes', '')
        self.model_name = self._group.create_string_parameter('model_name', 'PtychoPINNv2')

        # Multi-stage training. Zero epochs disables a stage.
        self.stage_1_epochs = self._group.create_integer_parameter('stage_1_epochs', 0, minimum=0)
        self.stage_2_epochs = self._group.create_integer_parameter('stage_2_epochs', 0, minimum=0)
        self.stage_3_epochs = self._group.create_integer_parameter('stage_3_epochs', 0, minimum=0)
        self.stage_3_lr_factor = self._group.create_real_parameter(
            'stage_3_lr_factor', 0.1, minimum=0.0
        )
        self.physics_weight_schedule = self._group.create_string_parameter(
            'physics_weight_schedule', 'cosine'
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNInference')
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
