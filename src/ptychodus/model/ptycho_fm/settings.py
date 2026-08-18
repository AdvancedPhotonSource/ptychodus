from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoFMDataSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoFMData')
        self._group.add_observer(self)

        self.scale = self._group.create_real_parameter('scale', 10000.0, minimum=0.0)
        self.default_normalization = self._group.create_real_parameter(
            'default_normalization', 100000.0, minimum=0.0
        )
        self.cache_object = self._group.create_boolean_parameter('cache_object', True)
        self.max_probe_modes = self._group.create_integer_parameter(
            'max_probe_modes', 10, minimum=1
        )
        self.target_size = self._group.create_integer_parameter('target_size', 256, minimum=32)
        self.train_split = self._group.create_real_parameter(
            'train_split', 0.80, minimum=0.0, maximum=1.0
        )
        self.random_seed = self._group.create_integer_parameter('random_seed', 8)
        # 0 → treat as "use all files" (null in the YAML).
        self.max_files = self._group.create_integer_parameter('max_files', 0, minimum=0)
        self.num_workers = self._group.create_integer_parameter('num_workers', 4, minimum=0)
        self.prefetch_factor = self._group.create_integer_parameter('prefetch_factor', 2, minimum=0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoFMModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoFMModel')
        self._group.add_observer(self)

        self.encoder_type = self._group.create_string_parameter('encoder_type', 'custom')
        self.img_size = self._group.create_integer_parameter('img_size', 256, minimum=32)
        self.patch_size = self._group.create_integer_parameter('patch_size', 16, minimum=1)
        self.embed_dim = self._group.create_integer_parameter('embed_dim', 512, minimum=1)
        self.depth = self._group.create_integer_parameter('depth', 12, minimum=1)
        self.num_heads = self._group.create_integer_parameter('num_heads', 8, minimum=1)
        self.mlp_ratio = self._group.create_real_parameter('mlp_ratio', 4.0, minimum=0.0)
        self.use_cls_token = self._group.create_boolean_parameter('use_cls_token', False)
        self.dropout = self._group.create_real_parameter('dropout', 0.1, minimum=0.0, maximum=1.0)
        self.attn_dropout = self._group.create_real_parameter(
            'attn_dropout', 0.0, minimum=0.0, maximum=1.0
        )
        self.timm_model_name = self._group.create_string_parameter(
            'timm_model_name', 'vit_large_patch32_224'
        )

        self.decoder_base_channels = self._group.create_integer_parameter(
            'decoder_base_channels', 64, minimum=1
        )
        self.decoder_latent_dim = self._group.create_integer_parameter(
            'decoder_latent_dim', 512, minimum=1
        )
        self.decoder_num_stages = self._group.create_integer_parameter(
            'decoder_num_stages', 4, minimum=1
        )
        self.decoder_use_batchnorm = self._group.create_boolean_parameter(
            'decoder_use_batchnorm', True
        )
        self.decoder_dropout = self._group.create_real_parameter(
            'decoder_dropout', 0.1, minimum=0.0, maximum=1.0
        )

        self.init_enabled = self._group.create_boolean_parameter('init_enabled', False)
        self.init_method = self._group.create_string_parameter('init_method', 'trunc_normal')

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoFMTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoFMTraining')
        self._group.add_observer(self)

        self.batch_size = self._group.create_integer_parameter('batch_size', 64, minimum=1)
        self.learning_rate = self._group.create_real_parameter('learning_rate', 1.0e-5, minimum=0.0)
        self.epochs = self._group.create_integer_parameter('epochs', 11, minimum=1)
        self.loss_function = self._group.create_string_parameter('loss_function', 'weighted')
        self.weighted_loss_type = self._group.create_string_parameter('weighted_loss_type', 'mse')
        self.weighted_loss_threshold = self._group.create_real_parameter(
            'weighted_loss_threshold', 0.0
        )
        self.weighted_loss_alpha = self._group.create_real_parameter('weighted_loss_alpha', 1.0)
        self.save_epoch_models = self._group.create_boolean_parameter('save_epoch_models', True)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoFMInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoFMInference')
        self._group.add_observer(self)

        self.central_crop = self._group.create_integer_parameter('central_crop', 64, minimum=1)
        self.pad = self._group.create_integer_parameter('pad', 32, minimum=0)
        self.batch_size = self._group.create_integer_parameter('batch_size', 256, minimum=1)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
