from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNModel')
        self._group.add_observer(self)

        self.gridsize = self._group.create_integer_parameter('gridsize', 1, minimum=1, maximum=5)
        self.n_filters_scale = self._group.create_integer_parameter(
            'n_filters_scale', 2, minimum=1, maximum=4
        )
        self.amp_activation = self._group.create_string_parameter('amp_activation', 'sigmoid')
        self.object_big = self._group.create_boolean_parameter('object_big', True)
        self.probe_big = self._group.create_boolean_parameter('probe_big', True)
        self.probe_mask = self._group.create_boolean_parameter('probe_mask', False)
        self.pad_object = self._group.create_boolean_parameter('pad_object', True)
        self.probe_scale = self._group.create_real_parameter('probe_scale', 4.0, minimum=0.0)
        self.gaussian_smoothing_sigma = self._group.create_real_parameter(
            'gaussian_smoothing_sigma', 0.0, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTraining')
        self._group.add_observer(self)

        self.nphotons = self._group.create_real_parameter('NPhotons', 1e6)  # TODO remove
        self.data_dir = self._group.create_path_parameter(
            'data_dir', Path('/path/to/training_data')
        )
        self.batch_size = self._group.create_integer_parameter(
            'batch_size', 16, minimum=1, maximum=1 << 30
        )  # must be positive powers of two
        self.nepochs = self._group.create_integer_parameter('nepochs', 50, minimum=1)
        self.mae_weight = self._group.create_real_parameter(
            'mae_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.nll_weight = self._group.create_real_parameter(
            'nll_weight', 1.0, minimum=0.0, maximum=1.0
        )
        self.realspace_mae_weight = self._group.create_real_parameter(
            'realspace_mae_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.realspace_weight = self._group.create_real_parameter(
            'realspace_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.positions_provided = self._group.create_boolean_parameter('positions_provided', True)
        self.probe_trainable = self._group.create_boolean_parameter('probe_trainable', False)
        self.intensity_scale_trainable = self._group.create_boolean_parameter(
            'intensity_scale_trainable', True
        )
        self.output_dir = self._group.create_path_parameter(
            'output_dir', Path('/path/to/output_data')
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNInference')
        self._group.add_observer(self)

        self.model_path = self._group.create_path_parameter(
            'model_path', Path('/path/to/model.zip')
        )
        self.n_nearest_neighbors = self._group.create_integer_parameter(
            'n_nearest_neighbors', 7, minimum=0
        )
        self.n_samples = self._group.create_integer_parameter('n_samples', 1, minimum=1)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
