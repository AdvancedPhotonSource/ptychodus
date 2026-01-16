from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class SynapsIInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('SynapsIInference')
        self._group.add_observer(self)

        self.model_path = self._group.create_path_parameter(
            'ModelPath', Path('/path/to/best_model.pth')
        )
        self.config_path = self._group.create_path_parameter(
            'ConfigPath', Path('/path/to/config.yaml')
        )
        self.batch_size = self._group.create_integer_parameter('BatchSize', 16, minimum=1)
        self.use_cuda = self._group.create_boolean_parameter('UseCUDA', True)
        self.max_probe_modes = self._group.create_integer_parameter(
            'MaxProbeModes', 8, minimum=1
        )
        self.specify_normalization = self._group.create_boolean_parameter(
            'SpecifyNormalization', False
        )
        self.normalization = self._group.create_real_parameter(
            'Normalization', 100000.0, minimum=0.0
        )
        self.scale = self._group.create_real_parameter('Scale', 10000.0, minimum=0.0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
