from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoNNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.create_group('PtychoNN')
        self._settings_group.add_observer(self)

        self.num_convolution_kernels = self._settings_group.create_integer_parameter(
            'NumberOfConvolutionKernels', 16, minimum=1
        )
        self.batch_size = self._settings_group.create_integer_parameter('BatchSize', 64, minimum=1)
        self.use_batch_normalization = self._settings_group.create_boolean_parameter(
            'UseBatchNormalization', False
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notify_observers()


class PtychoNNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.create_group('PtychoNNTraining')
        self._settings_group.add_observer(self)

        self.training_data_path = self._settings_group.create_path_parameter(
            'TrainingDataPath', Path('/path/to/training_data')
        )
        self.validation_set_fractional_size = self._settings_group.create_real_parameter(
            'ValidationSetFractionalSize', 0.1, minimum=0.0, maximum=1.0
        )
        self.max_learning_rate = self._settings_group.create_real_parameter(
            'MaximumLearningRate', 1e-3, minimum=0.0, maximum=1.0
        )
        self.min_learning_rate = self._settings_group.create_real_parameter(
            'MinimumLearningRate', 1e-4, minimum=0.0, maximum=1.0
        )
        self.training_epochs = self._settings_group.create_integer_parameter(
            'TrainingEpochs', 50, minimum=1
        )
        self.status_interval_in_epochs = self._settings_group.create_integer_parameter(
            'StatusIntervalInEpochs', 1, minimum=1
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notify_observers()
