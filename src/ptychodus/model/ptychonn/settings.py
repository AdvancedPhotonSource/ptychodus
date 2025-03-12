from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoNNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtychoNN')
        self._settingsGroup.add_observer(self)

        self.numberOfConvolutionKernels = self._settingsGroup.create_integer_parameter(
            'NumberOfConvolutionKernels', 16
        )
        self.batchSize = self._settingsGroup.create_integer_parameter('BatchSize', 64)
        self.useBatchNormalization = self._settingsGroup.create_boolean_parameter(
            'UseBatchNormalization', False
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class PtychoNNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('PtychoNNTraining')
        self._settingsGroup.add_observer(self)

        self.trainingDataPath = self._settingsGroup.create_path_parameter(
            'TrainingDataPath', Path('/path/to/training_data')
        )
        self.validationSetFractionalSize = self._settingsGroup.create_real_parameter(
            'ValidationSetFractionalSize', 0.1
        )
        self.maximumLearningRate = self._settingsGroup.create_real_parameter(
            'MaximumLearningRate', 1e-3
        )
        self.minimumLearningRate = self._settingsGroup.create_real_parameter(
            'MinimumLearningRate', 1e-4
        )
        self.trainingEpochs = self._settingsGroup.create_integer_parameter('TrainingEpochs', 50)
        self.statusIntervalInEpochs = self._settingsGroup.create_integer_parameter(
            'StatusIntervalInEpochs', 1
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()
