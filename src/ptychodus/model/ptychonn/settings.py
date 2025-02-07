from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoNNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoNN')
        self._settingsGroup.addObserver(self)

        self.numberOfConvolutionKernels = self._settingsGroup.createIntegerParameter(
            'NumberOfConvolutionKernels', 16
        )
        self.batchSize = self._settingsGroup.createIntegerParameter('BatchSize', 64)
        self.useBatchNormalization = self._settingsGroup.createBooleanParameter(
            'UseBatchNormalization', False
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoNNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoNNTraining')
        self._settingsGroup.addObserver(self)

        self.trainingDataPath = self._settingsGroup.createPathParameter(
            'TrainingDataPath', Path('/path/to/training_data')
        )
        self.validationSetFractionalSize = self._settingsGroup.createRealParameter(
            'ValidationSetFractionalSize', 0.1
        )
        self.maximumLearningRate = self._settingsGroup.createRealParameter(
            'MaximumLearningRate', 1e-3
        )
        self.minimumLearningRate = self._settingsGroup.createRealParameter(
            'MinimumLearningRate', 1e-4
        )
        self.trainingEpochs = self._settingsGroup.createIntegerParameter('TrainingEpochs', 50)
        self.statusIntervalInEpochs = self._settingsGroup.createIntegerParameter(
            'StatusIntervalInEpochs', 1
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
