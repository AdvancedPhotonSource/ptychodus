from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoNNModelSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoNN')
        self._settingsGroup.addObserver(self)

        self.modelFilePath = self._settingsGroup.createPathEntry('ModelFilePath',
                                                                 Path('/path/to/best_model.pth'))
        self.numberOfConvolutionKernels = self._settingsGroup.createIntegerEntry(
            'NumberOfConvolutionKernels', 16)
        self.batchSize = self._settingsGroup.createIntegerEntry('BatchSize', 64)
        self.useBatchNormalization = self._settingsGroup.createBooleanEntry(
            'UseBatchNormalization', False)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoNNTrainingSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoNNTraining')
        self._settingsGroup.addObserver(self)

        self.maximumTrainingDatasetSize = self._settingsGroup.createIntegerEntry(
            'MaximumTrainingDatasetSize', 100000)
        self.validationSetFractionalSize = self._settingsGroup.createRealEntry(
            'ValidationSetFractionalSize', '0.1')
        self.maximumLearningRate = self._settingsGroup.createRealEntry(
            'MaximumLearningRate', '1e-3')
        self.minimumLearningRate = self._settingsGroup.createRealEntry(
            'MinimumLearningRate', '1e-4')
        self.trainingEpochs = self._settingsGroup.createIntegerEntry('TrainingEpochs', 50)
        self.saveTrainingArtifacts = self._settingsGroup.createBooleanEntry(
            'SaveTrainingArtifacts', False)
        self.trainingArtifactsDirectory = self._settingsGroup.createPathEntry(
            'TrainingArtifactsDirectory', Path('/path/to/artifacts'))
        self.trainingArtifactsSuffix = self._settingsGroup.createStringEntry(
            'TrainingArtifactsSuffix', 'suffix.pth')
        self.statusIntervalInEpochs = self._settingsGroup.createIntegerEntry(
            'StatusIntervalInEpochs', 1)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
