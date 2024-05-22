from __future__ import annotations
from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry, SettingsGroup


class PtychoNNModelSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.stateFilePath = settingsGroup.createPathEntry('StateFilePath',
                                                           Path('/path/to/best_model.pth'))
        self.numberOfConvolutionKernels = settingsGroup.createIntegerEntry(
            'NumberOfConvolutionKernels', 16)
        self.batchSize = settingsGroup.createIntegerEntry('BatchSize', 64)
        self.useBatchNormalization = settingsGroup.createBooleanEntry(
            'UseBatchNormalization', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoNNModelSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoNN')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoNNTrainingSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.maximumTrainingDatasetSize = settingsGroup.createIntegerEntry(
            'MaximumTrainingDatasetSize', 100000)
        self.validationSetFractionalSize = settingsGroup.createRealEntry(
            'ValidationSetFractionalSize', '0.1')
        self.optimizationEpochsPerHalfCycle = settingsGroup.createIntegerEntry(
            'OptimizationEpochsPerHalfCycle', 6)
        self.maximumLearningRate = settingsGroup.createRealEntry('MaximumLearningRate', '1e-3')
        self.minimumLearningRate = settingsGroup.createRealEntry('MinimumLearningRate', '1e-4')
        self.trainingEpochs = settingsGroup.createIntegerEntry('TrainingEpochs', 50)
        self.saveTrainingArtifacts = settingsGroup.createBooleanEntry(
            'SaveTrainingArtifacts', False)
        self.outputPath = settingsGroup.createPathEntry('OutputPath', Path('/path/to/output'))
        self.outputSuffix = settingsGroup.createStringEntry('OutputSuffix', 'suffix')
        self.statusIntervalInEpochs = settingsGroup.createIntegerEntry('StatusIntervalInEpochs', 1)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoNNTrainingSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoNNTraining')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
