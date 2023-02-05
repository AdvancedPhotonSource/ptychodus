from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class PtychoNNSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.modelInputSize = settingsGroup.createIntegerEntry('ModelInputSize', 128)
        self.modelOutputSize = settingsGroup.createIntegerEntry('ModelOutputSize', 128)
        self.modelStateFilePath = settingsGroup.createPathEntry('ModelStateFilePath',
                                                                Path('/path/to/best_model.pth'))
        self.numberOfConvolutionChannels = settingsGroup.createIntegerEntry(
            'NumberOfConvolutionChannels', 16)
        self.batchSize = settingsGroup.createIntegerEntry('BatchSize', 10)
        self.useBatchNormalization = settingsGroup.createBooleanEntry(
            'UseBatchNormalization', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoNNSettings:
        settings = cls(settingsRegistry.createGroup('PtychoNN'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoNNTrainingSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        self.validationSetFractionalSize = settingsGroup.createRealEntry(
            'ValidationSetFractionalSize', '0.1')
        self.optimizationEpochsPerHalfCycle = settingsGroup.createIntegerEntry(
            'OptimizationEpochsPerHalfCycle', 6)
        self.maximumLearningRate = settingsGroup.createRealEntry('MaximumLearningRate', '1e-3')
        self.minimumLearningRate = settingsGroup.createRealEntry('MinimumLearningRate', '1e-4')
        self.trainingEpochs = settingsGroup.createIntegerEntry('TrainingEpochs', 1)
        self.saveTrainingArtifacts = settingsGroup.createBooleanEntry(
            'SaveTrainingArtifacts', False)
        self.outputPath = settingsGroup.createPathEntry('OutputPath', Path('/path/to/output'))
        self.outputSuffix = settingsGroup.createStringEntry('OutputSuffix', 'suffix')
        self.statusIntervalInEpochs = settingsGroup.createIntegerEntry('StatusIntervalInEpochs', 1)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoNNTrainingSettings:
        settings = cls(settingsRegistry.createGroup('PtychoNN'))
        settings._settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
