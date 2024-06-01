from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNModelSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoPINN')
        self._settingsGroup.addObserver(self)

        self.learningRate = self._settingsGroup.createRealEntry('LearningRate', '1e-3')
        self.N = self._settingsGroup.createIntegerEntry('N', 64)
        self.offset = self._settingsGroup.createIntegerEntry('Offset', 4)
        self.gridsize = self._settingsGroup.createIntegerEntry('Gridsize', 2)
        self.batchSize = self._settingsGroup.createIntegerEntry('BatchSize', 16)
        self.nFiltersScale = self._settingsGroup.createIntegerEntry('NFiltersScale', 2)
        self.probeTrainable = self._settingsGroup.createBooleanEntry('ProbeTrainable', False)
        self.intensityScaleTrainable = self._settingsGroup.createBooleanEntry(
            'IntensityScaleTrainable', False)
        self.objectBig = self._settingsGroup.createBooleanEntry('ObjectBig', True)
        self.probeBig = self._settingsGroup.createBooleanEntry('ProbeBig', False)
        self.probeScale = self._settingsGroup.createRealEntry('ProbeScale', '10.')
        self.probeMask = self._settingsGroup.createBooleanEntry('ProbeMask', True)
        self.modelType = self._settingsGroup.createStringEntry('ModelType', 'pinn')
        self.size = self._settingsGroup.createIntegerEntry('Size', 392)
        self.ampActivation = self._settingsGroup.createStringEntry('AmpActivation', 'sigmoid')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoPINNTrainingSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoPINNTraining')
        self._settingsGroup.addObserver(self)

        self.maeWeight = self._settingsGroup.createRealEntry('MAEWeight', '0.')
        self.nllWeight = self._settingsGroup.createRealEntry('NLLWeight', '1.')
        self.tvWeight = self._settingsGroup.createRealEntry('TVWeight', '0.')
        self.realspaceMAEWeight = self._settingsGroup.createRealEntry('RealspaceMAEWeight', '0.')
        self.realspaceWeight = self._settingsGroup.createRealEntry('RealspaceWeight', '0.')

        # generic settings shared with ptychonn
        self.maximumTrainingDatasetSize = self._settingsGroup.createIntegerEntry(
            'MaximumTrainingDatasetSize', 100000)
        self.validationSetFractionalSize = self._settingsGroup.createRealEntry(
            'ValidationSetFractionalSize', '0.1')
        self.optimizationEpochsPerHalfCycle = self._settingsGroup.createIntegerEntry(
            'OptimizationEpochsPerHalfCycle', 6)
        self.maximumLearningRate = self._settingsGroup.createRealEntry(
            'MaximumLearningRate', '1e-3')
        self.minimumLearningRate = self._settingsGroup.createRealEntry(
            'MinimumLearningRate', '1e-4')
        self.trainingEpochs = self._settingsGroup.createIntegerEntry('TrainingEpochs', 50)
        self.saveTrainingArtifacts = self._settingsGroup.createBooleanEntry(
            'SaveTrainingArtifacts', False)
        self.outputPath = self._settingsGroup.createPathEntry('OutputPath',
                                                              Path('/path/to/output'))
        self.outputSuffix = self._settingsGroup.createStringEntry('OutputSuffix', 'suffix')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
