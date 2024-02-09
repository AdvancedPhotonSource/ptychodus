from __future__ import annotations
from pathlib import Path

from ...api.observer import Observable, Observer
from ...api.settings import SettingsRegistry, SettingsGroup


class PtychoPINNModelSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup
        # Define settings specific to PtychoPINN
        # Example:
        self.learningRate = settingsGroup.createRealEntry('LearningRate', '1e-3')
        # Importing settings from params.py
        self.N = settingsGroup.createIntegerEntry('N', 64)
        self.offset = settingsGroup.createIntegerEntry('Offset', 4)
        self.gridsize = settingsGroup.createIntegerEntry('Gridsize', 2)
        self.outer_offset_train = settingsGroup.createIntegerEntry('OuterOffsetTrain', 0)
        self.outer_offset_test = settingsGroup.createIntegerEntry('OuterOffsetTest', 0)
        self.batch_size = settingsGroup.createIntegerEntry('BatchSize', 16)
        self.nepochs = settingsGroup.createIntegerEntry('NEpochs', 60)
        self.n_filters_scale = settingsGroup.createIntegerEntry('NFiltersScale', 2)
        self.nphotons = settingsGroup.createRealEntry('NPhotons', '1e9')
        self.probe_trainable = settingsGroup.createBooleanEntry('ProbeTrainable', False)
        self.intensity_scale_trainable = settingsGroup.createBooleanEntry('IntensityScaleTrainable', False)
        self.object_big = settingsGroup.createBooleanEntry('ObjectBig', True)
        self.probe_big = settingsGroup.createBooleanEntry('ProbeBig', False)
        self.probe_scale = settingsGroup.createRealEntry('ProbeScale', '10.')
        self.probe_mask = settingsGroup.createBooleanEntry('ProbeMask', True)
        self.model_type = settingsGroup.createStringEntry('ModelType', 'pinn')
        self.size = settingsGroup.createIntegerEntry('Size', 392)
        self.amp_activation = settingsGroup.createStringEntry('AmpActivation', 'sigmoid')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNModelSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoPINN')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoPINNTrainingSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup

        # generic settings shared with ptychonn
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

        # ptychopinn specific settings
        self.mae_weight = settingsGroup.createRealEntry('MAEWeight', '0.')
        self.nll_weight = settingsGroup.createRealEntry('NLLWeight', '1.')
        self.tv_weight = settingsGroup.createRealEntry('TVWeight', '0.')
        self.realspace_mae_weight = settingsGroup.createRealEntry('RealspaceMAEWeight', '0.')
        self.realspace_weight = settingsGroup.createRealEntry('RealspaceWeight', '0.')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoPINNTrainingSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoPINNTraining')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings
