from __future__ import annotations
from pathlib import Path
import dataclasses

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry, SettingsGroup

# from ptychonn.position.configs import InferenceConfig
from ptychodus.model.ptychonn.config_temp import InferenceConfig


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

class PtychoNNPositionPredictionSettings(Observable, Observer):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__()
        self._settingsGroup = settingsGroup

        # configs = InferenceConfig()
        # for field in dataclasses.fields(configs):
            # print(field)
            # settingsGroup.createIntegerEntry(field.name, 100000)

        self.reconstructorImagePath = settingsGroup.createPathEntry(
            'Reconstructor Image Path', Path('/path/to/output'))
        self.probePositionListPath = settingsGroup.createPathEntry(
            'Probe Position List Path', Path('/path/to/output'))
        self.probePositionDataUnit = settingsGroup.createStringEntry(
            'Probe Position Data Unit', 'nm')
        self.pixelSizeNM = settingsGroup.createRealEntry(
            'Pixel Size NM', 1)
        self.baselinePositionList = settingsGroup.createPathEntry(
            'Baseline Position List', Path('/path/to/output'))
        self.centralCrop = settingsGroup.createStringEntry(
            'Central Crop', '1, 2')
        self.method = settingsGroup.createStringEntry(
            'Method', 'serial')
        self.numberNeighborsCollective = settingsGroup.createIntegerEntry(
            'Number of Neighbors Collective', 4)
        self.offsetEstimatorOrder = settingsGroup.createIntegerEntry(
            'Offset Estimator Order', 1)
        self.offsetEstimatorBeta = settingsGroup.createRealEntry(
            'Offset Estimator Beta', 0.5)
        self.smoothConstraintWeight = settingsGroup.createRealEntry(
            'Smooth Contraint Weight', 1e-2)
        self.rectangularGrid = settingsGroup.createBooleanEntry(
            'Rectangular Grid', False)
        self.randomSeed = settingsGroup.createIntegerEntry(
            'Random Seed', 123)
        self.debug = settingsGroup.createBooleanEntry(
            'Debug', False)
        self.registrationMethod = settingsGroup.createStringEntry(    
            'RegisrationMethod', 'hybrid')
        self.hybridRegistrationAlgs = settingsGroup.createStringEntry(    
            'Hybrid Registration Algs', 'error_map_expandable')
        self.hybridRegistrationTols = settingsGroup.createIntegerEntry(
            'Nonhybrid Registration Tols', 1)
        self.maxShift = settingsGroup.createIntegerEntry(
            'Max Shift', 40)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> PtychoNNPositionPredictionSettings:
        settingsGroup = settingsRegistry.createGroup('PtychoNNPositionPrediction')
        settings = cls(settingsGroup)
        settingsGroup.addObserver(settings)
        return settings

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()

