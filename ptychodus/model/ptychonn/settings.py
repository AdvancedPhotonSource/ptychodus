from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

# from ptychonn.position.configs import InferenceConfig
from ptychodus.model.ptychonn.config_temp import InferenceConfig


class PtychoNNModelSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoNN')
        self._settingsGroup.addObserver(self)

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
        self.statusIntervalInEpochs = self._settingsGroup.createIntegerEntry(
            'StatusIntervalInEpochs', 1)

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

