from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    DecimalParameter,
    IntegerParameter,
)
from ptychodus.api.settings import SettingsRegistry


class PtychoNNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("PtychoNN")
        self._settingsGroup.addObserver(self)

        self.numberOfConvolutionKernels = IntegerParameter(
            self._settingsGroup, "NumberOfConvolutionKernels", 16
        )
        self.batchSize = IntegerParameter(self._settingsGroup, "BatchSize", 64)
        self.useBatchNormalization = BooleanParameter(
            self._settingsGroup, "UseBatchNormalization", False
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class PtychoNNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup("PtychoNNTraining")
        self._settingsGroup.addObserver(self)

        self.maximumTrainingDatasetSize = IntegerParameter(
            self._settingsGroup, "MaximumTrainingDatasetSize", 100000
        )
        self.validationSetFractionalSize = DecimalParameter(
            self._settingsGroup, "ValidationSetFractionalSize", "0.1"
        )
        self.maximumLearningRate = DecimalParameter(
            self._settingsGroup, "MaximumLearningRate", "1e-3"
        )
        self.minimumLearningRate = DecimalParameter(
            self._settingsGroup, "MinimumLearningRate", "1e-4"
        )
        self.trainingEpochs = IntegerParameter(self._settingsGroup, "TrainingEpochs", 50)
        self.statusIntervalInEpochs = IntegerParameter(
            self._settingsGroup, "StatusIntervalInEpochs", 1
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
