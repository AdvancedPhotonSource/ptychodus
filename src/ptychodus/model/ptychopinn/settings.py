from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPINNModel')
        self._settings_group.addObserver(self)

        self.learning_rate = self._settings_group.createRealParameter('learning_rate', 1e-3)
        self.N = self._settings_group.createIntegerParameter('N', 64)
        self.offset = self._settings_group.createIntegerParameter('offset', 4)
        self.gridsize = self._settings_group.createIntegerParameter('gridsize', 2, minimum=1)
        self.batch_size = self._settings_group.createIntegerParameter('batch_size', 16, minimum=1)
        self.n_filters_scale = self._settings_group.createIntegerParameter(
            'n_filters_scale', 2, minimum=1
        )

        self.is_probe_trainable = self._settings_group.createBooleanParameter(
            'probe.trainable', False
        )
        self.intensity_scale_trainable = self._settings_group.createBooleanParameter(
            'intensity_scale.trainable', False
        )
        self.object_big = self._settings_group.createBooleanParameter('object.big', True)
        self.probe_big = self._settings_group.createBooleanParameter('probe.big', False)
        self.probe_scale = self._settings_group.createRealParameter(
            'probe_scale', 10.0, minimum=1.0e-3, maximum=1.0e3
        )
        self.probe_mask = self._settings_group.createBooleanParameter('probe.mask', True)
        self.model_type = self._settings_group.createStringParameter(
            'model_type', 'pinn'
        )  # pinn or supervised
        self.size = self._settings_group.createIntegerParameter('size', 392, minimum=1)
        self.amp_activation = self._settings_group.createStringParameter(
            'amp_activation', 'sigmoid'
        )  # sigmoid or swish

    def update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notifyObservers()


class PtychoPINNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPINNTraining')
        self._settings_group.addObserver(self)

        self.mae_weight = self._settings_group.createRealParameter(
            'mae_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.nll_weight = self._settings_group.createRealParameter(
            'nll_weight', 1.0, minimum=0.0, maximum=1.0
        )
        self.tv_weight = self._settings_group.createRealParameter(
            'tv_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.realspace_mae_weight = self._settings_group.createRealParameter(
            'realspace_mae_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.realspace_weight = self._settings_group.createRealParameter(
            'realspace_weight', 0.0, minimum=0.0, maximum=1.0
        )

        # FIXME BEGIN
        # generic settings shared with ptychonn
        self.maximumTrainingDatasetSize = self._settings_group.createIntegerParameter(
            'MaximumTrainingDatasetSize', 100000
        )
        self.validationSetFractionalSize = self._settings_group.createRealParameter(
            'ValidationSetFractionalSize', 0.1, minimum=0.0, maximum=1.0
        )
        self.optimizationEpochsPerHalfCycle = self._settings_group.createIntegerParameter(
            'OptimizationEpochsPerHalfCycle', 6
        )
        self.maximumLearningRate = self._settings_group.createRealParameter(
            'MaximumLearningRate', 1e-3
        )
        self.minimumLearningRate = self._settings_group.createRealParameter(
            'MinimumLearningRate', 1e-4
        )
        # nepochs: number of epochs?
        self.trainingEpochs = self._settings_group.createIntegerParameter(
            'TrainingEpochs', 50, minimum=1
        )
        self.saveTrainingArtifacts = self._settings_group.createBooleanParameter(
            'SaveTrainingArtifacts', False
        )
        # FIXME END

        self.output_path = self._settings_group.createPathParameter(
            'output_path', Path('/path/to/output')
        )
        self.output_prefix = self._settings_group.createStringParameter('output_prefix', 'outputs')
        self.output_suffix = self._settings_group.createStringParameter('output_suffix', 'suffix')

    def update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notifyObservers()
