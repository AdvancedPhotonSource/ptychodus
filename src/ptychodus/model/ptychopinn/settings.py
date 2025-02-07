from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPINNModel')
        self._settings_group.addObserver(self)

        self.gridsize = self._settings_group.createIntegerParameter(
            'gridsize', 1, minimum=1, maximum=5
        )
        self.n_filters_scale = self._settings_group.createIntegerParameter(
            'n_filters_scale', 2, minimum=1, maximum=4
        )
        self.amp_activation = self._settings_group.createStringParameter(
            'amp_activation', 'sigmoid'
        )
        self.object_big = self._settings_group.createBooleanParameter('object_big', True)
        self.probe_big = self._settings_group.createBooleanParameter('probe_big', True)
        self.probe_mask = self._settings_group.createBooleanParameter('probe_mask', False)
        self.pad_object = self._settings_group.createBooleanParameter('pad_object', True)
        self.probe_scale = self._settings_group.createRealParameter('probe_scale', 4.0, minimum=0.0)
        self.gaussian_smoothing_sigma = self._settings_group.createRealParameter(
            'gaussian_smoothing_sigma', 0.0, minimum=0.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notifyObservers()


class PtychoPINNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPINNTraining')
        self._settings_group.addObserver(self)

        self.nphotons = self._settings_group.createRealParameter('NPhotons', 1e6)  # FIXME remove
        self.data_dir = self._settings_group.createPathParameter(
            'data_dir', Path('/path/to/training_data')
        )
        self.batch_size = self._settings_group.createIntegerParameter(
            'batch_size', 16, minimum=1, maximum=1 << 30
        )  # must be positive powers of two
        self.nepochs = self._settings_group.createIntegerParameter('nepochs', 50, minimum=1)
        self.mae_weight = self._settings_group.createRealParameter(
            'mae_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.nll_weight = self._settings_group.createRealParameter(
            'nll_weight', 1.0, minimum=0.0, maximum=1.0
        )
        self.realspace_mae_weight = self._settings_group.createRealParameter(
            'realspace_mae_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.realspace_weight = self._settings_group.createRealParameter(
            'realspace_weight', 0.0, minimum=0.0, maximum=1.0
        )
        self.positions_provided = self._settings_group.createBooleanParameter(
            'positions_provided', True
        )
        self.probe_trainable = self._settings_group.createBooleanParameter('probe_trainable', False)
        self.intensity_scale_trainable = self._settings_group.createBooleanParameter(
            'intensity_scale_trainable', True
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notifyObservers()


class PtychoPINNInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settings_group = registry.createGroup('PtychoPINNInference')
        self._settings_group.addObserver(self)

        self.model_path = self._settings_group.createPathParameter(
            'model_path', Path('/path/to/model.zip')
        )
        self.n_nearest_neighbors = self._settings_group.createIntegerParameter(
            'n_nearest_neighbors', 7, minimum=0
        )
        self.n_samples = self._settings_group.createIntegerParameter('n_samples', 1, minimum=1)

    def update(self, observable: Observable) -> None:
        if observable is self._settings_group:
            self.notifyObservers()
