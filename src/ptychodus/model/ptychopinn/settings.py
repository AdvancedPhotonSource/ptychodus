from pathlib import Path

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPINNModelSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNModel')
        self._group.add_observer(self)

        self.gridsize = self._group.create_integer_parameter('GridSize', 1, minimum=1, maximum=5)
        self.n_filters_scale = self._group.create_integer_parameter(
            'NumFiltersScale', 2, minimum=1, maximum=4
        )
        self.amp_activation = self._group.create_string_parameter('AmpActivation', 'sigmoid')
        self.object_big = self._group.create_boolean_parameter('ObjectBig', True)
        self.probe_big = self._group.create_boolean_parameter('ProbeBig', True)
        self.probe_mask = self._group.create_boolean_parameter('ProbeMask', False)
        self.pad_object = self._group.create_boolean_parameter('PadObject', True)
        self.probe_scale = self._group.create_real_parameter('ProbeScale', 4.0, minimum=0.0)
        self.gaussian_smoothing_sigma = self._group.create_real_parameter(
            'GaussianSmoothingSigma', 0.0, minimum=0.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNTrainingSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNTraining')
        self._group.add_observer(self)

        self.nphotons = self._group.create_real_parameter('NumPhotons', 1e6)  # TODO remove
        self.data_dir = self._group.create_path_parameter('DataDir', Path('/path/to/training_data'))
        self.batch_size = self._group.create_integer_parameter(
            'BatchSize', 16, minimum=1, maximum=1 << 30
        )  # must be positive powers of two
        self.nepochs = self._group.create_integer_parameter('NumEpochs', 50, minimum=1)
        self.mae_weight = self._group.create_real_parameter(
            'MAEWeight', 0.0, minimum=0.0, maximum=1.0
        )
        self.nll_weight = self._group.create_real_parameter(
            'NLLWeight', 1.0, minimum=0.0, maximum=1.0
        )
        self.realspace_mae_weight = self._group.create_real_parameter(
            'RealspaceMAEWeight', 0.0, minimum=0.0, maximum=1.0
        )
        self.realspace_weight = self._group.create_real_parameter(
            'RealspaceWeight', 0.0, minimum=0.0, maximum=1.0
        )
        self.positions_provided = self._group.create_boolean_parameter('PositionsProvided', True)
        self.probe_trainable = self._group.create_boolean_parameter('ProbeTrainable', False)
        self.intensity_scale_trainable = self._group.create_boolean_parameter(
            'IntensityScaleTrainable', True
        )
        self.output_dir = self._group.create_path_parameter(
            'OutputDir', Path('/path/to/output_data')
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class PtychoPINNInferenceSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('PtychoPINNInference')
        self._group.add_observer(self)

        self.model_path = self._group.create_path_parameter('ModelPath', Path('/path/to/model.zip'))
        self.n_nearest_neighbors = self._group.create_integer_parameter(
            'NumNearestNeighbors', 7, minimum=0
        )
        self.n_samples = self._group.create_integer_parameter('NumSamples', 1, minimum=1)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
