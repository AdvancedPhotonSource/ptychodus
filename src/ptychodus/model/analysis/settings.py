from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class AffineTransformEstimatorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('AffineTransformEstimator')
        self._group.add_observer(self)

        self.num_iterations = self._group.create_integer_parameter(
            'NumberOfIterations', 1000, minimum=1
        )
        self.inlier_threshold = self._group.create_real_parameter(
            'InlierThreshold', 0.05, minimum=0.0
        )
        self.min_inliers = self._group.create_integer_parameter('MinimumInliers', 10, minimum=3)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class DiffractionSimulatorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('DiffractionSimulator')
        self._group.add_observer(self)

        self.add_poisson_noise = self._group.create_boolean_parameter('AddPoissonNoise', False)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class ProbePropagatorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('ProbePropagator')
        self._group.add_observer(self)

        self.begin_coordinate_m = self._group.create_real_parameter(
            'BeginCoordinateInMeters', -1e-3
        )
        self.end_coordinate_m = self._group.create_real_parameter('EndCoordinateInMeters', 1e-3)
        self.num_steps = self._group.create_integer_parameter(
            'NumberOfSteps', 100, minimum=1, maximum=999
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
