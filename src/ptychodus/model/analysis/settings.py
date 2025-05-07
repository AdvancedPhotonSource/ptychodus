from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class ProbePropagationSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('ProbePropagation')
        self._group.add_observer(self)

        self.begin_coordinate_m = self._group.create_real_parameter(
            'BeginCoordinateInMeters', -1e-3
        )
        self.end_coordinate_m = self._group.create_real_parameter('EndCoordinateInMeters', 1e-3)
        self.num_steps = self._group.create_integer_parameter('NumberOfSteps', 100)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class AffineTransformEstimatorSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('AffineTransformEstimator')
        self._group.add_observer(self)

        self.num_shuffles = self._group.create_integer_parameter('NumberOfShuffles', 10, minimum=1)
        self.inlier_threshold = self._group.create_real_parameter(
            'InlierThreshold', 1e-6, minimum=0.0
        )
        self.min_inliers = self._group.create_integer_parameter('MinimumInliers', 10, minimum=3)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
