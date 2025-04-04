from collections.abc import Sequence
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

logger = logging.getLogger(__name__)


class TikeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('Tike')
        self._group.add_observer(self)

        self.num_gpus = self._group.create_string_parameter('NumGpus', '1')
        self.noise_model = self._group.create_string_parameter('NoiseModel', 'gaussian')
        self.num_batch = self._group.create_integer_parameter('NumBatch', 10, minimum=1)
        self.batch_method = self._group.create_string_parameter('BatchMethod', 'wobbly_center')
        self.num_iter = self._group.create_integer_parameter('NumIter', 1, minimum=1)
        self.convergence_window = self._group.create_integer_parameter(
            'ConvergenceWindow', 0, minimum=0
        )
        self.alpha = self._group.create_real_parameter('Alpha', 0.05, minimum=0.0, maximum=1.0)
        self._logger = logging.getLogger('tike')

    def get_noise_models(self) -> Sequence[str]:
        return ['poisson', 'gaussian']

    def get_batch_methods(self) -> Sequence[str]:
        return ['wobbly_center', 'wobbly_center_random_bootstrap', 'compact']

    def get_log_levels(self) -> Sequence[str]:
        return ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

    def get_log_level(self) -> str:
        level = self._logger.getEffectiveLevel()
        return logging.getLevelName(level)

    def set_log_level(self, name: str) -> None:
        name_before = self.get_log_level()

        if name == name_before:
            return

        try:
            self._logger.setLevel(name)
        except ValueError:
            logger.error(f'Bad log level "{name}".')

        name_after = self.get_log_level()
        logger.info(f'Changed Tike log level {name_before} -> {name_after}')
        self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class TikeMultigridSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('TikeMultigrid')
        self._group.add_observer(self)

        self.use_multigrid = self._group.create_boolean_parameter('UseMultigrid', False)
        self.num_levels = self._group.create_integer_parameter('NumLevels', 3, minimum=1)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class TikeObjectCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('TikeObjectCorrection')
        self._group.add_observer(self)

        self.use_object_correction = self._group.create_boolean_parameter(
            'UseObjectCorrection', True
        )
        self.positivity_constraint = self._group.create_real_parameter(
            'PositivityConstraint', 0.0, minimum=0.0, maximum=1.0
        )
        self.smoothness_constraint = self._group.create_real_parameter(
            'SmoothnessConstraint',
            0.0,
            minimum=0.0,
            maximum=1.0 / 8,
        )
        self.use_magnitude_clipping = self._group.create_boolean_parameter(
            'UseMagnitudeClipping', False
        )

        self.use_adaptive_moment = self._group.create_boolean_parameter('UseAdaptiveMoment', False)
        self.mdecay = self._group.create_real_parameter('MDecay', 0.9, minimum=0.0, maximum=1.0)
        self.vdecay = self._group.create_real_parameter('VDecay', 0.999, minimum=0.0, maximum=1.0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class TikeProbeCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('TikeProbeCorrection')
        self._group.add_observer(self)

        self.use_probe_correction = self._group.create_boolean_parameter('UseProbeCorrection', True)
        self.force_orthogonality = self._group.create_boolean_parameter('ForceOrthogonality', False)
        self.force_centered_intensity = self._group.create_boolean_parameter(
            'ForceCenteredIntensity', False
        )
        self.force_sparsity = self._group.create_real_parameter(
            'ForceSparsity', 0.0, minimum=0.0, maximum=1.0
        )
        self.use_finite_probe_support = self._group.create_boolean_parameter(
            'UseFiniteProbeSupport', False
        )
        self.probe_support_weight = self._group.create_real_parameter(
            'ProbeSupportWeight', 10, minimum=0.0
        )
        self.probe_support_radius = self._group.create_real_parameter(
            'ProbeSupportRadius', 0.35, minimum=0.0, maximum=0.5
        )
        self.probe_support_degree = self._group.create_real_parameter(
            'ProbeSupportDegree', 2.5, minimum=0.0
        )
        self.additional_probe_penalty = self._group.create_real_parameter(
            'AdditionalProbePenalty', 0.0, minimum=0.0
        )

        self.use_adaptive_moment = self._group.create_boolean_parameter('UseAdaptiveMoment', False)
        self.mdecay = self._group.create_real_parameter('MDecay', 0.9, minimum=0.0, maximum=1.0)
        self.vdecay = self._group.create_real_parameter('VDecay', 0.999, minimum=0.0, maximum=1.0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()


class TikePositionCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._group = registry.create_group('TikePositionCorrection')
        self._group.add_observer(self)

        self.use_position_correction = self._group.create_boolean_parameter(
            'UsePositionCorrection', False
        )
        self.use_position_regularization = self._group.create_boolean_parameter(
            'UsePositionRegularization', False
        )
        self.update_magnitude_limit = self._group.create_real_parameter(
            'UpdateMagnitudeLimit', 0.0, minimum=0.0
        )
        # TODO transform: Global transform of positions.
        # TODO origin: The rotation center of the transformation.

        self.use_adaptive_moment = self._group.create_boolean_parameter('UseAdaptiveMoment', False)
        self.mdecay = self._group.create_real_parameter('MDecay', 0.9, minimum=0.0, maximum=1.0)
        self.vdecay = self._group.create_real_parameter('VDecay', 0.999, minimum=0.0, maximum=1.0)

    def _update(self, observable: Observable) -> None:
        if observable is self._group:
            self.notify_observers()
