from collections.abc import Sequence
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

logger = logging.getLogger(__name__)


class TikeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('Tike')
        self._settingsGroup.add_observer(self)

        self.numGpus = self._settingsGroup.create_string_parameter('NumGpus', '1')
        self.noiseModel = self._settingsGroup.create_string_parameter('NoiseModel', 'gaussian')
        self.numBatch = self._settingsGroup.create_integer_parameter('NumBatch', 10, minimum=1)
        self.batchMethod = self._settingsGroup.create_string_parameter(
            'BatchMethod', 'wobbly_center'
        )
        self.numIter = self._settingsGroup.create_integer_parameter('NumIter', 1, minimum=1)
        self.convergenceWindow = self._settingsGroup.create_integer_parameter(
            'ConvergenceWindow', 0, minimum=0
        )
        self.alpha = self._settingsGroup.create_real_parameter(
            'Alpha', 0.05, minimum=0.0, maximum=1.0
        )
        self._logger = logging.getLogger('tike')

    def getNoiseModels(self) -> Sequence[str]:
        return ['poisson', 'gaussian']

    def getBatchMethods(self) -> Sequence[str]:
        return ['wobbly_center', 'wobbly_center_random_bootstrap', 'compact']

    def getLogLevels(self) -> Sequence[str]:
        return ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG']

    def getLogLevel(self) -> str:
        level = self._logger.getEffectiveLevel()
        return logging.getLevelName(level)

    def setLogLevel(self, name: str) -> None:
        nameBefore = self.getLogLevel()

        if name == nameBefore:
            return

        try:
            self._logger.setLevel(name)
        except ValueError:
            logger.error(f'Bad log level "{name}".')

        nameAfter = self.getLogLevel()
        logger.info(f'Changed Tike log level {nameBefore} -> {nameAfter}')
        self.notify_observers()

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class TikeMultigridSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('TikeMultigrid')
        self._settingsGroup.add_observer(self)

        self.useMultigrid = self._settingsGroup.create_boolean_parameter('UseMultigrid', False)
        self.numLevels = self._settingsGroup.create_integer_parameter('NumLevels', 3, minimum=1)

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class TikeObjectCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('TikeObjectCorrection')
        self._settingsGroup.add_observer(self)

        self.useObjectCorrection = self._settingsGroup.create_boolean_parameter(
            'UseObjectCorrection', True
        )
        self.positivityConstraint = self._settingsGroup.create_real_parameter(
            'PositivityConstraint', 0.0, minimum=0.0, maximum=1.0
        )
        self.smoothnessConstraint = self._settingsGroup.create_real_parameter(
            'SmoothnessConstraint',
            0.0,
            minimum=0.0,
            maximum=1.0 / 8,
        )
        self.useMagnitudeClipping = self._settingsGroup.create_boolean_parameter(
            'UseMagnitudeClipping', False
        )

        self.useAdaptiveMoment = self._settingsGroup.create_boolean_parameter(
            'UseAdaptiveMoment', False
        )
        self.mdecay = self._settingsGroup.create_real_parameter(
            'MDecay', 0.9, minimum=0.0, maximum=1.0
        )
        self.vdecay = self._settingsGroup.create_real_parameter(
            'VDecay', 0.999, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class TikeProbeCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('TikeProbeCorrection')
        self._settingsGroup.add_observer(self)

        self.useProbeCorrection = self._settingsGroup.create_boolean_parameter(
            'UseProbeCorrection', True
        )
        self.forceOrthogonality = self._settingsGroup.create_boolean_parameter(
            'ForceOrthogonality', False
        )
        self.forceCenteredIntensity = self._settingsGroup.create_boolean_parameter(
            'ForceCenteredIntensity', False
        )
        self.forceSparsity = self._settingsGroup.create_real_parameter(
            'ForceSparsity', 0.0, minimum=0.0, maximum=1.0
        )
        self.useFiniteProbeSupport = self._settingsGroup.create_boolean_parameter(
            'UseFiniteProbeSupport', False
        )
        self.probeSupportWeight = self._settingsGroup.create_real_parameter(
            'ProbeSupportWeight', 10, minimum=0.0
        )
        self.probeSupportRadius = self._settingsGroup.create_real_parameter(
            'ProbeSupportRadius', 0.35, minimum=0.0, maximum=0.5
        )
        self.probeSupportDegree = self._settingsGroup.create_real_parameter(
            'ProbeSupportDegree', 2.5, minimum=0.0
        )
        self.additionalProbePenalty = self._settingsGroup.create_real_parameter(
            'AdditionalProbePenalty', 0.0, minimum=0.0
        )

        self.useAdaptiveMoment = self._settingsGroup.create_boolean_parameter(
            'UseAdaptiveMoment', False
        )
        self.mdecay = self._settingsGroup.create_real_parameter(
            'MDecay', 0.9, minimum=0.0, maximum=1.0
        )
        self.vdecay = self._settingsGroup.create_real_parameter(
            'VDecay', 0.999, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()


class TikePositionCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.create_group('TikePositionCorrection')
        self._settingsGroup.add_observer(self)

        self.usePositionCorrection = self._settingsGroup.create_boolean_parameter(
            'UsePositionCorrection', False
        )
        self.usePositionRegularization = self._settingsGroup.create_boolean_parameter(
            'UsePositionRegularization', False
        )
        self.updateMagnitudeLimit = self._settingsGroup.create_real_parameter(
            'UpdateMagnitudeLimit', 0.0, minimum=0.0
        )
        # TODO transform: Global transform of positions.
        # TODO origin: The rotation center of the transformation.

        self.useAdaptiveMoment = self._settingsGroup.create_boolean_parameter(
            'UseAdaptiveMoment', False
        )
        self.mdecay = self._settingsGroup.create_real_parameter(
            'MDecay', 0.9, minimum=0.0, maximum=1.0
        )
        self.vdecay = self._settingsGroup.create_real_parameter(
            'VDecay', 0.999, minimum=0.0, maximum=1.0
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notify_observers()
