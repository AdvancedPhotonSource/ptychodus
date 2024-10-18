from collections.abc import Sequence
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry

logger = logging.getLogger(__name__)


class TikeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Tike')
        self._settingsGroup.addObserver(self)

        self.numGpus = self._settingsGroup.createStringParameter('NumGpus', '1')
        self.noiseModel = self._settingsGroup.createStringParameter('NoiseModel', 'gaussian')
        self.numBatch = self._settingsGroup.createIntegerParameter('NumBatch', 10, minimum=1)
        self.batchMethod = self._settingsGroup.createStringParameter('BatchMethod', 'wobbly_center')
        self.numIter = self._settingsGroup.createIntegerParameter('NumIter', 1, minimum=1)
        self.convergenceWindow = self._settingsGroup.createIntegerParameter(
            'ConvergenceWindow', 0, minimum=0
        )
        self.alpha = self._settingsGroup.createRealParameter(
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
        self.notifyObservers()

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeMultigridSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('TikeMultigrid')
        self._settingsGroup.addObserver(self)

        self.useMultigrid = self._settingsGroup.createBooleanParameter('UseMultigrid', False)
        self.numLevels = self._settingsGroup.createIntegerParameter('NumLevels', 3, minimum=1)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeObjectCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('TikeObjectCorrection')
        self._settingsGroup.addObserver(self)

        self.useObjectCorrection = self._settingsGroup.createBooleanParameter(
            'UseObjectCorrection', True
        )
        self.positivityConstraint = self._settingsGroup.createRealParameter(
            'PositivityConstraint', 0.0, minimum=0.0, maximum=1.0
        )
        self.smoothnessConstraint = self._settingsGroup.createRealParameter(
            'SmoothnessConstraint',
            0.0,
            minimum=0.0,
            maximum=1.0 / 8,
        )
        self.useMagnitudeClipping = self._settingsGroup.createBooleanParameter(
            'UseMagnitudeClipping', False
        )

        self.useAdaptiveMoment = self._settingsGroup.createBooleanParameter(
            'UseAdaptiveMoment', False
        )
        self.mdecay = self._settingsGroup.createRealParameter(
            'MDecay', 0.9, minimum=0.0, maximum=1.0
        )
        self.vdecay = self._settingsGroup.createRealParameter(
            'VDecay', 0.999, minimum=0.0, maximum=1.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeProbeCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('TikeProbeCorrection')
        self._settingsGroup.addObserver(self)

        self.useProbeCorrection = self._settingsGroup.createBooleanParameter(
            'UseProbeCorrection', True
        )
        self.forceOrthogonality = self._settingsGroup.createBooleanParameter(
            'ForceOrthogonality', False
        )
        self.forceCenteredIntensity = self._settingsGroup.createBooleanParameter(
            'ForceCenteredIntensity', False
        )
        self.forceSparsity = self._settingsGroup.createRealParameter(
            'ForceSparsity', 0.0, minimum=0.0, maximum=1.0
        )
        self.useFiniteProbeSupport = self._settingsGroup.createBooleanParameter(
            'UseFiniteProbeSupport', False
        )
        self.probeSupportWeight = self._settingsGroup.createRealParameter(
            'ProbeSupportWeight', 10, minimum=0.0
        )
        self.probeSupportRadius = self._settingsGroup.createRealParameter(
            'ProbeSupportRadius', 0.35, minimum=0.0, maximum=0.5
        )
        self.probeSupportDegree = self._settingsGroup.createRealParameter(
            'ProbeSupportDegree', 2.5, minimum=0.0
        )
        self.additionalProbePenalty = self._settingsGroup.createRealParameter(
            'AdditionalProbePenalty', 0.0, minimum=0.0
        )

        self.useAdaptiveMoment = self._settingsGroup.createBooleanParameter(
            'UseAdaptiveMoment', False
        )
        self.mdecay = self._settingsGroup.createRealParameter(
            'MDecay', 0.9, minimum=0.0, maximum=1.0
        )
        self.vdecay = self._settingsGroup.createRealParameter(
            'VDecay', 0.999, minimum=0.0, maximum=1.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikePositionCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('TikePositionCorrection')
        self._settingsGroup.addObserver(self)

        self.usePositionCorrection = self._settingsGroup.createBooleanParameter(
            'UsePositionCorrection', False
        )
        self.usePositionRegularization = self._settingsGroup.createBooleanParameter(
            'UsePositionRegularization', False
        )
        self.updateMagnitudeLimit = self._settingsGroup.createRealParameter(
            'UpdateMagnitudeLimit', 0.0, minimum=0.0
        )
        # TODO transform: Global transform of positions.
        # TODO origin: The rotation center of the transformation.

        self.useAdaptiveMoment = self._settingsGroup.createBooleanParameter(
            'UseAdaptiveMoment', False
        )
        self.mdecay = self._settingsGroup.createRealParameter(
            'MDecay', 0.9, minimum=0.0, maximum=1.0
        )
        self.vdecay = self._settingsGroup.createRealParameter(
            'VDecay', 0.999, minimum=0.0, maximum=1.0
        )

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
