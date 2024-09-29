from __future__ import annotations
from collections.abc import Sequence
import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    BooleanParameter,
    IntegerParameter,
    RealParameter,
    StringParameter,
)
from ptychodus.api.settings import SettingsRegistry

logger = logging.getLogger(__name__)


class TikeSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('Tike')
        self._settingsGroup.addObserver(self)

        self.numGpus = StringParameter(self._settingsGroup, 'NumGpus', '1')
        self.noiseModel = StringParameter(self._settingsGroup, 'NoiseModel', 'gaussian')
        self.numBatch = IntegerParameter(self._settingsGroup, 'NumBatch', 10, minimum=1)
        self.batchMethod = StringParameter(self._settingsGroup, 'BatchMethod', 'wobbly_center')
        self.numIter = IntegerParameter(self._settingsGroup, 'NumIter', 1, minimum=1)
        self.convergenceWindow = IntegerParameter(
            self._settingsGroup, 'ConvergenceWindow', 0, minimum=0
        )
        self.alpha = RealParameter(self._settingsGroup, 'Alpha', 0.05, minimum=0.0, maximum=1.0)
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

        self.useMultigrid = BooleanParameter(self._settingsGroup, 'UseMultigrid', False)
        self.numLevels = IntegerParameter(self._settingsGroup, 'NumLevels', 3, minimum=1)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeObjectCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('TikeObjectCorrection')
        self._settingsGroup.addObserver(self)

        self.useObjectCorrection = BooleanParameter(
            self._settingsGroup, 'UseObjectCorrection', True
        )
        self.positivityConstraint = RealParameter(
            self._settingsGroup, 'PositivityConstraint', 0.0, minimum=0.0, maximum=1.0
        )
        self.smoothnessConstraint = RealParameter(
            self._settingsGroup,
            'SmoothnessConstraint',
            0.0,
            minimum=0.0,
            maximum=1.0 / 8,
        )
        self.useMagnitudeClipping = BooleanParameter(
            self._settingsGroup, 'UseMagnitudeClipping', False
        )

        self.useAdaptiveMoment = BooleanParameter(self._settingsGroup, 'UseAdaptiveMoment', False)
        self.mdecay = RealParameter(self._settingsGroup, 'MDecay', 0.9, minimum=0.0, maximum=1.0)
        self.vdecay = RealParameter(self._settingsGroup, 'VDecay', 0.999, minimum=0.0, maximum=1.0)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikeProbeCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('TikeProbeCorrection')
        self._settingsGroup.addObserver(self)

        self.useProbeCorrection = BooleanParameter(self._settingsGroup, 'UseProbeCorrection', True)
        self.forceOrthogonality = BooleanParameter(self._settingsGroup, 'ForceOrthogonality', False)
        self.forceCenteredIntensity = BooleanParameter(
            self._settingsGroup, 'ForceCenteredIntensity', False
        )
        self.forceSparsity = RealParameter(
            self._settingsGroup, 'ForceSparsity', 0.0, minimum=0.0, maximum=1.0
        )
        self.useFiniteProbeSupport = BooleanParameter(
            self._settingsGroup, 'UseFiniteProbeSupport', False
        )
        self.probeSupportWeight = RealParameter(
            self._settingsGroup, 'ProbeSupportWeight', 10, minimum=0.0
        )
        self.probeSupportRadius = RealParameter(
            self._settingsGroup, 'ProbeSupportRadius', 0.35, minimum=0.0, maximum=0.5
        )
        self.probeSupportDegree = RealParameter(
            self._settingsGroup, 'ProbeSupportDegree', 2.5, minimum=0.0
        )
        self.additionalProbePenalty = RealParameter(
            self._settingsGroup, 'AdditionalProbePenalty', 0.0, minimum=0.0
        )

        self.useAdaptiveMoment = BooleanParameter(self._settingsGroup, 'UseAdaptiveMoment', False)
        self.mdecay = RealParameter(self._settingsGroup, 'MDecay', 0.9, minimum=0.0, maximum=1.0)
        self.vdecay = RealParameter(self._settingsGroup, 'VDecay', 0.999, minimum=0.0, maximum=1.0)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()


class TikePositionCorrectionSettings(Observable, Observer):
    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('TikePositionCorrection')
        self._settingsGroup.addObserver(self)

        self.usePositionCorrection = BooleanParameter(
            self._settingsGroup, 'UsePositionCorrection', False
        )
        self.usePositionRegularization = BooleanParameter(
            self._settingsGroup, 'UsePositionRegularization', False
        )
        self.updateMagnitudeLimit = RealParameter(
            self._settingsGroup, 'UpdateMagnitudeLimit', 0.0, minimum=0.0
        )
        # TODO transform: Global transform of positions.
        # TODO origin: The rotation center of the transformation.

        self.useAdaptiveMoment = BooleanParameter(self._settingsGroup, 'UseAdaptiveMoment', False)
        self.mdecay = RealParameter(self._settingsGroup, 'MDecay', 0.9, minimum=0.0, maximum=1.0)
        self.vdecay = RealParameter(self._settingsGroup, 'VDecay', 0.999, minimum=0.0, maximum=1.0)

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
