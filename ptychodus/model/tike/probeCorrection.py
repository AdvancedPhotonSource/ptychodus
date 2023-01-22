from __future__ import annotations
from decimal import Decimal

from ...api.geometry import Interval
from ...api.settings import SettingsGroup, SettingsRegistry
from .adaptiveMoment import TikeAdaptiveMomentPresenter, TikeAdaptiveMomentSettings


class TikeProbeCorrectionSettings(TikeAdaptiveMomentSettings):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useProbeCorrection = settingsGroup.createBooleanEntry('UseProbeCorrection', True)
        self.orthogonalityConstraint = settingsGroup.createBooleanEntry(
            'OrthogonalityConstraint', True)
        self.centeredIntensityConstraint = settingsGroup.createBooleanEntry(
            'CenteredIntensityConstraint', False)
        self.sparsityConstraint = settingsGroup.createRealEntry('SparsityConstraint', '1')
        self.useFiniteProbeSupport = settingsGroup.createBooleanEntry(
            'UseFiniteProbeSupport', True)
        self.probeSupportWeight = settingsGroup.createRealEntry('ProbeSupportWeight', '10')
        self.probeSupportRadius = settingsGroup.createRealEntry('ProbeSupportRadius', '0.3')
        self.probeSupportDegree = settingsGroup.createRealEntry('ProbeSupportDegree', '5')
        self.additionalProbePenalty = settingsGroup.createRealEntry('AdditionalProbePenalty', '0')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeProbeCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikeProbeCorrection'))


class TikeProbeCorrectionPresenter(TikeAdaptiveMomentPresenter[TikeProbeCorrectionSettings]):

    def __init__(self, settings: TikeProbeCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(cls, settings: TikeProbeCorrectionSettings) -> TikeProbeCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isProbeCorrectionEnabled(self) -> bool:
        return self._settings.useProbeCorrection.value

    def setProbeCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.useProbeCorrection.value = enabled

    def isOrthogonalityConstraintEnabled(self) -> bool:
        return self._settings.orthogonalityConstraint.value

    def setOrthogonalityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.orthogonalityConstraint.value = enabled

    def isCenteredIntensityConstraintEnabled(self) -> bool:
        return self._settings.centeredIntensityConstraint.value

    def setCenteredIntensityConstraintEnabled(self, enabled: bool) -> None:
        self._settings.centeredIntensityConstraint.value = enabled

    def getSparsityConstraintLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getSparsityConstraint(self) -> Decimal:
        limits = self.getSparsityConstraintLimits()
        return limits.clamp(self._settings.sparsityConstraint.value)

    def setSparsityConstraint(self, value: Decimal) -> None:
        self._settings.sparsityConstraint.value = value

    def isFiniteProbeSupportEnabled(self) -> bool:
        return self._settings.useFiniteProbeSupport.value

    def setFiniteProbeSupportEnabled(self, enabled: bool) -> None:
        self._settings.useFiniteProbeSupport.value = enabled

    def getProbeSupportWeightMinimum(self) -> Decimal:
        return Decimal()

    def getProbeSupportWeight(self) -> Decimal:
        return max(self._settings.probeSupportWeight.value, self.getProbeSupportWeightMinimum())

    def setProbeSupportWeight(self, value: Decimal) -> None:
        self._settings.probeSupportWeight.value = value

    def getProbeSupportRadiusLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1) / 2)

    def getProbeSupportRadius(self) -> Decimal:
        limits = self.getProbeSupportRadiusLimits()
        return limits.clamp(self._settings.probeSupportRadius.value)

    def setProbeSupportRadius(self, value: Decimal) -> None:
        self._settings.probeSupportRadius.value = value

    def getProbeSupportDegreeMinimum(self) -> Decimal:
        return Decimal()

    def getProbeSupportDegree(self) -> Decimal:
        return max(self._settings.probeSupportDegree.value, self.getProbeSupportDegreeMinimum())

    def setProbeSupportDegree(self, value: Decimal) -> None:
        self._settings.probeSupportDegree.value = value

    def getAdditionalProbePenaltyMinimum(self) -> Decimal:
        return Decimal()

    def getAdditionalProbePenalty(self) -> Decimal:
        return max(self._settings.additionalProbePenalty.value,
                   self.getAdditionalProbePenaltyMinimum())

    def setAdditionalProbePenalty(self, value: Decimal) -> None:
        self._settings.additionalProbePenalty.value = value
