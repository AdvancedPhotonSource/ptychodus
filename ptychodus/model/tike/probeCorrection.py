from __future__ import annotations
from decimal import Decimal

from ptychodus.api.geometry import Interval
from ptychodus.api.settings import SettingsGroup, SettingsRegistry

from .adaptiveMoment import TikeAdaptiveMomentPresenter, TikeAdaptiveMomentSettings


class TikeProbeCorrectionSettings(TikeAdaptiveMomentSettings):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useProbeCorrection = settingsGroup.createBooleanEntry('UseProbeCorrection', True)
        self.forceOrthogonality = settingsGroup.createBooleanEntry('ForceOrthogonality', False)
        self.forceCenteredIntensity = settingsGroup.createBooleanEntry(
            'ForceCenteredIntensity', False)
        self.forceSparsity = settingsGroup.createRealEntry('ForceSparsity', '0')
        self.useFiniteProbeSupport = settingsGroup.createBooleanEntry(
            'UseFiniteProbeSupport', False)
        self.probeSupportWeight = settingsGroup.createRealEntry('ProbeSupportWeight', '10')
        self.probeSupportRadius = settingsGroup.createRealEntry('ProbeSupportRadius', '0.35')
        self.probeSupportDegree = settingsGroup.createRealEntry('ProbeSupportDegree', '2.5')
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
        return self._settings.useProbeCorrection.getValue()

    def setProbeCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.useProbeCorrection.setValue(enabled)

    def isForceOrthogonalityEnabled(self) -> bool:
        return self._settings.forceOrthogonality.getValue()

    def setForceOrthogonalityEnabled(self, enabled: bool) -> None:
        self._settings.forceOrthogonality.setValue(enabled)

    def isForceCenteredIntensityEnabled(self) -> bool:
        return self._settings.forceCenteredIntensity.getValue()

    def setForceCenteredIntensityEnabled(self, enabled: bool) -> None:
        self._settings.forceCenteredIntensity.setValue(enabled)

    def getForceSparsityLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getForceSparsity(self) -> Decimal:
        limits = self.getForceSparsityLimits()
        return limits.clamp(self._settings.forceSparsity.getValue())

    def setForceSparsity(self, value: Decimal) -> None:
        self._settings.forceSparsity.setValue(value)

    def isFiniteProbeSupportEnabled(self) -> bool:
        return self._settings.useFiniteProbeSupport.getValue()

    def setFiniteProbeSupportEnabled(self, enabled: bool) -> None:
        self._settings.useFiniteProbeSupport.setValue(enabled)

    def getProbeSupportWeightMinimum(self) -> Decimal:
        return Decimal()

    def getProbeSupportWeight(self) -> Decimal:
        return max(self._settings.probeSupportWeight.getValue(),
                   self.getProbeSupportWeightMinimum())

    def setProbeSupportWeight(self, value: Decimal) -> None:
        self._settings.probeSupportWeight.setValue(value)

    def getProbeSupportRadiusLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1) / 2)

    def getProbeSupportRadius(self) -> Decimal:
        limits = self.getProbeSupportRadiusLimits()
        return limits.clamp(self._settings.probeSupportRadius.getValue())

    def setProbeSupportRadius(self, value: Decimal) -> None:
        self._settings.probeSupportRadius.setValue(value)

    def getProbeSupportDegreeMinimum(self) -> Decimal:
        return Decimal()

    def getProbeSupportDegree(self) -> Decimal:
        return max(self._settings.probeSupportDegree.getValue(),
                   self.getProbeSupportDegreeMinimum())

    def setProbeSupportDegree(self, value: Decimal) -> None:
        self._settings.probeSupportDegree.setValue(value)

    def getAdditionalProbePenaltyMinimum(self) -> Decimal:
        return Decimal()

    def getAdditionalProbePenalty(self) -> Decimal:
        return max(self._settings.additionalProbePenalty.getValue(),
                   self.getAdditionalProbePenaltyMinimum())

    def setAdditionalProbePenalty(self, value: Decimal) -> None:
        self._settings.additionalProbePenalty.setValue(value)
