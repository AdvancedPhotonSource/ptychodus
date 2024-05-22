from __future__ import annotations
from decimal import Decimal

from ptychodus.api.geometry import Interval
from ptychodus.api.settings import SettingsGroup, SettingsRegistry

from .adaptiveMoment import TikeAdaptiveMomentPresenter, TikeAdaptiveMomentSettings


class TikeObjectCorrectionSettings(TikeAdaptiveMomentSettings):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useObjectCorrection = settingsGroup.createBooleanEntry('UseObjectCorrection', True)
        self.positivityConstraint = settingsGroup.createRealEntry('PositivityConstraint', '0')
        self.smoothnessConstraint = settingsGroup.createRealEntry('SmoothnessConstraint', '0')
        self.useMagnitudeClipping = settingsGroup.createBooleanEntry('UseMagnitudeClipping', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeObjectCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikeObjectCorrection'))


class TikeObjectCorrectionPresenter(TikeAdaptiveMomentPresenter[TikeObjectCorrectionSettings]):

    def __init__(self, settings: TikeObjectCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(cls,
                       settings: TikeObjectCorrectionSettings) -> TikeObjectCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isObjectCorrectionEnabled(self) -> bool:
        return self._settings.useObjectCorrection.value

    def setObjectCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.useObjectCorrection.value = enabled

    def getPositivityConstraintLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1))

    def getPositivityConstraint(self) -> Decimal:
        limits = self.getPositivityConstraintLimits()
        return limits.clamp(self._settings.positivityConstraint.value)

    def setPositivityConstraint(self, value: Decimal) -> None:
        self._settings.positivityConstraint.value = value

    def getSmoothnessConstraintLimits(self) -> Interval[Decimal]:
        return Interval[Decimal](Decimal(0), Decimal(1) / 8)

    def getSmoothnessConstraint(self) -> Decimal:
        limits = self.getSmoothnessConstraintLimits()
        return limits.clamp(self._settings.smoothnessConstraint.value)

    def setSmoothnessConstraint(self, value: Decimal) -> None:
        self._settings.smoothnessConstraint.value = value

    def isMagnitudeClippingEnabled(self) -> bool:
        return self._settings.useMagnitudeClipping.value

    def setMagnitudeClippingEnabled(self, enabled: bool) -> None:
        self._settings.useMagnitudeClipping.value = enabled
