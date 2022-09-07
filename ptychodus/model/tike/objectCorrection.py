from __future__ import annotations
from decimal import Decimal

from ...api.settings import SettingsGroup, SettingsRegistry
from .adaptiveMoment import TikeAdaptiveMomentPresenter, TikeAdaptiveMomentSettings


class TikeObjectCorrectionSettings(TikeAdaptiveMomentSettings):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.useObjectCorrection = settingsGroup.createBooleanEntry('UseObjectCorrection', True)
        self.positivityConstraint = settingsGroup.createRealEntry('PositivityConstraint', '0')
        self.smoothnessConstraint = settingsGroup.createRealEntry('SmoothnessConstraint', '0')

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikeObjectCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikeObjectCorrection'))


class TikeObjectCorrectionPresenter(TikeAdaptiveMomentPresenter):

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

    def getMinPositivityConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxPositivityConstraint(self) -> Decimal:
        return Decimal(1)

    def getPositivityConstraint(self) -> Decimal:
        return self._clamp(self._settings.positivityConstraint.value,
                           self.getMinPositivityConstraint(), self.getMaxPositivityConstraint())

    def setPositivityConstraint(self, value: Decimal) -> None:
        self._settings.positivityConstraint.value = value

    def getMinSmoothnessConstraint(self) -> Decimal:
        return Decimal(0)

    def getMaxSmoothnessConstraint(self) -> Decimal:
        return Decimal('0.125')

    def getSmoothnessConstraint(self) -> Decimal:
        return self._clamp(self._settings.smoothnessConstraint.value,
                           self.getMinSmoothnessConstraint(), self.getMaxSmoothnessConstraint())

    def setSmoothnessConstraint(self, value: Decimal) -> None:
        self._settings.smoothnessConstraint.value = value
