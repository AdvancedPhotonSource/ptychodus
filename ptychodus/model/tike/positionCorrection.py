from __future__ import annotations

from ...api.settings import SettingsGroup, SettingsRegistry
from .adaptiveMoment import TikeAdaptiveMomentPresenter, TikeAdaptiveMomentSettings


class TikePositionCorrectionSettings(TikeAdaptiveMomentSettings):

    def __init__(self, settingsGroup: SettingsGroup) -> None:
        super().__init__(settingsGroup)
        self.usePositionCorrection = settingsGroup.createBooleanEntry(
            'UsePositionCorrection', False)
        self.usePositionRegularization = settingsGroup.createBooleanEntry(
            'UsePositionRegularization', False)

    @classmethod
    def createInstance(cls, settingsRegistry: SettingsRegistry) -> TikePositionCorrectionSettings:
        return cls(settingsRegistry.createGroup('TikePositionCorrection'))


class TikePositionCorrectionPresenter(TikeAdaptiveMomentPresenter):

    def __init__(self, settings: TikePositionCorrectionSettings) -> None:
        super().__init__(settings)

    @classmethod
    def createInstance(
            cls, settings: TikePositionCorrectionSettings) -> TikePositionCorrectionPresenter:
        presenter = cls(settings)
        return presenter

    def isPositionCorrectionEnabled(self) -> bool:
        return self._settings.usePositionCorrection.value

    def setPositionCorrectionEnabled(self, enabled: bool) -> None:
        self._settings.usePositionCorrection.value = enabled

    def isPositionRegularizationEnabled(self) -> bool:
        return self._settings.usePositionRegularization.value

    def setPositionRegularizationEnabled(self, enabled: bool) -> None:
        self._settings.usePositionRegularization.value = enabled
