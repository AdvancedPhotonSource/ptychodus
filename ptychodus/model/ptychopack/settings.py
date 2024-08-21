from ptychodus.api.observer import Observable, Observer
from ptychodus.api.settings import SettingsRegistry


class PtychoPackSettings(Observable, Observer):

    def __init__(self, registry: SettingsRegistry) -> None:
        super().__init__()
        self._settingsGroup = registry.createGroup('PtychoPack')
        self._settingsGroup.addObserver(self)

        self.objectTuningParameter = self._settingsGroup.createRealEntry(
            'ObjectTuningParameter', '1.')
        self.objectAlpha = self._settingsGroup.createRealEntry('ObjectAlpha', '0.05')
        self.probeTuningParameter = self._settingsGroup.createRealEntry(
            'ProbeTuningParameter', '1.')
        self.probeAlpha = self._settingsGroup.createRealEntry('ProbeAlpha', '0.8')
        self.positionCorrectionProbeThreshold = self._settingsGroup.createRealEntry(
            'PositionCorrectionProbeThreshold', '0.1')
        self.positionCorrectionFeedbackParameter = self._settingsGroup.createRealEntry(
            'PositionCorrectionFeedbackParameter', '0.')

    def update(self, observable: Observable) -> None:
        if observable is self._settingsGroup:
            self.notifyObservers()
