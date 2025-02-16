import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.patterns import (
    DiffractionFileReader,
    DiffractionFileWriter,
)
from ptychodus.api.plugins import PluginChooser
from ptychodus.api.settings import SettingsRegistry

from .api import PatternsAPI
from .dataset import AssembledDiffractionDataset
from .settings import DetectorSettings, PatternSettings
from .sizer import PatternSizer

logger = logging.getLogger(__name__)


class PatternsCore(Observer):
    def __init__(
        self,
        settingsRegistry: SettingsRegistry,
        fileReaderChooser: PluginChooser[DiffractionFileReader],
        fileWriterChooser: PluginChooser[DiffractionFileWriter],
        reinitObservable: Observable,
    ) -> None:
        super().__init__()
        self.detectorSettings = DetectorSettings(settingsRegistry)
        self.patternSettings = PatternSettings(settingsRegistry)
        self.patternSizer = PatternSizer(self.detectorSettings, self.patternSettings)
        self.dataset = AssembledDiffractionDataset(self.patternSettings, self.patternSizer)
        self.patternsAPI = PatternsAPI(
            self.patternSettings,
            self.detectorSettings,
            self.dataset,
            fileReaderChooser,
            fileWriterChooser,
        )

        # TODO vvv refactor vvv
        fileReaderChooser.setCurrentPluginByName(self.patternSettings.fileType.getValue())
        fileWriterChooser.setCurrentPluginByName(self.patternSettings.fileType.getValue())
        # TODO ^^^^^^^^^^^^^^^^

        self._reinitObservable = reinitObservable
        reinitObservable.addObserver(self)

    def start(self) -> None:
        self.dataset.start()

    def stop(self) -> None:
        self.dataset.stop(finish_assembling=False)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self.patternsAPI.openPatterns(
                filePath=self.patternSettings.filePath.getValue(),
                fileType=self.patternSettings.fileType.getValue(),
                assemble=True,
            )
