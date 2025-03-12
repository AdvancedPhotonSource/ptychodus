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

        fileReaderChooser.synchronize_with_parameter(self.patternSettings.fileType)
        fileWriterChooser.set_current_plugin(self.patternSettings.fileType.get_value())

        self._reinitObservable = reinitObservable
        reinitObservable.add_observer(self)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.dataset.finish_loading(block=False)

    def _update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self.patternsAPI.open_patterns(
                filePath=self.patternSettings.filePath.get_value(),
                file_type=self.patternSettings.fileType.get_value(),
            )
            self.dataset.start_loading()
