import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.patterns import (
    BadPixelsFileReader,
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
        settings_registry: SettingsRegistry,
        bad_pixels_file_reader_chooser: PluginChooser[BadPixelsFileReader],
        file_reader_chooser: PluginChooser[DiffractionFileReader],
        file_writer_chooser: PluginChooser[DiffractionFileWriter],
        reinit_observable: Observable,
    ) -> None:
        super().__init__()
        self.detector_settings = DetectorSettings(settings_registry)
        self.pattern_settings = PatternSettings(settings_registry)
        self.pattern_sizer = PatternSizer(self.detector_settings, self.pattern_settings)
        self.dataset = AssembledDiffractionDataset(self.pattern_settings, self.pattern_sizer)
        self.patterns_api = PatternsAPI(
            self.pattern_settings,
            self.detector_settings,
            self.dataset,
            bad_pixels_file_reader_chooser,
            file_reader_chooser,
            file_writer_chooser,
        )

        bad_pixels_file_reader_chooser.synchronize_with_parameter(
            self.detector_settings.bad_pixels_file_type
        )
        file_reader_chooser.synchronize_with_parameter(self.pattern_settings.file_type)
        file_writer_chooser.set_current_plugin(self.pattern_settings.file_type.get_value())

        self._reinit_observable = reinit_observable
        reinit_observable.add_observer(self)

    def start(self) -> None:
        pass

    def stop(self) -> None:
        self.patterns_api.finish_assembling_diffraction_patterns(block=False)

    def _update(self, observable: Observable) -> None:
        if observable is self._reinit_observable:
            self.patterns_api.open_bad_pixels(
                file_path=self.detector_settings.bad_pixels_file_path.get_value(),
                file_type=self.detector_settings.bad_pixels_file_type.get_value(),
            )
            self.patterns_api.open_patterns(
                file_path=self.pattern_settings.file_path.get_value(),
                file_type=self.pattern_settings.file_type.get_value(),
            )
            self.patterns_api.start_assembling_diffraction_patterns()
            self.patterns_api.finish_assembling_diffraction_patterns(block=True)
