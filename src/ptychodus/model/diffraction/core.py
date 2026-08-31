import logging

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.diffraction import (
    BadPixelsFileReader,
    DiffractionFileReader,
    DiffractionFileWriter,
)
from ptychodus.api.plugins import PluginChooser, PluginChooserParameter
from ptychodus.api.settings import SettingsRegistry

from ..task_manager import TaskManager
from .api import DiffractionAPI
from .monitor import DiffractionTaskMonitor
from .repository import DiffractionDatasetRepository, build_default_factory
from .settings import DetectorSettings, DiffractionSettings

logger = logging.getLogger(__name__)


class DiffractionCore(Observer):
    def __init__(
        self,
        task_manager: TaskManager,
        settings_registry: SettingsRegistry,
        bad_pixels_file_reader_chooser: PluginChooser[BadPixelsFileReader],
        file_reader_chooser: PluginChooser[DiffractionFileReader],
        file_writer_chooser: PluginChooser[DiffractionFileWriter],
        reinit_observable: Observable,
    ) -> None:
        super().__init__()
        self.detector_settings = DetectorSettings(settings_registry)
        self.diffraction_settings = DiffractionSettings(settings_registry)
        self.task_monitor = DiffractionTaskMonitor(task_manager)
        self.repository = DiffractionDatasetRepository(
            factory=build_default_factory(
                self.diffraction_settings,
                self.detector_settings,
                task_manager,
                self.task_monitor,
            )
        )
        # Display-name views of each reader chooser, bound to the settings parameter
        # that persists the selection. These are what the GUI binds combo boxes to.
        self.bad_pixels_file_reader_parameter = PluginChooserParameter(
            bad_pixels_file_reader_chooser, self.detector_settings.bad_pixels_file_type
        )
        self.file_reader_parameter = PluginChooserParameter(
            file_reader_chooser, self.diffraction_settings.file_type
        )

        self.diffraction_api = DiffractionAPI(
            self.diffraction_settings,
            self.repository,
            bad_pixels_file_reader_chooser,
            file_reader_chooser,
            file_writer_chooser,
            self.bad_pixels_file_reader_parameter,
            self.file_reader_parameter,
        )

        # Deliberately unbound: the writer shares file_type with the reader above, so
        # binding both would make them fight over the same parameter.
        file_writer_chooser.set_current_plugin(self.diffraction_settings.file_type.get_value())

        self._reinit_observable = reinit_observable
        reinit_observable.add_observer(self)

    def _update(self, observable: Observable) -> None:
        if observable is self._reinit_observable:
            # BadPixelsFilePath defaults to a placeholder, so honoring the enable
            # flag here is what keeps a settings file that never opted in from
            # failing the load outright. The GUI wizard gates on the same flag.
            bad_pixels_file_path = (
                self.detector_settings.bad_pixels_file_path.get_value()
                if self.detector_settings.bad_pixels_enabled.get_value()
                else None
            )
            dataset_index = self.diffraction_api.open_patterns(
                file_path=self.diffraction_settings.file_path.get_value(),
                file_type=self.diffraction_settings.file_type.get_value(),
                bad_pixels_file_path=bad_pixels_file_path,
                bad_pixels_file_type=self.detector_settings.bad_pixels_file_type.get_value(),
            )

            if dataset_index >= 0:
                self.diffraction_api.load_all_arrays(dataset_index=dataset_index)
