import numpy

from ...api.plugins import PluginChooser
from ...api.scan import ScanFileReader, ScanFileWriter
from ...api.settings import SettingsRegistry
from ..patterns import ActiveDiffractionDataset
from .factory import ScanBuilderFactory
from .settings import ScanSettings
from .streaming import StreamingScanBuilder


class ScanCore:

    def __init__(self, rng: numpy.random.Generator, settingsRegistry: SettingsRegistry,
                 dataset: ActiveDiffractionDataset,
                 fileReaderChooser: PluginChooser[ScanFileReader],
                 fileWriterChooser: PluginChooser[ScanFileWriter]) -> None:
        self._builder = StreamingScanBuilder()  # FIXME
        self._settings = ScanSettings.createInstance(settingsRegistry)
        self._builderFactory = ScanBuilderFactory(self._settings, fileReaderChooser,
                                                  fileWriterChooser)
