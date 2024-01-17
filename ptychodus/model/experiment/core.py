from ...api.experiment import ExperimentFileReader, ExperimentFileWriter
from ...api.plugins import PluginChooser
from ...api.settings import SettingsRegistry
from ..object import ObjectBuilderFactory
from ..patterns import PatternSizer
from ..probe import ProbeBuilderFactory
from ..scan import ScanBuilderFactory
from .metadata import MetadataRepository
from .object import ObjectRepository
from .probe import ProbeRepository
from .repository import ExperimentRepository
from .scan import ScanRepository
from .settings import ExperimentSettings


class ExperimentCore:

    def __init__(self, settingsRegistry: SettingsRegistry, patternSizer: PatternSizer,
                 scanBuilderFactory: ScanBuilderFactory, probeBuilderFactory: ProbeBuilderFactory,
                 objectBuilderFactory: ObjectBuilderFactory,
                 fileReaderChooser: PluginChooser[ExperimentFileReader],
                 fileWriterChooser: PluginChooser[ExperimentFileWriter]) -> None:
        self.settings = ExperimentSettings.createInstance(settingsRegistry)
        self._repository = ExperimentRepository(patternSizer, fileReaderChooser, fileWriterChooser)
        self.metadataRepository = MetadataRepository(self._repository)
        self.scanRepository = ScanRepository(self._repository, scanBuilderFactory)
        self.probeRepository = ProbeRepository(self._repository, probeBuilderFactory)
        self.objectRepository = ObjectRepository(self._repository, objectBuilderFactory)
