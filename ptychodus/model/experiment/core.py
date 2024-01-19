from ...api.experiment import ExperimentFileReader, ExperimentFileWriter
from ...api.plugins import PluginChooser
from ..metadata import MetadataBuilder
from ..object import ObjectBuilderFactory, ObjectRepositoryItemFactory
from ..patterns import PatternSizer
from ..probe import ProbeBuilderFactory, ProbeRepositoryItemFactory
from ..scan import ScanBuilderFactory, ScanRepositoryItemFactory
from .metadata import MetadataRepository
from .object import ObjectRepository
from .probe import ProbeRepository
from .repository import ExperimentRepository
from .scan import ScanRepository


class ExperimentCore:

    def __init__(self, patternSizer: PatternSizer, metadataBuilder: MetadataBuilder,
                 scanRepositoryItemFactory: ScanRepositoryItemFactory,
                 scanBuilderFactory: ScanBuilderFactory,
                 objectRepositoryItemFactory: ObjectRepositoryItemFactory,
                 objectBuilderFactory: ObjectBuilderFactory,
                 probeRepositoryItemFactory: ProbeRepositoryItemFactory,
                 probeBuilderFactory: ProbeBuilderFactory,
                 fileReaderChooser: PluginChooser[ExperimentFileReader],
                 fileWriterChooser: PluginChooser[ExperimentFileWriter]) -> None:
        self._repository = ExperimentRepository(patternSizer, scanRepositoryItemFactory,
                                                probeRepositoryItemFactory,
                                                objectRepositoryItemFactory)
        self.metadataRepository = MetadataRepository(self._repository, metadataBuilder,
                                                     fileReaderChooser, fileWriterChooser)
        self.scanRepository = ScanRepository(self._repository, scanBuilderFactory)
        self.probeRepository = ProbeRepository(self._repository, probeBuilderFactory)
        self.objectRepository = ObjectRepository(self._repository, objectBuilderFactory)
