from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import FresnelZonePlate, Probe, ProbeFileReader, ProbeFileWriter

from ...patterns import ActiveDiffractionDataset, Detector
from .averagePattern import AveragePatternProbeBuilder
from .builder import FromFileProbeBuilder, ProbeBuilder
from .disk import DiskProbeBuilder
from .fzp import FresnelZonePlateProbeBuilder
from .rect import RectangularProbeBuilder
from .settings import ProbeSettings
from .superGaussian import SuperGaussianProbeBuilder
from .zernike import ZernikeProbeBuilder

logger = logging.getLogger(__name__)


class ProbeBuilderFactory(Iterable[str]):

    def __init__(self, settings: ProbeSettings, detector: Detector,
                 patterns: ActiveDiffractionDataset,
                 fresnelZonePlateChooser: PluginChooser[FresnelZonePlate],
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        super().__init__()
        self._settings = settings
        self._detector = detector
        self._patterns = patterns
        self._fresnelZonePlateChooser = fresnelZonePlateChooser
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._builders: Mapping[str, Callable[[], ProbeBuilder]] = {
            'disk': lambda: DiskProbeBuilder(settings),
            'average_pattern': self._createAveragePatternBuilder,
            'fresnel_zone_plate': self._createFresnelZonePlateBuilder,
            'rectangular': lambda: RectangularProbeBuilder(settings),
            'super_gaussian': lambda: SuperGaussianProbeBuilder(settings),
            'zernike': lambda: ZernikeProbeBuilder(settings),
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str) -> ProbeBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown probe builder \"{name}\"!') from exc

        return factory()

    def createDefault(self) -> ProbeBuilder:
        return next(iter(self._builders.values()))()

    def createFromSettings(self) -> ProbeBuilder:
        name = self._settings.builder.value
        nameRepaired = name.casefold()

        if nameRepaired == 'from_file':
            return self.createProbeFromFile(
                self._settings.filePath.value,
                self._settings.fileType.value,
            )

        return self.create(nameRepaired)

    def _createAveragePatternBuilder(self) -> ProbeBuilder:
        return AveragePatternProbeBuilder(self._detector, self._patterns)

    def _createFresnelZonePlateBuilder(self) -> ProbeBuilder:
        return FresnelZonePlateProbeBuilder(self._settings, self._fresnelZonePlateChooser)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def createProbeFromFile(self, filePath: Path, fileFilter: str) -> ProbeBuilder:
        self._fileReaderChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileReaderChooser.currentPlugin.simpleName
        fileReader = self._fileReaderChooser.currentPlugin.strategy
        return FromFileProbeBuilder(filePath, fileType, fileReader)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveProbe(self, filePath: Path, fileFilter: str, probe: Probe) -> None:
        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        fileWriter = self._fileWriterChooser.currentPlugin.strategy
        fileWriter.write(filePath, probe)
