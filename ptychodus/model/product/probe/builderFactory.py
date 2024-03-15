from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import (FresnelZonePlate, Probe, ProbeFileReader, ProbeFileWriter,
                                 ProbeGeometryProvider)

from ...patterns import ActiveDiffractionDataset
from .average import AverageProbeBuilder
from .builder import FromFileProbeBuilder, ProbeBuilder
from .disk import DiskProbeBuilder
from .fzp import FresnelZonePlateProbeBuilder
from .rect import RectangularProbeBuilder
from .superGaussian import SuperGaussianProbeBuilder
from .zernike import ZernikeProbeBuilder

logger = logging.getLogger(__name__)


class ProbeBuilderFactory(Iterable[str]):

    def __init__(self, patterns: ActiveDiffractionDataset,
                 fresnelZonePlateChooser: PluginChooser[FresnelZonePlate],
                 fileReaderChooser: PluginChooser[ProbeFileReader],
                 fileWriterChooser: PluginChooser[ProbeFileWriter]) -> None:
        super().__init__()
        self._patterns = patterns
        self._fresnelZonePlateChooser = fresnelZonePlateChooser
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._builders: Mapping[str, Callable[[ProbeGeometryProvider], ProbeBuilder]] = {
            'Average': self._createAverageBuilder,
            'Disk': self._createDiskBuilder,
            'Fresnel Zone Plate': self._createFZPBuilder,
            'Rectangular': self._createRectangularBuilder,
            'SuperGaussian': self._createSuperGaussianBuilder,
            'Zernike': self._createZernikeBuilder,
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str, geometryProvider: ProbeGeometryProvider) -> ProbeBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown probe builder \"{name}\"!') from exc

        return factory(geometryProvider)

    def _createAverageBuilder(self, geometryProvider: ProbeGeometryProvider) -> ProbeBuilder:
        return AverageProbeBuilder(self._patterns, geometryProvider)

    def _createDiskBuilder(self, geometryProvider: ProbeGeometryProvider) -> ProbeBuilder:
        return DiskProbeBuilder(geometryProvider)

    def _createFZPBuilder(self, geometryProvider: ProbeGeometryProvider) -> ProbeBuilder:
        return FresnelZonePlateProbeBuilder(geometryProvider, self._fresnelZonePlateChooser)

    def _createRectangularBuilder(self, geometryProvider: ProbeGeometryProvider) -> ProbeBuilder:
        return RectangularProbeBuilder(geometryProvider)

    def _createSuperGaussianBuilder(self, geometryProvider: ProbeGeometryProvider) -> ProbeBuilder:
        return SuperGaussianProbeBuilder(geometryProvider)

    def _createZernikeBuilder(self, geometryProvider: ProbeGeometryProvider) -> ProbeBuilder:
        return ZernikeProbeBuilder(geometryProvider)

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
