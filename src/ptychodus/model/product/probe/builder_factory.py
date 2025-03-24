from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import (
    FresnelZonePlate,
    Probe,
    ProbeFileReader,
    ProbeFileWriter,
)

from ...patterns import AssembledDiffractionDataset
from .average_pattern import AveragePatternProbeBuilder
from .builder import FromFileProbeBuilder, ProbeBuilder
from .disk import DiskProbeBuilder
from .fzp import FresnelZonePlateProbeBuilder
from .rect import RectangularProbeBuilder
from .settings import ProbeSettings
from .super_gaussian import SuperGaussianProbeBuilder
from .zernike import ZernikeProbeBuilder

logger = logging.getLogger(__name__)


class ProbeBuilderFactory(Iterable[str]):
    def __init__(
        self,
        settings: ProbeSettings,
        dataset: AssembledDiffractionDataset,
        fresnelZonePlateChooser: PluginChooser[FresnelZonePlate],
        fileReaderChooser: PluginChooser[ProbeFileReader],
        fileWriterChooser: PluginChooser[ProbeFileWriter],
    ) -> None:
        super().__init__()
        self._settings = settings
        self._dataset = dataset
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
            raise KeyError(f'Unknown probe builder "{name}"!') from exc

        return factory()

    def create_default(self) -> ProbeBuilder:
        return next(iter(self._builders.values()))()

    def create_from_settings(self) -> ProbeBuilder:
        name = self._settings.builder.get_value()
        nameRepaired = name.casefold()

        if nameRepaired == 'from_file':
            return self.create_probe_from_file(
                self._settings.filePath.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(nameRepaired)

    def _createAveragePatternBuilder(self) -> ProbeBuilder:
        return AveragePatternProbeBuilder(self._settings, self._dataset)

    def _createFresnelZonePlateBuilder(self) -> ProbeBuilder:
        return FresnelZonePlateProbeBuilder(self._settings, self._fresnelZonePlateChooser)

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._fileReaderChooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._fileReaderChooser.get_current_plugin().display_name

    def create_probe_from_file(self, filePath: Path, fileFilter: str) -> ProbeBuilder:
        self._fileReaderChooser.set_current_plugin(fileFilter)
        fileType = self._fileReaderChooser.get_current_plugin().simple_name
        fileReader = self._fileReaderChooser.get_current_plugin().strategy
        return FromFileProbeBuilder(self._settings, filePath, fileType, fileReader)

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._fileWriterChooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._fileWriterChooser.get_current_plugin().display_name

    def save_probe(self, filePath: Path, fileFilter: str, probe: Probe) -> None:
        self._fileWriterChooser.set_current_plugin(fileFilter)
        fileType = self._fileWriterChooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{filePath}" as "{fileType}"')
        fileWriter = self._fileWriterChooser.get_current_plugin().strategy
        fileWriter.write(filePath, probe)
