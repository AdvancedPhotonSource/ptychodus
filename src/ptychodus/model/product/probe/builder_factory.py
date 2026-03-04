from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
import logging

import numpy

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import ProbeFileReader, ProbeFileWriter, ProbeSequence
from ptychodus.api.probe_gen import FresnelZonePlate

from ...diffraction import DiffractionAPI
from .average_pattern import AveragePatternProbeBuilder
from .builder import FromFileProbeBuilder, ProbeSequenceBuilder
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
        rng: numpy.random.Generator,
        settings: ProbeSettings,
        diffraction_api: DiffractionAPI,
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
        file_reader_chooser: PluginChooser[ProbeFileReader],
        file_writer_chooser: PluginChooser[ProbeFileWriter],
    ) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._diffraction_api = diffraction_api
        self._fresnel_zone_plate_chooser = fresnel_zone_plate_chooser
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._builders: Mapping[str, Callable[[], ProbeSequenceBuilder]] = {
            'disk': lambda: DiskProbeBuilder(rng, settings),
            'average_pattern': self._create_average_pattern_builder,
            'fresnel_zone_plate': self._create_fresnel_zone_plate_builder,
            'rectangular': lambda: RectangularProbeBuilder(rng, settings),
            'super_gaussian': lambda: SuperGaussianProbeBuilder(rng, settings),
            'zernike': lambda: ZernikeProbeBuilder(rng, settings),
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._builders)

    def create(self, name: str) -> ProbeSequenceBuilder:
        try:
            factory = self._builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown probe builder "{name}"!') from exc

        return factory()

    def create_default(self) -> ProbeSequenceBuilder:
        return next(iter(self._builders.values()))()

    def create_from_settings(self) -> ProbeSequenceBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_probe_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired)

    def _create_average_pattern_builder(self) -> ProbeSequenceBuilder:
        return AveragePatternProbeBuilder(self._rng, self._settings, self._diffraction_api)

    def _create_fresnel_zone_plate_builder(self) -> ProbeSequenceBuilder:
        return FresnelZonePlateProbeBuilder(
            self._rng, self._settings, self._fresnel_zone_plate_chooser
        )

    def get_open_file_filters(self) -> Iterator[str]:
        for plugin in self._file_reader_chooser:
            yield plugin.display_name

    def get_open_file_filter(self) -> str:
        return self._file_reader_chooser.get_current_plugin().display_name

    def create_probe_from_file(self, file_path: Path, file_filter: str) -> ProbeSequenceBuilder:
        self._file_reader_chooser.set_current_plugin(file_filter)
        file_type = self._file_reader_chooser.get_current_plugin().simple_name
        file_reader = self._file_reader_chooser.get_current_plugin().strategy
        return FromFileProbeBuilder(self._settings, file_path, file_type, file_reader)

    def get_save_file_filters(self) -> Iterator[str]:
        for plugin in self._file_writer_chooser:
            yield plugin.display_name

    def get_save_file_filter(self) -> str:
        return self._file_writer_chooser.get_current_plugin().display_name

    def save_probe(self, file_path: Path, file_filter: str, probe: ProbeSequence) -> None:
        self._file_writer_chooser.set_current_plugin(file_filter)
        file_type = self._file_writer_chooser.get_current_plugin().simple_name
        logger.debug(f'Writing "{file_path}" as "{file_type}"')
        file_writer = self._file_writer_chooser.get_current_plugin().strategy
        file_writer.write(file_path, probe)
