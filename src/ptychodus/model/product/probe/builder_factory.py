from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
import logging

import numpy

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.probe import ProbeFileReader, ProbeFileWriter, ProbeSequence
from ptychodus.api.probe_gen import FresnelZonePlate

from ...diffraction import AssembledDiffractionDataset
from .average_pattern import AveragePatternProbeBuilder
from .builder import FromFileProbeBuilder, ProbeSequenceBuilder
from .disk import DiskProbeBuilder
from .fzp import FresnelZonePlateProbeBuilder
from .hermite import HermiteProbeBuilder
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
        fresnel_zone_plate_chooser: PluginChooser[FresnelZonePlate],
        file_reader_chooser: PluginChooser[ProbeFileReader],
        file_writer_chooser: PluginChooser[ProbeFileWriter],
    ) -> None:
        super().__init__()
        self._rng = rng
        self._settings = settings
        self._fresnel_zone_plate_chooser = fresnel_zone_plate_chooser
        self._file_reader_chooser = file_reader_chooser
        self._file_writer_chooser = file_writer_chooser
        self._non_diffraction_builders: Mapping[str, Callable[[], ProbeSequenceBuilder]] = {
            'disk': lambda: DiskProbeBuilder(rng, settings),
            'fresnel_zone_plate': self._create_fresnel_zone_plate_builder,
            'hermite': lambda: HermiteProbeBuilder(rng, settings),
            'rectangular': lambda: RectangularProbeBuilder(rng, settings),
            'super_gaussian': lambda: SuperGaussianProbeBuilder(rng, settings),
            'zernike': lambda: ZernikeProbeBuilder(rng, settings),
        }
        self._diffraction_builders: Mapping[
            str, Callable[[AssembledDiffractionDataset], ProbeSequenceBuilder]
        ] = {
            'average_pattern': lambda dataset: AveragePatternProbeBuilder(rng, settings, dataset),
        }

    def __iter__(self) -> Iterator[str]:
        yield from self._non_diffraction_builders
        yield from self._diffraction_builders

    def create(
        self, name: str, *, dataset: AssembledDiffractionDataset | None = None
    ) -> ProbeSequenceBuilder:
        diffraction_factory = self._diffraction_builders.get(name)
        if diffraction_factory is not None:
            if dataset is None:
                raise RuntimeError(
                    f'Probe builder "{name}" requires an associated diffraction dataset.'
                )
            return diffraction_factory(dataset)

        try:
            factory = self._non_diffraction_builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown probe builder "{name}"!') from exc
        return factory()

    def create_default(self) -> ProbeSequenceBuilder:
        return next(iter(self._non_diffraction_builders.values()))()

    def create_from_settings(
        self, *, dataset: AssembledDiffractionDataset | None = None
    ) -> ProbeSequenceBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_probe_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired, dataset=dataset)

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
        file_reader = self._file_reader_chooser.get_current_plugin().strategy

        builder = FromFileProbeBuilder(self._settings, file_reader)
        builder.file_path.set_value(file_path)
        builder.file_type.set_value(self._file_reader_chooser.get_current_plugin().simple_name)
        return builder

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
