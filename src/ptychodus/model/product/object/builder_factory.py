from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
import logging

import numpy

from ptychodus.api.object import Object, ObjectFileReader, ObjectFileWriter
from ptychodus.api.plugins import PluginChooser

from ...diffraction import AssembledDiffractionDataset
from ...file_types import FileTypeChooser
from .builder import FromFileObjectBuilder, ObjectBuilder
from .dead_leaves import DeadLeavesObjectBuilder
from .fractal_noise import FractalNoiseObjectBuilder
from .grf import GaussianRandomFieldObjectBuilder
from .paganin import PaganinObjectBuilder
from .random import RandomObjectBuilder
from .settings import ObjectSettings
from .siemens_star import SiemensStarObjectBuilder
from .stxm_dpc import STXMDPCObjectBuilder

logger = logging.getLogger(__name__)


class ObjectBuilderFactory(Iterable[str]):
    def __init__(
        self,
        rng: numpy.random.Generator,
        settings: ObjectSettings,
        file_reader_chooser: PluginChooser[ObjectFileReader],
        file_writer_chooser: PluginChooser[ObjectFileWriter],
    ) -> None:
        self._rng = rng
        self._settings = settings
        self.file_readers = FileTypeChooser(file_reader_chooser)
        self.file_writers = FileTypeChooser(file_writer_chooser)
        self._non_diffraction_builders: Mapping[str, Callable[[], ObjectBuilder]] = {
            'random': lambda: RandomObjectBuilder(rng, settings),
            'dead_leaves': lambda: DeadLeavesObjectBuilder(rng, settings),
            'fractal_noise': lambda: FractalNoiseObjectBuilder(rng, settings),
            'grf': lambda: GaussianRandomFieldObjectBuilder(rng, settings),
            'siemens_star': lambda: SiemensStarObjectBuilder(settings),
        }
        self._diffraction_builders: Mapping[
            str, Callable[[AssembledDiffractionDataset], ObjectBuilder]
        ] = {
            'stxm_dpc': lambda dataset: STXMDPCObjectBuilder(settings, dataset),
            'paganin': lambda dataset: PaganinObjectBuilder(settings, dataset),
        }

    def __iter__(self) -> Iterator[str]:
        yield from self._non_diffraction_builders
        yield from self._diffraction_builders

    def create(
        self, name: str, *, dataset: AssembledDiffractionDataset | None = None
    ) -> ObjectBuilder:
        diffraction_factory = self._diffraction_builders.get(name)
        if diffraction_factory is not None:
            if dataset is None:
                raise RuntimeError(
                    f'Object builder "{name}" requires an associated diffraction dataset.'
                )
            return diffraction_factory(dataset)

        try:
            factory = self._non_diffraction_builders[name]
        except KeyError as exc:
            raise KeyError(f'Unknown object builder "{name}"!') from exc
        return factory()

    def create_default(self) -> ObjectBuilder:
        return next(iter(self._non_diffraction_builders.values()))()

    def create_from_settings(
        self, *, dataset: AssembledDiffractionDataset | None = None
    ) -> ObjectBuilder:
        name = self._settings.builder.get_value()
        name_repaired = name.casefold()

        if name_repaired == 'from_file':
            return self.create_object_from_file(
                self._settings.file_path.get_value(),
                self._settings.file_type.get_value(),
            )

        return self.create(name_repaired, dataset=dataset)

    def create_object_from_file(self, file_path: Path, file_type: str) -> ObjectBuilder:
        plugin = self.file_readers.select(file_type)

        builder = FromFileObjectBuilder(self._settings, plugin.strategy)
        builder.file_path.set_value(file_path)
        builder.file_type.set_value(plugin.simple_name)
        return builder

    def save_object(self, file_path: Path, file_type: str, object_: Object) -> None:
        plugin = self.file_writers.select(file_type)
        logger.debug(f'Writing "{file_path}" as "{plugin.simple_name}"')
        plugin.strategy.write(file_path, object_)
