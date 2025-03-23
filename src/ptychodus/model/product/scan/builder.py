from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.scan import PositionSequence, PositionFileReader, ScanPoint

from .settings import ScanSettings

logger = logging.getLogger(__name__)


class ScanBuilder(ParameterGroup):
    def __init__(self, settings: ScanSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

    def get_name(self) -> str:
        return self._name.get_value()

    def sync_to_settings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    @abstractmethod
    def copy(self) -> ScanBuilder:
        pass

    @abstractmethod
    def build(self) -> PositionSequence:
        pass


class FromMemoryScanBuilder(ScanBuilder):
    def __init__(self, settings: ScanSettings, points: Sequence[ScanPoint]) -> None:
        super().__init__(settings, 'from_memory')
        self._settings = settings
        self._scan = PositionSequence(points)

    def copy(self) -> FromMemoryScanBuilder:
        return FromMemoryScanBuilder(self._settings, self._scan)

    def build(self) -> PositionSequence:
        return self._scan


class FromFileScanBuilder(ScanBuilder):
    def __init__(
        self,
        settings: ScanSettings,
        file_path: Path,
        file_type: str,
        file_reader: PositionFileReader,
    ) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self.file_path = settings.file_path.copy()
        self.file_path.set_value(file_path)
        self._add_parameter('file_path', self.file_path)
        self.file_type = settings.file_type.copy()
        self.file_type.set_value(file_type)
        self._add_parameter('file_type', self.file_type)
        self._file_reader = file_reader

    def copy(self) -> FromFileScanBuilder:
        return FromFileScanBuilder(
            self._settings,
            self.file_path.get_value(),
            self.file_type.get_value(),
            self._file_reader,
        )

    def build(self) -> PositionSequence:
        file_path = self.file_path.get_value()
        file_type = self.file_type.get_value()
        logger.debug(f'Reading "{file_path}" as "{file_type}"')

        try:
            scan = self._file_reader.read(file_path)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{file_path}"') from exc

        return scan
