from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.parametric import ParameterGroup
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

from .settings import ScanSettings

logger = logging.getLogger(__name__)


class ScanBuilder(ParameterGroup):
    def __init__(self, settings: ScanSettings, name: str) -> None:
        super().__init__()
        self._name = settings.builder.copy()
        self._name.set_value(name)
        self._add_parameter('name', self._name)

    def getName(self) -> str:
        return self._name.get_value()

    def syncToSettings(self) -> None:
        for parameter in self.parameters().values():
            parameter.sync_value_to_parent()

    @abstractmethod
    def copy(self) -> ScanBuilder:
        pass

    @abstractmethod
    def build(self) -> Scan:
        pass


class FromMemoryScanBuilder(ScanBuilder):
    def __init__(self, settings: ScanSettings, points: Sequence[ScanPoint]) -> None:
        super().__init__(settings, 'from_memory')
        self._settings = settings
        self._scan = Scan(points)

    def copy(self) -> FromMemoryScanBuilder:
        return FromMemoryScanBuilder(self._settings, self._scan)

    def build(self) -> Scan:
        return self._scan


class FromFileScanBuilder(ScanBuilder):
    def __init__(
        self, settings: ScanSettings, filePath: Path, fileType: str, fileReader: ScanFileReader
    ) -> None:
        super().__init__(settings, 'from_file')
        self._settings = settings
        self.filePath = settings.filePath.copy()
        self.filePath.set_value(filePath)
        self._add_parameter('file_path', self.filePath)
        self.fileType = settings.fileType.copy()
        self.fileType.set_value(fileType)
        self._add_parameter('file_type', self.fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileScanBuilder:
        return FromFileScanBuilder(
            self._settings, self.filePath.get_value(), self.fileType.get_value(), self._fileReader
        )

    def build(self) -> Scan:
        filePath = self.filePath.get_value()
        fileType = self.fileType.get_value()
        logger.debug(f'Reading "{filePath}" as "{fileType}"')

        try:
            scan = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{filePath}"') from exc

        return scan
