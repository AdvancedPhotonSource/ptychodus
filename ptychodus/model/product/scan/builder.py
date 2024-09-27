from __future__ import annotations
from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.parametric import ParameterGroup, PathParameter, StringParameter
from ptychodus.api.scan import Scan, ScanFileReader, ScanPoint

logger = logging.getLogger(__name__)


class ScanBuilder(ParameterGroup):
    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = StringParameter(self, 'name', name)

    def getName(self) -> str:
        return self._name.getValue()

    @abstractmethod
    def copy(self) -> ScanBuilder:
        pass

    @abstractmethod
    def build(self) -> Scan:
        pass


class FromMemoryScanBuilder(ScanBuilder):
    def __init__(self, points: Sequence[ScanPoint]) -> None:
        super().__init__('from_memory')
        self._scan = Scan(points)

    def copy(self) -> FromMemoryScanBuilder:
        return FromMemoryScanBuilder(self._scan)

    def build(self) -> Scan:
        return self._scan


class FromFileScanBuilder(ScanBuilder):
    def __init__(self, filePath: Path, fileType: str, fileReader: ScanFileReader) -> None:
        super().__init__('from_file')
        self.filePath = PathParameter(self, 'file_path', filePath)
        self.fileType = StringParameter(self, 'file_type', fileType)
        self._fileReader = fileReader

    def copy(self) -> FromFileScanBuilder:
        return FromFileScanBuilder(
            self.filePath.getValue(), self.fileType.getValue(), self._fileReader
        )

    def build(self) -> Scan:
        filePath = self.filePath.getValue()
        fileType = self.fileType.getValue()
        logger.debug(f'Reading "{filePath}" as "{fileType}"')

        try:
            scan = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read "{filePath}"') from exc

        return scan
