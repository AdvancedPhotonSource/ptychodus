from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

from ...api.parametric import ParametricBase
from ...api.scan import Scan, ScanFileReader, ScanPoint

logger = logging.getLogger(__name__)


class ScanBuilder(ParametricBase):

    @abstractmethod
    def build(self) -> Scan:
        pass


class FromMemoryScanBuilder(ScanBuilder):

    def __init__(self, pointSeq: Sequence[ScanPoint]) -> None:
        super().__init__('From Memory')
        self._pointList = list(pointSeq)

    def append(self, point: ScanPoint) -> None:
        self._pointList.append(point)

    def build(self) -> Scan:
        self._pointList.sort(key=lambda point: point.index)
        return Scan(self._pointList)


class FromFileScanBuilder(ScanBuilder):

    def __init__(self, filePath: Path, fileType: str, fileReader: ScanFileReader) -> None:
        super().__init__('From File')
        self._filePath = self._registerPathParameter('FilePath', filePath)
        self._fileType = self._registerStringParameter('FileType', fileType)
        self._fileReader = fileReader

    def build(self) -> Scan:
        filePath = self._filePath.getValue()
        fileType = self._fileType.getValue()
        logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')

        try:
            scan = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{filePath}\"') from exc

        return scan
