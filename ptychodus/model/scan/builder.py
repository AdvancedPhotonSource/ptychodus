from abc import abstractmethod
from collections.abc import Sequence
from pathlib import Path
import logging

from ...api.parametric import ParameterRepository
from ...api.scan import Scan, ScanFileReader, ScanPoint
from .metrics import ScanMetrics

logger = logging.getLogger(__name__)


class ScanBuilder(ParameterRepository):

    def __init__(self, name: str) -> None:
        super().__init__('Builder')
        self._name = self._registerStringParameter('Name', name)

    @abstractmethod
    def build(self) -> Scan:
        pass

    @abstractmethod
    def getScanMetrics(self) -> ScanMetrics:
        pass


class FromMemoryScanBuilder(ScanBuilder):

    def __init__(self, points: Sequence[ScanPoint]) -> None:
        super().__init__('From Memory')
        self._scan = Scan(points)
        self._metrics = ScanMetrics.createFromPoints(points)

    def build(self) -> Scan:
        return self._scan

    def getScanMetrics(self) -> ScanMetrics:
        return self._metrics


class FromFileScanBuilder(ScanBuilder):

    def __init__(self, filePath: Path, fileType: str, fileReader: ScanFileReader) -> None:
        super().__init__('From File')
        self.filePath = self._registerPathParameter('FilePath', filePath)
        self.fileType = self._registerStringParameter('FileType', fileType)
        self._fileReader = fileReader

    def build(self) -> Scan:
        filePath = self.filePath.getValue()
        fileType = self.fileType.getValue()
        logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')

        try:
            scan = self._fileReader.read(filePath)
        except Exception as exc:
            raise RuntimeError(f'Failed to read \"{filePath}\"') from exc

        return scan
