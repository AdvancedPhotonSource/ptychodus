from pathlib import Path
from typing import Optional
import logging

from ...api.scan import TabularScan
from .active import ActiveScan
from .itemFactory import ScanRepositoryItemFactory
from .itemRepository import ScanRepository
from .streaming import StreamingScanBuilder
from .tabular import ScanFileInfo

logger = logging.getLogger(__name__)


class ScanAPI:

    def __init__(self, builder: StreamingScanBuilder, factory: ScanRepositoryItemFactory,
                 repository: ScanRepository, scan: ActiveScan) -> None:
        self._builder = builder
        self._factory = factory
        self._repository = repository
        self._scan = scan

    def insertScanIntoRepositoryFromFile(self, filePath: Path, fileFilter: str) -> list[str]:
        itemNameList: list[str] = list()
        initializerList = self._factory.openScan(filePath, fileFilter)

        for initializer in initializerList:
            itemName = self._repository.insertItem(initializer)
            itemNameList.append(itemName)

        return itemNameList

    def insertScanIntoRepositoryFromInitializer(self, initializerName: str) -> Optional[str]:
        itemName: Optional[str] = None
        scan = self._factory.createItem(initializerName)

        if scan is None:
            logger.error(f'Unknown scan initializer \"{initializerName}\"!')
        else:
            itemName = self._repository.insertItem(scan)

        return itemName

    def insertScanIntoRepository(self, scan: TabularScan, fileInfo: Optional[ScanFileInfo]) -> str:
        item = self._factory.createTabularItem(scan, fileInfo)
        return self._repository.insertItem(item)

    def setActiveScan(self, name: str) -> None:
        self._scan.setActiveScan(name)

    def initializeStreamingScan(self) -> None:
        self._builder.reset()

    def insertArrayTimeStamp(self, arrayIndex: int, timeStamp: float) -> None:
        self._builder.insertArrayTimeStamp(arrayIndex, timeStamp)

    def assembleScanPositionsX(self, valuesInMeters: list[float], timeStamps: list[float]) -> None:
        self._builder.assembleScanPositionsX(valuesInMeters, timeStamps)

    def assembleScanPositionsY(self, valuesInMeters: list[float], timeStamps: list[float]) -> None:
        self._builder.assembleScanPositionsY(valuesInMeters, timeStamps)

    def finalizeStreamingScan(self) -> None:
        scan = self._builder.build()
        itemName = self.insertScanIntoRepository(scan, ScanFileInfo.createNull())
        self.setActiveScan(itemName)
