from decimal import Decimal
from pathlib import Path
from typing import Optional
import logging

from ...api.geometry import Box2D, Interval
from ...api.scan import Scan, TabularScan
from .factory import ScanRepositoryItemFactory
from .repository import ScanRepository
from .selected import SelectedScan
from .sizer import ScanSizer
from .streaming import StreamingScanBuilder
from .tabular import ScanFileInfo

logger = logging.getLogger(__name__)


class ScanAPI:

    def __init__(self, builder: StreamingScanBuilder, factory: ScanRepositoryItemFactory,
                 repository: ScanRepository, scan: SelectedScan, sizer: ScanSizer) -> None:
        self._builder = builder
        self._factory = factory
        self._repository = repository
        self._scan = scan
        self._sizer = sizer

    def insertScanIntoRepositoryFromFile(self, filePath: Path, fileFilter: str) -> list[str]:
        itemNameList: list[str] = list()
        itemList = self._factory.openScan(filePath, fileFilter)

        for item in itemList:
            itemName = self._repository.insertItem(item)
            itemNameList.append(itemName)

        return itemNameList

    def insertScanIntoRepositoryFromInitializer(self, initializerName: str) -> list[str]:
        itemNameList: list[str] = list()
        itemList = self._factory.createItem(initializerName)

        for item in itemList:
            itemName = self._repository.insertItem(item)
            itemNameList.append(itemName)

        return itemNameList

    def insertScanIntoRepository(self, scan: TabularScan, fileInfo: Optional[ScanFileInfo]) -> str:
        item = self._factory.createTabularItem(scan, fileInfo)
        return self._repository.insertItem(item)

    def getSelectedScan(self) -> Scan:
        return self._scan.getSelectedItem()

    def selectScan(self, name: str) -> None:
        self._scan.selectItem(name)

    def getBoundingBoxInMeters(self) -> Box2D[Decimal]:
        box = self._sizer.getBoundingBoxInMeters()
        zero = Interval[Decimal](Decimal(), Decimal())
        return box or Box2D[Decimal](zero, zero)

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
        self.selectScan(itemName)
