from collections.abc import Sequence
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

logger = logging.getLogger(__name__)


class ScanAPI:

    def __init__(self, builder: StreamingScanBuilder, factory: ScanRepositoryItemFactory,
                 repository: ScanRepository, scan: SelectedScan, sizer: ScanSizer) -> None:
        self._builder = builder
        self._factory = factory
        self._repository = repository
        self._scan = scan
        self._sizer = sizer

    def insertItemIntoRepositoryFromFile(self,
                                         filePath: Path,
                                         *,
                                         simpleFileType: str = '',
                                         displayFileType: str = '') -> Optional[str]:
        itemName: Optional[str] = None
        item = self._factory.openItemFromFile(filePath,
                                              simpleFileType=simpleFileType,
                                              displayFileType=displayFileType)

        if item is None:
            logger.error(f'Unable to open scan from \"{filePath}\"!')
        else:
            itemName = self._repository.insertItem(item)

        return itemName

    def insertItemIntoRepositoryFromScan(self,
                                         nameHint: str,
                                         scan: Scan,
                                         *,
                                         filePath: Optional[Path] = None,
                                         simpleFileType: str = '',
                                         displayFileType: str = '') -> Optional[str]:
        item = self._factory.createItemFromScan(nameHint,
                                                scan,
                                                filePath=filePath,
                                                simpleFileType=simpleFileType,
                                                displayFileType=displayFileType)
        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromInitializer(self, initializerName: str) -> Optional[str]:
        itemName: Optional[str] = None
        item = self._factory.createItem(initializerName)

        if item is None:
            logger.error(f'Unknown scan initailizer \"{initializerName}\"!')
        else:
            itemName = self._repository.insertItem(item)

        return itemName

    def selectItem(self, itemName: str) -> None:
        self._scan.selectItem(itemName)

    def initializeAndActivateItem(self, initializerName: str) -> None:
        itemName = self.insertItemIntoRepositoryFromInitializer(initializerName)

        if itemName is None:
            logger.error('Failed to initialize \"{initializerName}\"!')
        else:
            self.selectItem(itemName)

    def getSelectedScan(self) -> Optional[Scan]:
        return self._scan.getSelectedItem()

    def getBoundingBoxInMeters(self) -> Box2D[Decimal]:
        box = self._sizer.getBoundingBoxInMeters()
        zero = Interval[Decimal](Decimal(), Decimal())
        return box or Box2D[Decimal](zero, zero)

    def initializeStreamingScan(self) -> None:
        self._builder.reset()

    def insertArrayTimeStamp(self, arrayIndex: int, timeStamp: float) -> None:
        self._builder.insertArrayTimeStamp(arrayIndex, timeStamp)

    def assembleScanPositionsX(self, valuesInMeters: Sequence[float],
                               timeStamps: Sequence[float]) -> None:
        self._builder.assembleScanPositionsX(valuesInMeters, timeStamps)

    def assembleScanPositionsY(self, valuesInMeters: Sequence[float],
                               timeStamps: Sequence[float]) -> None:
        self._builder.assembleScanPositionsY(valuesInMeters, timeStamps)

    def finalizeStreamingScan(self) -> None:
        scan = self._builder.build()
        itemName = self.insertItemIntoRepositoryFromScan('Stream', scan)

        if itemName is None:
            logger.error('Failed to initialize \"{name}\"!')
        else:
            self.selectItem(itemName)
