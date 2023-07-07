from collections.abc import Sequence
from pathlib import Path
from typing import Optional
import logging

from ...api.scan import Scan
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
                                         displayFileType: str = '',
                                         selectItem: bool = False) -> Optional[str]:
        item = self._factory.openItemFromFile(filePath,
                                              simpleFileType=simpleFileType,
                                              displayFileType=displayFileType)

        if item is None:
            logger.error(f'Failed to open scan from \"{filePath}\"!')

        itemName = self._repository.insertItem(item)

        if itemName is None:
            logger.error('Failed to insert scan!')
        elif selectItem:
            self._scan.selectItem(itemName)

        return itemName

    def insertItemIntoRepositoryFromScan(self,
                                         name: str,
                                         scan: Scan,
                                         *,
                                         filePath: Optional[Path] = None,
                                         simpleFileType: str = '',
                                         displayFileType: str = '',
                                         replaceItem: bool = False,
                                         selectItem: bool = False) -> Optional[str]:
        item = self._factory.createItemFromScan(name,
                                                scan,
                                                filePath=filePath,
                                                simpleFileType=simpleFileType,
                                                displayFileType=displayFileType)
        itemName = self._repository.insertItem(item, name=name if replaceItem else None)

        if itemName is None:
            logger.error(f'Failed to insert tabular scan \"{name}\"!')
        elif selectItem:
            self._scan.selectItem(itemName)

        return itemName

    def insertItemIntoRepositoryFromInitializerSimpleName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromSimpleName(name)
        return self._repository.insertItem(item)

    def insertItemIntoRepositoryFromInitializerDisplayName(self, name: str) -> Optional[str]:
        item = self._factory.createItemFromDisplayName(name)
        return self._repository.insertItem(item)

    def getSelectedScan(self) -> Optional[Scan]:
        return self._scan.getSelectedItem()

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
        self.insertItemIntoRepositoryFromScan('Stream', scan, selectItem=True)
