from __future__ import annotations
from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging
import sys

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import Product, ProductFileReader, ProductFileWriter
from ptychodus.api.visualize import Plot2D

from ..patterns import ActiveDiffractionDataset, DiffractionDatasetSettings, PatternSizer
from .item import ProductRepositoryItem, ProductRepositoryItemObserver, ProductRepositoryObserver
from .metadataFactory import MetadataRepositoryItemFactory
from .object import ObjectRepositoryItemFactory
from .probe import ProbeRepositoryItemFactory
from .productGeometry import ProductGeometry
from .productValidator import ProductValidator
from .scan import ScanRepositoryItemFactory

logger = logging.getLogger(__name__)


class ProductRepository(Sequence[ProductRepositoryItem], ProductRepositoryItemObserver):

    def __init__(self, datasetSettings: DiffractionDatasetSettings, patternSizer: PatternSizer,
                 patterns: ActiveDiffractionDataset,
                 scanRepositoryItemFactory: ScanRepositoryItemFactory,
                 probeRepositoryItemFactory: ProbeRepositoryItemFactory,
                 objectRepositoryItemFactory: ObjectRepositoryItemFactory,
                 fileReaderChooser: PluginChooser[ProductFileReader],
                 fileWriterChooser: PluginChooser[ProductFileWriter]) -> None:
        super().__init__()
        self._patternSizer = patternSizer
        self._patterns = patterns
        self._scanRepositoryItemFactory = scanRepositoryItemFactory
        self._probeRepositoryItemFactory = probeRepositoryItemFactory
        self._objectRepositoryItemFactory = objectRepositoryItemFactory
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser
        self._itemList: list[ProductRepositoryItem] = list()
        self._metadataRepositoryItemFactory = MetadataRepositoryItemFactory(self, datasetSettings)
        self._observerList: list[ProductRepositoryObserver] = [
            self._metadataRepositoryItemFactory,  # NOTE must be first!
        ]

    @overload
    def __getitem__(self, index: int) -> ProductRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[ProductRepositoryItem]:
        ...

    def __getitem__(self,
                    index: int | slice) -> ProductRepositoryItem | Sequence[ProductRepositoryItem]:
        return self._itemList[index]

    def __len__(self) -> int:
        return len(self._itemList)

    def _insertProduct(self, item: ProductRepositoryItem) -> int:
        index = len(self._itemList)
        self._itemList.append(item)

        for observer in self._observerList:
            observer.handleItemInserted(index, item)

        return index

    def createNewProduct(self, name: str = 'Unnamed') -> int:
        metadataItem = self._metadataRepositoryItemFactory.createDefault(name)
        scanItem = self._scanRepositoryItemFactory.createDefault()
        geometry = ProductGeometry(metadataItem, scanItem, self._patternSizer)
        probeItem = self._probeRepositoryItemFactory.createDefault(geometry)
        objectItem = self._objectRepositoryItemFactory.createDefault(geometry)

        item = ProductRepositoryItem(
            parent=self,
            metadata=metadataItem,
            scan=scanItem,
            geometry=geometry,
            probe=probeItem,
            object_=objectItem,
            validator=ProductValidator(self._patterns, scanItem, geometry, probeItem, objectItem),
            costs=Plot2D.createNull(),
        )

        return self._insertProduct(item)

    def duplicateProduct(self, sourceIndex: int) -> int:
        sourceItem = self._itemList[sourceIndex]

        metadataItem = sourceItem.getMetadata().copy()
        scanItem = sourceItem.getScan().copy()
        geometry = ProductGeometry(metadataItem, scanItem, self._patternSizer)
        probeItem = sourceItem.getProbe().copy(geometry)
        objectItem = sourceItem.getObject().copy(geometry)

        item = ProductRepositoryItem(
            parent=self,
            metadata=metadataItem,
            scan=scanItem,
            geometry=geometry,
            probe=probeItem,
            object_=objectItem,
            validator=ProductValidator(self._patterns, scanItem, geometry, probeItem, objectItem),
            costs=sourceItem.getCosts().copy(),
        )

        return self._insertProduct(item)

    def insertProduct(self, product: Product) -> int:
        metadataItem = self._metadataRepositoryItemFactory.create(product.metadata)
        scanItem = self._scanRepositoryItemFactory.create(product.scan)
        geometry = ProductGeometry(metadataItem, scanItem, self._patternSizer)
        probeItem = self._probeRepositoryItemFactory.create(product.probe)
        objectItem = self._objectRepositoryItemFactory.create(product.object_)

        item = ProductRepositoryItem(
            parent=self,
            metadata=metadataItem,
            scan=scanItem,
            geometry=geometry,
            probe=probeItem,
            object_=objectItem,
            validator=ProductValidator(self._patterns, scanItem, geometry, probeItem, objectItem),
            costs=product.costs,
        )

        return self._insertProduct(item)

    def removeProduct(self, index: int) -> None:
        try:
            item = self._itemList.pop(index)
        except IndexError:
            logger.debug(f'Failed to remove product item {index}!')
            return

        for observer in self._observerList:
            observer.handleItemRemoved(index, item)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openProduct(self, filePath: Path, fileFilter: str) -> None:
        if filePath.is_file():
            self._fileReaderChooser.setCurrentPluginByName(fileFilter)
            fileType = self._fileReaderChooser.currentPlugin.simpleName
            logger.debug(f'Reading \"{filePath}\" as \"{fileType}\"')
            fileReader = self._fileReaderChooser.currentPlugin.strategy

            try:
                product = fileReader.read(filePath)
            except Exception as exc:
                raise RuntimeError(f'Failed to read \"{filePath}\"') from exc
            else:
                self.insertProduct(product)
        else:
            logger.debug(f'Refusing to create product with invalid file path \"{filePath}\"')

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveProduct(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._itemList[index]
        except IndexError:
            logger.debug(f'Failed to save product {index}!')
            return

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, item.getProduct())

    def getInfoText(self) -> str:
        sizeInMB = sum(sys.getsizeof(prod) for prod in self._itemList) / (1024 * 1024)
        return f'Total: {len(self)} [{sizeInMB:.2f}MB]'

    def addObserver(self, observer: ProductRepositoryObserver) -> None:
        if observer not in self._observerList:
            self._observerList.append(observer)

    def removeObserver(self, observer: ProductRepositoryObserver) -> None:
        try:
            self._observerList.remove(observer)
        except ValueError:
            pass

    def handleMetadataChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()

        if index < 0:
            logger.warning(f'Failed to look up index for \"{item.getName()}\"!')
            return

        for observer in self._observerList:
            observer.handleMetadataChanged(index, metadata)

    def handleScanChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        scan = item.getScan()

        if index < 0:
            logger.warning(f'Failed to look up index for \"{item.getName()}\"!')
            return

        for observer in self._observerList:
            observer.handleScanChanged(index, scan)

    def handleProbeChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        probe = item.getProbe()

        if index < 0:
            logger.warning(f'Failed to look up index for \"{item.getName()}\"!')
            return

        for observer in self._observerList:
            observer.handleProbeChanged(index, probe)

    def handleObjectChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        object_ = item.getObject()

        if index < 0:
            logger.warning(f'Failed to look up index for \"{item.getName()}\"!')
            return

        for observer in self._observerList:
            observer.handleObjectChanged(index, object_)

    def handleCostsChanged(self, item: ProductRepositoryItem) -> None:
        metadata = item.getMetadata()
        index = metadata.getIndex()
        costs = item.getCosts()

        if index < 0:
            logger.warning(f'Failed to look up index for \"{item.getName()}\"!')
            return

        for observer in self._observerList:
            observer.handleCostsChanged(index, costs)
