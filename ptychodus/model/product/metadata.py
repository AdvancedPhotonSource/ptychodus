from collections.abc import Sequence
from pathlib import Path
from typing import overload
import logging
import sys

from ...api.product import Product, ProductFileReader, ProductFileWriter
from ...api.object import Object
from ...api.observer import ObservableSequence
from ...api.plugins import PluginChooser
from ...api.probe import Probe
from ...api.scan import Scan
from ..metadata import MetadataBuilder, MetadataRepositoryItem
from ..object import ObjectRepositoryItem
from ..probe import ProbeRepositoryItem
from ..scan import ScanRepositoryItem
from .repository import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver

logger = logging.getLogger(__name__)


class MetadataRepository(ObservableSequence[MetadataRepositoryItem], ProductRepositoryObserver):

    def __init__(self, repository: ProductRepository, metadataBuilder: MetadataBuilder,
                 fileReaderChooser: PluginChooser[ProductFileReader],
                 fileWriterChooser: PluginChooser[ProductFileWriter]) -> None:
        super().__init__()
        self._repository = repository
        self._repository.addObserver(self)
        self._metadataBuilder = metadataBuilder
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    @overload
    def __getitem__(self, index: int) -> MetadataRepositoryItem:
        ...

    @overload
    def __getitem__(self, index: slice) -> Sequence[MetadataRepositoryItem]:
        ...

    def __getitem__(
            self, index: int | slice) -> MetadataRepositoryItem | Sequence[MetadataRepositoryItem]:
        if isinstance(index, slice):
            return [item.getMetadata() for item in self._repository[index]]
        else:
            return self._repository[index].getMetadata()

    def __len__(self) -> int:
        return len(self._repository)

    def insertProduct(self, name: str) -> None:
        product = Product(
            metadata=self._metadataBuilder.build(name),
            scan=Scan(),
            probe=Probe(),
            object_=Object(),
        )
        self._repository.insertProduct(product)

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
                self._repository.insertProduct(product)
        else:
            logger.debug(f'Refusing to create product with invalid file path \"{filePath}\"')

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveProduct(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.debug(f'Failed to save product {index}!')
            return

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, item.getProduct())

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemInserted(index, item.getMetadata())

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        self.notifyObserversItemChanged(index, item)

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        self.notifyObserversItemRemoved(index, item.getMetadata())

    def getInfoText(self) -> str:
        sizeInMB = sum(sys.getsizeof(exp) for exp in self._repository) / (1024 * 1024)
        return f'Total: {len(self)} [{sizeInMB:.2f}MB]'
