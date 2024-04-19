from collections.abc import Sequence
from pathlib import Path
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import ProductFileReader, ProductFileWriter

from .productRepository import ProductRepository

logger = logging.getLogger(__name__)


class ProductAPI:

    def __init__(self, repository: ProductRepository,
                 fileReaderChooser: PluginChooser[ProductFileReader],
                 fileWriterChooser: PluginChooser[ProductFileWriter]) -> None:
        self._repository = repository
        self._fileReaderChooser = fileReaderChooser
        self._fileWriterChooser = fileWriterChooser

    def createNewProduct(self, name: str = 'Unnamed', *, likeIndex: int = -1) -> int:
        return self._repository.createNewProduct(name, likeIndex=likeIndex)

    def getItemName(self, productIndex: int) -> str:
        item = self._repository[productIndex]
        return item.getName()

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileReaderChooser.getDisplayNameList()

    def getOpenFileFilter(self) -> str:
        return self._fileReaderChooser.currentPlugin.displayName

    def openProduct(self, filePath: Path, fileFilter: str) -> int:
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
                return self._repository.insertProduct(product)
        else:
            logger.debug(f'Refusing to create product with invalid file path \"{filePath}\"')

        return -1

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
