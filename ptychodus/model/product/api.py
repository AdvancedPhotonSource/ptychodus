from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging

from ptychodus.api.plugins import PluginChooser
from ptychodus.api.product import ProductFileReader, ProductFileWriter

from .object.builderFactory import ObjectBuilderFactory
from .objectRepository import ObjectRepository
from .probe.builderFactory import ProbeBuilderFactory
from .probeRepository import ProbeRepository
from .productRepository import ProductRepository
from .scan.builderFactory import ScanBuilderFactory
from .scanRepository import ScanRepository

logger = logging.getLogger(__name__)


class ScanAPI:

    def __init__(self, repository: ScanRepository, builderFactory: ScanBuilderFactory) -> None:
        self._repository = repository
        self._builderFactory = builderFactory

    def builderNames(self) -> Iterator[str]:
        return iter(self._builderFactory)

    def buildScan(self,
                  index: int,
                  builderName: str,
                  builderParameters: Mapping[str, Any] = {}) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create(builderName)
        except KeyError:
            logger.warning(f'Failed to create builder {builderName}!')
            return

        for parameterName, parameterValue in builderParameters.items():
            try:
                parameter = builder[parameterName]
            except KeyError:
                logger.warning(f'Scan builder \"{builder.getName()}\" does not have'
                               f' parameter \"{parameterName}\"!')
            else:
                parameter.setValue(parameterValue)

        item.setBuilder(builder)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._builderFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._builderFactory.getOpenFileFilter()

    def openScan(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._builderFactory.createScanFromFile(filePath, fileFilter)

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open scan {index}!')
        else:
            item.setBuilder(builder)

    def copyScan(self, sourceIndex: int, destinationIndex: int) -> None:
        logger.debug(f'Copying {sourceIndex} -> {destinationIndex}')

        try:
            sourceItem = self._repository[sourceIndex]
        except IndexError:
            logger.warning(f'Failed to access source scan {sourceIndex} for copying!')
            return

        try:
            destinationItem = self._repository[destinationIndex]
        except IndexError:
            logger.warning(f'Failed to access destination scan {destinationIndex} for copying!')
            return

        destinationItem.assign(sourceItem)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._builderFactory.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._builderFactory.getSaveFileFilter()

    def saveScan(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save scan {index}!')
        else:
            self._builderFactory.saveScan(filePath, fileFilter, item.getScan())


class ProbeAPI:

    def __init__(self, repository: ProbeRepository, builderFactory: ProbeBuilderFactory) -> None:
        self._repository = repository
        self._builderFactory = builderFactory

    def builderNames(self) -> Iterator[str]:
        return iter(self._builderFactory)

    def buildProbe(self,
                   index: int,
                   builderName: str,
                   builderParameters: Mapping[str, Any] = {}) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create(builderName)
        except KeyError:
            logger.warning(f'Failed to create builder {builderName}!')
            return

        for parameterName, parameterValue in builderParameters.items():
            try:
                parameter = builder[parameterName]
            except KeyError:
                logger.warning(f'Probe builder \"{builder.getName()}\" does not have'
                               f' parameter \"{parameterName}\"!')
            else:
                parameter.setValue(parameterValue)

        item.setBuilder(builder)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._builderFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._builderFactory.getOpenFileFilter()

    def openProbe(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._builderFactory.createProbeFromFile(filePath, fileFilter)

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open probe {index}!')
        else:
            item.setBuilder(builder)

    def copyProbe(self, sourceIndex: int, destinationIndex: int) -> None:
        logger.debug(f'Copying {sourceIndex} -> {destinationIndex}')

        try:
            sourceItem = self._repository[sourceIndex]
        except IndexError:
            logger.warning(f'Failed to access source probe {sourceIndex} for copying!')
            return

        try:
            destinationItem = self._repository[destinationIndex]
        except IndexError:
            logger.warning(f'Failed to access destination probe {destinationIndex} for copying!')
            return

        destinationItem.assign(sourceItem)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._builderFactory.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._builderFactory.getSaveFileFilter()

    def saveProbe(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save probe {index}!')
        else:
            self._builderFactory.saveProbe(filePath, fileFilter, item.getProbe())


class ObjectAPI:

    def __init__(self, repository: ObjectRepository, builderFactory: ObjectBuilderFactory) -> None:
        self._repository = repository
        self._builderFactory = builderFactory

    def builderNames(self) -> Iterator[str]:
        return iter(self._builderFactory)

    def buildObject(self,
                    index: int,
                    builderName: str,
                    builderParameters: Mapping[str, Any] = {}) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to access item {index}!')
            return

        try:
            builder = self._builderFactory.create(builderName)
        except KeyError:
            logger.warning(f'Failed to create builder {builderName}!')
            return

        for parameterName, parameterValue in builderParameters.items():
            try:
                parameter = builder[parameterName]
            except KeyError:
                logger.warning(f'Object builder \"{builder.getName()}\" does not have'
                               f' parameter \"{parameterName}\"!')
            else:
                parameter.setValue(parameterValue)

        item.setBuilder(builder)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._builderFactory.getOpenFileFilterList()

    def getOpenFileFilter(self) -> str:
        return self._builderFactory.getOpenFileFilter()

    def openObject(self, index: int, filePath: Path, fileFilter: str) -> None:
        builder = self._builderFactory.createObjectFromFile(filePath, fileFilter)

        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to open object {index}!')
        else:
            item.setBuilder(builder)

    def copyObject(self, sourceIndex: int, destinationIndex: int) -> None:
        logger.debug(f'Copying {sourceIndex} -> {destinationIndex}')

        try:
            sourceItem = self._repository[sourceIndex]
        except IndexError:
            logger.warning(f'Failed to access source object {sourceIndex} for copying!')
            return

        try:
            destinationItem = self._repository[destinationIndex]
        except IndexError:
            logger.warning(f'Failed to access destination object {destinationIndex} for copying!')
            return

        destinationItem.assign(sourceItem)

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._builderFactory.getSaveFileFilterList()

    def getSaveFileFilter(self) -> str:
        return self._builderFactory.getSaveFileFilter()

    def saveObject(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save object {index}!')
        else:
            self._builderFactory.saveObject(filePath, fileFilter, item.getObject())


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
            logger.warning(f'Refusing to create product with invalid file path \"{filePath}\"')

        return -1

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileWriterChooser.getDisplayNameList()

    def getSaveFileFilter(self) -> str:
        return self._fileWriterChooser.currentPlugin.displayName

    def saveProduct(self, index: int, filePath: Path, fileFilter: str) -> None:
        try:
            item = self._repository[index]
        except IndexError:
            logger.warning(f'Failed to save product {index}!')
            return

        self._fileWriterChooser.setCurrentPluginByName(fileFilter)
        fileType = self._fileWriterChooser.currentPlugin.simpleName
        logger.debug(f'Writing \"{filePath}\" as \"{fileType}\"')
        writer = self._fileWriterChooser.currentPlugin.strategy
        writer.write(filePath, item.getProduct())
