from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging

from ..scanRepository import ScanRepository
from .builderFactory import ScanBuilderFactory

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
