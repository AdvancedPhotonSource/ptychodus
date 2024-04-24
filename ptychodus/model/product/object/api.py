from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any
import logging

from ..objectRepository import ObjectRepository
from .builderFactory import ObjectBuilderFactory

logger = logging.getLogger(__name__)


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
