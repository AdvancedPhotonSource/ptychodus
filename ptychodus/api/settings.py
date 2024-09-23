from __future__ import annotations
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, TypeVar
from uuid import UUID
import configparser
import logging

from .observer import Observable, Observer
from .parametric import (BooleanParameter, DecimalParameter, IntegerParameter, Parameter,
                         ParameterRepository, PathParameter, StringParameter, UUIDParameter)

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathPrefixChange:
    findPathPrefix: Path
    replacementPathPrefix: Path


class SettingsGroup(Mapping[str, Parameter[Any]], Observable, Observer):

    def __init__(self, name: str) -> None:
        super().__init__()
        self._repository = ParameterRepository(name)
        self._repository.addObserver(self)

    @property
    def name(self) -> str:
        return self._repository.repositoryName

    def createStringEntry(self, name: str, defaultValue: str) -> StringParameter:
        return self._repository._registerStringParameter(name, defaultValue)

    def createPathEntry(self, name: str, defaultValue: Path) -> PathParameter:
        return self._repository._registerPathParameter(name, defaultValue)

    def createUUIDEntry(self, name: str, defaultValue: UUID) -> UUIDParameter:
        return self._repository._registerUUIDParameter(name, defaultValue)

    def createBooleanEntry(self, name: str, defaultValue: bool) -> BooleanParameter:
        return self._repository._registerBooleanParameter(name, defaultValue)

    def createIntegerEntry(self, name: str, defaultValue: int) -> IntegerParameter:
        return self._repository._registerIntegerParameter(name, defaultValue)

    def createRealEntry(self, name: str, defaultValue: str | Decimal) -> DecimalParameter:
        parameter = DecimalParameter(defaultValue)
        self._repository._registerParameter(name, parameter)
        return parameter

    def __iter__(self) -> Iterator[str]:
        return iter(self._repository)

    def __getitem__(self, name: str) -> Parameter[Any]:
        return self._repository[name]

    def __len__(self) -> int:
        return len(self._repository)

    def update(self, observable: Observable) -> None:
        if observable is self._repository:
            self.notifyObservers()


class SettingsRegistry(Observable):

    def __init__(self) -> None:
        super().__init__()
        self._groupList: list[SettingsGroup] = list()
        self._fileFilterList: list[str] = ['Initialization Files (*.ini)']

    def createGroup(self, name: str) -> SettingsGroup:
        for existingGroup in self._groupList:
            if name.casefold() == existingGroup.name.casefold():
                return existingGroup

        group = SettingsGroup(name)
        self._groupList.append(group)
        self._groupList.sort(key=lambda group: group.name)
        return group

    def __iter__(self) -> Iterator[SettingsGroup]:
        return iter(self._groupList)

    def __getitem__(self, index: int) -> SettingsGroup:
        return self._groupList[index]

    def __len__(self) -> int:
        return len(self._groupList)

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getOpenFileFilter(self) -> str:
        return self._fileFilterList[0]

    def openSettings(self, filePath: Path) -> None:
        config = configparser.ConfigParser(interpolation=None)
        logger.debug(f'Reading settings from \"{filePath}\"')
        config.read(filePath)

        for settingsGroup in self._groupList:
            if not config.has_section(settingsGroup.name):
                continue

            for name, parameter in settingsGroup.items():
                if config.has_option(settingsGroup.name, name):
                    valueString = config.get(settingsGroup.name, name)
                    parameter.setValueFromString(valueString)

        self.notifyObservers()

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getSaveFileFilter(self) -> str:
        return self._fileFilterList[0]

    def saveSettings(self,
                     filePath: Path,
                     changePathPrefix: PathPrefixChange | None = None) -> None:
        config = configparser.ConfigParser(interpolation=None)
        setattr(config, 'optionxform', lambda option: option)

        for settingsGroup in self._groupList:
            config.add_section(settingsGroup.name)

            for name, parameter in settingsGroup.items():
                valueString = str(parameter)

                if changePathPrefix and isinstance(parameter, PathParameter):
                    modifiedPath = parameter.changePathPrefix(
                        changePathPrefix.findPathPrefix, changePathPrefix.replacementPathPrefix)
                    valueString = str(modifiedPath)

                config.set(settingsGroup.name, name, valueString)

        logger.debug(f'Writing settings to \"{filePath}\"')

        with filePath.open(mode='w') as configFile:
            config.write(configFile)
