from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Generic, TypeVar
from uuid import UUID
import configparser
import logging

from .observer import Observable, Observer
from .parametric import (BooleanParameter, DecimalParameter, IntegerParameter, Parameter,
                         PathParameter, StringParameter, UUIDParameter)

T = TypeVar('T')

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathPrefixChange:
    findPathPrefix: Path
    replacementPathPrefix: Path


class SettingsEntry(Generic[T], Observable, Observer):

    def __init__(self, name: str, parameter: Parameter[Any]) -> None:
        super().__init__()
        self._name = name
        self._parameter = parameter
        self._parameter.addObserver(self)

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> T:
        return self._parameter.getValue()

    @value.setter
    def value(self, value: T) -> None:
        self._parameter.setValue(value)

    def setValueFromString(self, valueString: str) -> None:
        self._parameter.setValueFromString(valueString)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.notifyObservers()


class SettingsGroup(Observable, Observer):

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._entryList: list[SettingsEntry[Any]] = list()

    @property
    def name(self) -> str:
        return self._name

    def createStringEntry(self, name: str, defaultValue: str) -> SettingsEntry[str]:
        parameter = StringParameter(defaultValue)
        candidateEntry = SettingsEntry[str](name, parameter)
        return self._registerEntryIfNonexistent(candidateEntry)

    def createPathEntry(self, name: str, defaultValue: Path) -> SettingsEntry[Path]:
        parameter = PathParameter(defaultValue)
        candidateEntry = SettingsEntry[Path](name, parameter)
        return self._registerEntryIfNonexistent(candidateEntry)

    def createUUIDEntry(self, name: str, defaultValue: UUID) -> SettingsEntry[UUID]:
        parameter = UUIDParameter(defaultValue)
        candidateEntry = SettingsEntry[UUID](name, parameter)
        return self._registerEntryIfNonexistent(candidateEntry)

    def createBooleanEntry(self, name: str, defaultValue: bool) -> SettingsEntry[bool]:
        parameter = BooleanParameter(defaultValue)
        candidateEntry = SettingsEntry[bool](name, parameter)
        return self._registerEntryIfNonexistent(candidateEntry)

    def createIntegerEntry(self, name: str, defaultValue: int) -> SettingsEntry[int]:
        parameter = IntegerParameter(defaultValue, minimum=None, maximum=None)
        candidateEntry = SettingsEntry[int](name, parameter)
        return self._registerEntryIfNonexistent(candidateEntry)

    def createRealEntry(self, name: str, defaultValue: str | Decimal) -> SettingsEntry[Decimal]:
        parameter = DecimalParameter(defaultValue)
        candidateEntry = SettingsEntry[Decimal](name, parameter)
        return self._registerEntryIfNonexistent(candidateEntry)

    def _registerEntryIfNonexistent(self,
                                    candidateEntry: SettingsEntry[Any]) -> SettingsEntry[Any]:
        for existingEntry in self._entryList:
            if existingEntry.name.casefold() == candidateEntry.name.casefold():
                if not isinstance(candidateEntry, existingEntry.value.__class__):
                    raise TypeError('Attempted to duplicate settings entry with conflicting type.')

                return existingEntry

        candidateEntry.addObserver(self)
        self._entryList.append(candidateEntry)
        self.notifyObservers()

        return candidateEntry

    def __iter__(self) -> Iterator[SettingsEntry[Any]]:
        return iter(self._entryList)

    def __getitem__(self, index: int) -> SettingsEntry[Any]:
        return self._entryList[index]

    def __len__(self) -> int:
        return len(self._entryList)

    def update(self, observable: Observable) -> None:
        if observable in self._entryList:
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

            for settingsEntry in settingsGroup:
                if config.has_option(settingsGroup.name, settingsEntry.name):
                    valueString = config.get(settingsGroup.name, settingsEntry.name)
                    settingsEntry.setValueFromString(valueString)

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

            for settingsEntry in settingsGroup:
                valueString = str(settingsEntry.value)

                if changePathPrefix and isinstance(settingsEntry.value, Path):
                    try:
                        relativePath = settingsEntry.value.relative_to(
                            changePathPrefix.findPathPrefix)
                    except ValueError:
                        pass
                    else:
                        modifiedPath = changePathPrefix.replacementPathPrefix / relativePath
                        valueString = str(modifiedPath)

                config.set(settingsGroup.name, settingsEntry.name, valueString)

        logger.debug(f'Writing settings to \"{filePath}\"')

        with filePath.open(mode='w') as configFile:
            config.write(configFile)
