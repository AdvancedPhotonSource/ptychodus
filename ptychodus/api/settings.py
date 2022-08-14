from __future__ import annotations
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Optional, TypeVar
from uuid import UUID
import configparser

from .observer import Observable, Observer

T = TypeVar('T')


class SettingsEntry(Generic[T], Observable):

    def __init__(self, name: str, defaultValue: T, stringConverter: Callable[[str], T]) -> None:
        super().__init__()
        self._name = name
        self._value = defaultValue
        self._stringConverter = stringConverter

    @property
    def name(self) -> str:
        return self._name

    @property
    def value(self) -> T:
        return self._value

    @value.setter
    def value(self, value: T) -> None:
        candidate_value = value

        if self._value != candidate_value:
            self._value = candidate_value
            self.notifyObservers()

    def setValueFromString(self, valueString: str) -> None:
        self.value = self._stringConverter(valueString)


class SettingsGroup(Observable, Observer):

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._entryList: list[SettingsEntry[Any]] = list()

    @staticmethod
    def convertFloatToDecimal(value: float) -> Decimal:
        return Decimal(repr(value))

    @property
    def name(self) -> str:
        return self._name

    def createStringEntry(self, name: str, defaultValue: str) -> SettingsEntry[str]:
        candidateEntry = SettingsEntry[str](name, defaultValue,
                                            lambda valueString: str(valueString))
        return self._registerEntryIfNonexistent(candidateEntry)

    def createPathEntry(self, name: str, defaultValue: Path) -> SettingsEntry[Path]:
        candidateEntry = SettingsEntry[Path](name, defaultValue,
                                             lambda valueString: Path(valueString))
        return self._registerEntryIfNonexistent(candidateEntry)

    def createUUIDEntry(self, name: str, defaultValue: UUID) -> SettingsEntry[UUID]:
        candidateEntry = SettingsEntry[UUID](name, defaultValue,
                                             lambda valueString: UUID(valueString))
        return self._registerEntryIfNonexistent(candidateEntry)

    def createBooleanEntry(self, name: str, defaultValue: bool) -> SettingsEntry[bool]:
        trueStringList = ['1', 'true', 't', 'yes', 'y']
        candidateEntry = SettingsEntry[bool](
            name, defaultValue, lambda valueString: valueString.lower() in trueStringList)
        return self._registerEntryIfNonexistent(candidateEntry)

    def createIntegerEntry(self, name: str, defaultValue: int) -> SettingsEntry[int]:
        candidateEntry = SettingsEntry[int](name, defaultValue,
                                            lambda valueString: int(valueString))
        return self._registerEntryIfNonexistent(candidateEntry)

    def createRealEntry(self, name: str, defaultValue: str) -> SettingsEntry[Decimal]:
        candidateEntry = SettingsEntry[Decimal](name, Decimal(defaultValue),
                                                lambda valueString: Decimal(valueString))
        return self._registerEntryIfNonexistent(candidateEntry)

    def _registerEntryIfNonexistent(self,
                                    candidateEntry: SettingsEntry[Any]) -> SettingsEntry[Any]:
        for existingEntry in self._entryList:
            if existingEntry.name.casefold() == candidateEntry.name.casefold():
                if type(existingEntry.value) != type(candidateEntry.value):
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
    PREFIX_PLACEHOLDER_TEXT = 'PREFIX'

    def __init__(self, replacementPathPrefix: Optional[str]) -> None:
        super().__init__()
        self._replacementPathPrefix = replacementPathPrefix
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

    def getReplacementPathPrefix(self) -> Optional[str]:
        return self._replacementPathPrefix

    def setReplacementPathPrefix(self, replacementPathPrefix: str) -> None:
        self._replacementPathPrefix = replacementPathPrefix

    def getOpenFileFilterList(self) -> list[str]:
        return self._fileFilterList

    def getOpenFileFilter(self) -> str:
        return self._fileFilterList[0]

    def openSettings(self, filePath: Path) -> None:
        config = configparser.ConfigParser(interpolation=None)
        config.read(filePath)

        for settingsGroup in self._groupList:
            if not config.has_section(settingsGroup.name):
                continue

            for settingsEntry in settingsGroup:
                if config.has_option(settingsGroup.name, settingsEntry.name):
                    valueString = config.get(settingsGroup.name, settingsEntry.name)

                    if self._replacementPathPrefix is not None \
                            and isinstance(settingsEntry.value, Path):
                        if valueString.startswith(SettingsRegistry.PREFIX_PLACEHOLDER_TEXT):
                            valueString = self._replacementPathPrefix \
                                    + valueString[len(SettingsRegistry.PREFIX_PLACEHOLDER_TEXT):]

                    settingsEntry.setValueFromString(valueString)

        self.notifyObservers()

    def getSaveFileFilterList(self) -> list[str]:
        return self._fileFilterList

    def getSaveFileFilter(self) -> str:
        return self._fileFilterList[0]

    def saveSettings(self, filePath: Path) -> None:
        config = configparser.ConfigParser(interpolation=None)
        config.optionxform = lambda option: option

        for settingsGroup in self._groupList:
            config.add_section(settingsGroup.name)

            for settingsEntry in settingsGroup:
                valueString = str(settingsEntry.value)

                if self._replacementPathPrefix is not None \
                        and isinstance(settingsEntry.value, Path):
                    if valueString.startswith(self._replacementPathPrefix):
                        valueString = SettingsRegistry.PREFIX_PLACEHOLDER_TEXT \
                                + valueString[len(self._replacementPathPrefix):]
                config.set(settingsGroup.name, settingsEntry.name, valueString)

        with open(filePath, 'w') as configFile:
            config.write(configFile)
