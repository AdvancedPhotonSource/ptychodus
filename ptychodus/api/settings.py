from __future__ import annotations
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import configparser
import logging

from .observer import Observable
from .parametric import ParameterGroup, PathParameter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PathPrefixChange:
    findPathPrefix: Path
    replacementPathPrefix: Path


class SettingsRegistry(Observable):
    def __init__(self) -> None:
        super().__init__()
        self._parameterGroup = ParameterGroup()
        self._fileFilterList: list[str] = ["Initialization Files (*.ini)"]

    def createGroup(self, name: str) -> ParameterGroup:
        return self._parameterGroup.createGroup(name)

    def __iter__(self) -> Iterator[str]:
        return iter(self._parameterGroup.groups())

    def __getitem__(self, name: str) -> ParameterGroup:
        return self._parameterGroup.getGroup(name)

    def __len__(self) -> int:
        return len(self._parameterGroup.groups())

    def getOpenFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getOpenFileFilter(self) -> str:
        return self._fileFilterList[0]

    def openSettings(self, filePath: Path) -> None:
        config = configparser.ConfigParser(interpolation=None)
        logger.debug(f'Reading settings from "{filePath}"')
        config.read(filePath)

        # TODO generalize to support nested parameter groups
        for groupName, group in self._parameterGroup.groups().items():
            try:
                groupConfig = config[groupName]
            except KeyError:
                pass
            else:
                for parameterName, parameter in group.parameters().items():
                    try:
                        valueString = groupConfig[parameterName]
                    except KeyError:
                        pass
                    else:
                        parameter.setValueFromString(valueString)

        self.notifyObservers()

    def getSaveFileFilterList(self) -> Sequence[str]:
        return self._fileFilterList

    def getSaveFileFilter(self) -> str:
        return self._fileFilterList[0]

    def saveSettings(
        self, filePath: Path, changePathPrefix: PathPrefixChange | None = None
    ) -> None:
        config = configparser.ConfigParser(interpolation=None)
        setattr(config, "optionxform", lambda option: option)

        for groupName, group in self._parameterGroup.groups().items():
            config.add_section(groupName)

            for parameterName, parameter in group.parameters().items():
                valueString = str(parameter)

                if changePathPrefix and isinstance(parameter, PathParameter):
                    modifiedPath = parameter.changePathPrefix(
                        changePathPrefix.findPathPrefix,
                        changePathPrefix.replacementPathPrefix,
                    )
                    valueString = str(modifiedPath)

                config.set(groupName, parameterName, valueString)

        logger.debug(f'Writing settings to "{filePath}"')

        with filePath.open(mode="w") as configFile:
            config.write(configFile)
