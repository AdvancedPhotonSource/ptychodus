from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import UUID

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import (
    ParameterGroup,
)


class DataLocator(ABC, Observable):
    @abstractmethod
    def setEndpointID(self, endpointID: UUID) -> None:
        pass

    @abstractmethod
    def getEndpointID(self) -> UUID:
        pass

    @abstractmethod
    def setGlobusPath(self, globusPath: str) -> None:
        pass

    @abstractmethod
    def getGlobusPath(self) -> str:
        pass

    @abstractmethod
    def setPosixPath(self, posixPath: Path) -> None:
        pass

    @abstractmethod
    def getPosixPath(self) -> Path:
        pass


class SimpleDataLocator(DataLocator, Observer):
    def __init__(self, group: ParameterGroup, entryPrefix: str) -> None:
        super().__init__()
        self._endpointID = group.create_uuid_parameter(f'{entryPrefix}DataEndpointID', UUID(int=0))
        self._globusPath = group.create_string_parameter(
            f'{entryPrefix}DataGlobusPath',
            f'/~/path/to/{entryPrefix.lower()}/data',
        )
        self._posixPath = group.create_path_parameter(
            f'{entryPrefix}DataPosixPath',
            Path(f'/path/to/{entryPrefix.lower()}/data'),
        )

        self._endpointID.add_observer(self)
        self._globusPath.add_observer(self)
        self._posixPath.add_observer(self)

    def setEndpointID(self, endpointID: UUID) -> None:
        self._endpointID.set_value(endpointID)

    def getEndpointID(self) -> UUID:
        return self._endpointID.get_value()

    def setGlobusPath(self, globusPath: str) -> None:
        self._globusPath.set_value(globusPath)

    def getGlobusPath(self) -> str:
        return self._globusPath.get_value()

    def setPosixPath(self, posixPath: Path) -> None:
        self._posixPath.set_value(posixPath)

    def getPosixPath(self) -> Path:
        return self._posixPath.get_value()

    def _update(self, observable: Observable) -> None:
        if observable is self._endpointID:
            self.notify_observers()
        elif observable is self._globusPath:
            self.notify_observers()
        elif observable is self._posixPath:
            self.notify_observers()


class OutputDataLocator(DataLocator, Observer):
    def __init__(
        self, group: ParameterGroup, entryPrefix: str, inputDataLocator: DataLocator
    ) -> None:
        super().__init__()
        self._useRoundTrip = group.create_boolean_parameter('UseRoundTrip', True)
        self._outputDataLocator = SimpleDataLocator(group, entryPrefix)
        self._inputDataLocator = inputDataLocator

        self._useRoundTrip.add_observer(self)
        self._inputDataLocator.add_observer(self)
        self._outputDataLocator.add_observer(self)

    def setRoundTripEnabled(self, enable: bool) -> None:
        self._useRoundTrip.set_value(enable)

    def isRoundTripEnabled(self) -> bool:
        return self._useRoundTrip.get_value()

    def setEndpointID(self, endpointID: UUID) -> None:
        self._outputDataLocator.setEndpointID(endpointID)

    def getEndpointID(self) -> UUID:
        return (
            self._inputDataLocator.getEndpointID()
            if self._useRoundTrip.get_value()
            else self._outputDataLocator.getEndpointID()
        )

    def setGlobusPath(self, globusPath: str) -> None:
        self._outputDataLocator.setGlobusPath(globusPath)

    def getGlobusPath(self) -> str:
        return (
            self._inputDataLocator.getGlobusPath()
            if self._useRoundTrip.get_value()
            else self._outputDataLocator.getGlobusPath()
        )

    def setPosixPath(self, posixPath: Path) -> None:
        self._outputDataLocator.setPosixPath(posixPath)

    def getPosixPath(self) -> Path:
        return (
            self._inputDataLocator.getPosixPath()
            if self._useRoundTrip.get_value()
            else self._outputDataLocator.getPosixPath()
        )

    def _update(self, observable: Observable) -> None:
        if observable is self._useRoundTrip:
            self.notify_observers()
        elif observable is self._inputDataLocator:
            self.notify_observers()
        elif observable is self._outputDataLocator:
            self.notify_observers()
