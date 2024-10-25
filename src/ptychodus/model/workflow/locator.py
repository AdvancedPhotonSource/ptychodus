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
        self._endpointID = group.createUUIDParameter(f'{entryPrefix}DataEndpointID', UUID(int=0))
        self._globusPath = group.createStringParameter(
            f'{entryPrefix}DataGlobusPath',
            f'/~/path/to/{entryPrefix.lower()}/data',
        )
        self._posixPath = group.createPathParameter(
            f'{entryPrefix}DataPosixPath',
            Path(f'/path/to/{entryPrefix.lower()}/data'),
        )

        self._endpointID.addObserver(self)
        self._globusPath.addObserver(self)
        self._posixPath.addObserver(self)

    def setEndpointID(self, endpointID: UUID) -> None:
        self._endpointID.setValue(endpointID)

    def getEndpointID(self) -> UUID:
        return self._endpointID.getValue()

    def setGlobusPath(self, globusPath: str) -> None:
        self._globusPath.setValue(globusPath)

    def getGlobusPath(self) -> str:
        return self._globusPath.getValue()

    def setPosixPath(self, posixPath: Path) -> None:
        self._posixPath.setValue(posixPath)

    def getPosixPath(self) -> Path:
        return self._posixPath.getValue()

    def update(self, observable: Observable) -> None:
        if observable is self._endpointID:
            self.notifyObservers()
        elif observable is self._globusPath:
            self.notifyObservers()
        elif observable is self._posixPath:
            self.notifyObservers()


class OutputDataLocator(DataLocator, Observer):
    def __init__(
        self, group: ParameterGroup, entryPrefix: str, inputDataLocator: DataLocator
    ) -> None:
        super().__init__()
        self._useRoundTrip = group.createBooleanParameter('UseRoundTrip', True)
        self._outputDataLocator = SimpleDataLocator(group, entryPrefix)
        self._inputDataLocator = inputDataLocator

        self._useRoundTrip.addObserver(self)
        self._inputDataLocator.addObserver(self)
        self._outputDataLocator.addObserver(self)

    def setRoundTripEnabled(self, enable: bool) -> None:
        self._useRoundTrip.setValue(enable)

    def isRoundTripEnabled(self) -> bool:
        return self._useRoundTrip.getValue()

    def setEndpointID(self, endpointID: UUID) -> None:
        self._outputDataLocator.setEndpointID(endpointID)

    def getEndpointID(self) -> UUID:
        return (
            self._inputDataLocator.getEndpointID()
            if self._useRoundTrip.getValue()
            else self._outputDataLocator.getEndpointID()
        )

    def setGlobusPath(self, globusPath: str) -> None:
        self._outputDataLocator.setGlobusPath(globusPath)

    def getGlobusPath(self) -> str:
        return (
            self._inputDataLocator.getGlobusPath()
            if self._useRoundTrip.getValue()
            else self._outputDataLocator.getGlobusPath()
        )

    def setPosixPath(self, posixPath: Path) -> None:
        self._outputDataLocator.setPosixPath(posixPath)

    def getPosixPath(self) -> Path:
        return (
            self._inputDataLocator.getPosixPath()
            if self._useRoundTrip.getValue()
            else self._outputDataLocator.getPosixPath()
        )

    def update(self, observable: Observable) -> None:
        if observable is self._useRoundTrip:
            self.notifyObservers()
        elif observable is self._inputDataLocator:
            self.notifyObservers()
        elif observable is self._outputDataLocator:
            self.notifyObservers()
