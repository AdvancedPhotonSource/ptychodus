from __future__ import annotations
from collections.abc import Mapping
from typing import Iterable, Iterator
import logging

from ...api.observer import Observable, Observer
from ...api.plugins import PluginEntry
from ...api.reconstructor import (NullReconstructor, ReconstructResult, Reconstructor,
                                  ReconstructorLibrary)
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ReconstructorRepository(Mapping[str, Reconstructor], Observable):

    def __init__(self) -> None:
        super().__init__()
        self._reconstructorDict: dict[str, Reconstructor] = dict()

    @classmethod
    def createInstance(cls, libraries: Iterable[ReconstructorLibrary]) -> ReconstructorRepository:
        repository = cls()

        for library in libraries:
            repository.registerLibrary(library)

        return repository

    def __iter__(self) -> Iterator[str]:
        return iter(self._reconstructorDict)

    def __getitem__(self, name: str) -> Reconstructor:
        return self._reconstructorDict[name]

    def __len__(self) -> int:
        return len(self._reconstructorDict)

    def registerLibrary(self, library: ReconstructorLibrary) -> None:
        for reconstructor in library:
            name = f'{library.name}/{reconstructor.name}'
            self._reconstructorDict[name] = reconstructor

        self.notifyObservers()


class ActiveReconstructor(Reconstructor, Observable, Observer):

    def __init__(self, settings: ReconstructorSettings,
                 repository: ReconstructorRepository) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository
        self._reconstructorPlugin = PluginEntry[Reconstructor]('None', 'None',
                                                               NullReconstructor('None'))

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings,
                       repository: ReconstructorRepository) -> ActiveReconstructor:
        reconstructor = cls(settings, repository)
        settings.algorithm.addObserver(reconstructor)
        repository.addObserver(reconstructor)

        reconstructor._syncReconstructorFromSettings()

        return reconstructor

    @property
    def name(self) -> str:
        return self._reconstructorPlugin.displayName

    def reconstruct(self) -> ReconstructResult:
        return self._reconstructorPlugin.strategy.reconstruct()

    @staticmethod
    def _simplifyName(name: str) -> str:
        # remove whitespace and casefold
        return ''.join(name.split()).casefold()

    def setActiveReconstructor(self, name: str) -> None:
        displayName = self._simplifyName(name)

        for rname in self._repository:
            if self._simplifyName(rname) == displayName:
                displayName = rname
                break

        try:
            reconstructor = self._repository[displayName]
        except KeyError:
            logger.debug(f'Invalid reconstructor name \"{name}\"')
            reconstructor = NullReconstructor('None')

        self._reconstructorPlugin = PluginEntry[Reconstructor](
            simpleName=''.join(displayName.split()),
            displayName=displayName,
            strategy=reconstructor,
        )

        self._settings.algorithm.value = self._reconstructorPlugin.simpleName
        self.notifyObservers()

    def _syncReconstructorFromSettings(self) -> None:
        self.setActiveReconstructor(self._settings.algorithm.value)

    def update(self, observable: Observable) -> None:
        if observable is self._settings.algorithm:
            self._syncReconstructorFromSettings()
