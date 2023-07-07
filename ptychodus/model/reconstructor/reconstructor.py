from __future__ import annotations
from collections.abc import Iterable, Iterator, Mapping
import logging
import time

from ...api.object import ObjectInterpolator
from ...api.observer import Observable, Observer
from ...api.plugins import PluginEntry
from ...api.probe import ProbeArrayType
from ...api.reconstructor import (NullReconstructor, ReconstructInput, ReconstructOutput,
                                  Reconstructor, ReconstructorLibrary)
from ...api.scan import Scan, ScanPoint, TabularScan
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..probe import ProbeAPI
from ..scan import ScanAPI
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


class ActiveReconstructor(Observable, Observer):

    def __init__(self, settings: ReconstructorSettings, repository: ReconstructorRepository,
                 diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                 probeAPI: ProbeAPI, objectAPI: ObjectAPI, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._repository = repository
        self._diffractionDataset = diffractionDataset
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._reinitObservable = reinitObservable
        self._reconstructorPlugin = PluginEntry[Reconstructor]('None', 'None',
                                                               NullReconstructor('None'))

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings, repository: ReconstructorRepository,
                       diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                       probeAPI: ProbeAPI, objectAPI: ObjectAPI,
                       reinitObservable: Observable) -> ActiveReconstructor:
        reconstructor = cls(settings, repository, diffractionDataset, scanAPI, probeAPI, objectAPI,
                            reinitObservable)
        reinitObservable.addObserver(reconstructor)
        reconstructor._syncFromSettings()
        return reconstructor

    @property
    def name(self) -> str:
        return self._reconstructorPlugin.displayName

    def _createFilteredCopyOfSelectedScan(self, name: str) -> tuple[str, Scan]:
        selectedScan = self._scanAPI.getSelectedScan()

        if selectedScan is None:
            raise ValueError('No scan is selected!')

        pointMap: dict[int, ScanPoint] = dict()

        for index in self._diffractionDataset.getAssembledIndexes():
            try:
                pointMap[index] = selectedScan[index]
            except KeyError:
                continue

        filteredScan = TabularScan(pointMap)
        scanName = self._scanAPI.insertItemIntoRepositoryFromScan(name,
                                                                  filteredScan,
                                                                  selectItem=True)

        if scanName is None:
            raise ValueError('Failed to clone selected scan!')

        return scanName, filteredScan

    def _cloneSelectedProbe(self, name: str) -> tuple[str, ProbeArrayType]:
        selectedProbe = self._probeAPI.getSelectedProbeArray()

        if selectedProbe is None:
            raise ValueError('No probe is selected!')

        probeName = self._probeAPI.insertItemIntoRepositoryFromArray(name,
                                                                     selectedProbe,
                                                                     selectItem=True)

        if probeName is None:
            raise ValueError('Failed to clone selected probe!')

        return probeName, selectedProbe

    def _cloneSelectedObject(self, name: str) -> tuple[str, ObjectInterpolator]:
        objectInterpolator = self._objectAPI.getSelectedObjectInterpolator()
        objectName = self._objectAPI.insertItemIntoRepositoryFromArray(
            name, objectInterpolator.getArray(), selectItem=True)

        if objectName is None:
            raise ValueError('Failed to clone selected object!')

        return objectName, objectInterpolator

    def reconstruct(self, name: str) -> ReconstructOutput:
        scanName, scan = self._createFilteredCopyOfSelectedScan(name)
        probeName, probeArray = self._cloneSelectedProbe(name)
        objectName, objectInterpolator = self._cloneSelectedObject(name)

        # FIXME prefilter diffraction patterns so that scan indexes are authoritative
        diffractionPatternArray = self._diffractionDataset.getAssembledData()

        parameters = ReconstructInput(
            diffractionPatternArray=diffractionPatternArray,
            scan=scan,
            probeArray=probeArray,
            objectInterpolator=objectInterpolator,
        )

        tic = time.perf_counter()
        result = self._reconstructorPlugin.strategy.execute(parameters)
        toc = time.perf_counter()
        logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')

        if result.scan is not None:
            self._scanAPI.insertItemIntoRepositoryFromScan(scanName, result.scan, replaceItem=True)

        if result.probeArray is not None:
            self._probeAPI.insertItemIntoRepositoryFromArray(probeName,
                                                             result.probeArray,
                                                             replaceItem=True)

        if result.objectArray is not None:
            self._objectAPI.insertItemIntoRepositoryFromArray(objectName,
                                                              result.objectArray,
                                                              replaceItem=True)

        return result

    @staticmethod
    def _simplifyName(name: str) -> str:
        # remove whitespace and casefold
        return ''.join(name.split()).casefold()

    def selectReconstructor(self, name: str) -> None:
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

    def _syncFromSettings(self) -> None:
        self.selectReconstructor(self._settings.algorithm.value)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()
