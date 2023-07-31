from __future__ import annotations
from collections.abc import Iterable, Sequence
import logging
import time

import numpy

from ...api.data import DiffractionPatternArrayType
from ...api.object import ObjectInterpolator
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.probe import ProbeArrayType
from ...api.reconstructor import (ReconstructInput, ReconstructOutput, Reconstructor,
                                  ReconstructorLibrary, TrainableReconstructor)
from ...api.scan import Scan, TabularScan
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..probe import ProbeAPI
from ..scan import ScanAPI
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ActiveReconstructor(Observable, Observer):

    def __init__(self, settings: ReconstructorSettings,
                 diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                 probeAPI: ProbeAPI, objectAPI: ObjectAPI, reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._diffractionDataset = diffractionDataset
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._reinitObservable = reinitObservable
        self._pluginChooser = PluginChooser[Reconstructor]()

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings,
                       diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                       probeAPI: ProbeAPI, objectAPI: ObjectAPI,
                       libraries: Iterable[ReconstructorLibrary],
                       reinitObservable: Observable) -> ActiveReconstructor:
        activeReconstructor = cls(settings, diffractionDataset, scanAPI, probeAPI, objectAPI,
                                  reinitObservable)

        for library in libraries:
            for reconstructor in library:
                activeReconstructor._pluginChooser.registerPlugin(
                    reconstructor,
                    simpleName=f'{library.name}/{reconstructor.name}',
                )

        reinitObservable.addObserver(activeReconstructor)
        activeReconstructor._syncFromSettings()
        return activeReconstructor

    def getReconstructorList(self) -> Sequence[str]:
        return self._pluginChooser.getDisplayNameList()

    def _prepareInputData(self, name: str) -> tuple[str, Scan, DiffractionPatternArrayType]:
        selectedScan = self._scanAPI.getSelectedScan()

        if selectedScan is None:
            raise ValueError('No scan is selected!')

        dataIndexes = self._diffractionDataset.getAssembledIndexes()
        scanIndexes = selectedScan.keys()
        commonIndexes = sorted(set(dataIndexes).intersection(scanIndexes))

        diffractionPatternArray = numpy.take(self._diffractionDataset.getAssembledData(),
                                             commonIndexes,
                                             axis=0)

        pointMap = {index: selectedScan[index] for index in commonIndexes}
        filteredScan = TabularScan(pointMap)
        scanName = self._scanAPI.insertItemIntoRepositoryFromScan(name,
                                                                  filteredScan,
                                                                  selectItem=True)

        if scanName is None:
            raise ValueError('Failed to clone selected scan!')

        return scanName, filteredScan, diffractionPatternArray

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

    @property
    def name(self) -> str:
        return self._pluginChooser.currentPlugin.displayName

    def execute(self, name: str) -> ReconstructOutput:
        scanName, scan, diffractionPatternArray = self._prepareInputData(name)
        probeName, probeArray = self._cloneSelectedProbe(name)
        objectName, objectInterpolator = self._cloneSelectedObject(name)

        parameters = ReconstructInput(
            diffractionPatternArray=diffractionPatternArray,
            scan=scan,
            probeArray=probeArray,
            objectInterpolator=objectInterpolator,
        )

        tic = time.perf_counter()
        result = self._pluginChooser.currentPlugin.strategy.execute(parameters)
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

    @property
    def isTrainable(self) -> bool:
        reconstructor = self._pluginChooser.currentPlugin.strategy
        return isinstance(reconstructor, TrainableReconstructor)

    def train(self) -> None:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            reconstructor.train()
        else:
            logger.error('Reconstructor is not trainable!')

    def reset(self) -> None:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            reconstructor.reset()
        else:
            logger.error('Reconstructor is not trainable!')

    def selectReconstructor(self, name: str) -> None:
        self._pluginChooser.setCurrentPluginByName(name)
        self._settings.algorithm.value = self._pluginChooser.currentPlugin.simpleName
        self.notifyObservers()

    def _syncFromSettings(self) -> None:
        self.selectReconstructor(self._settings.algorithm.value)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()
