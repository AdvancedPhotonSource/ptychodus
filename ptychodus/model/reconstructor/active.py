from __future__ import annotations
from collections.abc import Iterable, Sequence
import logging
import time

import numpy

from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.reconstructor import (ReconstructInput, ReconstructOutput, Reconstructor,
                                  ReconstructorLibrary, TrainableReconstructor)
from ...api.scan import TabularScan
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

    def _prepareInputData(self) -> ReconstructInput:
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

        return ReconstructInput(
            diffractionPatternArray=diffractionPatternArray,
            scan=TabularScan(pointMap),
            probeArray=self._probeAPI.getSelectedProbeArray(),
            objectInterpolator=self._objectAPI.getSelectedObjectInterpolator(),
        )

    @property
    def name(self) -> str:
        return self._pluginChooser.currentPlugin.displayName

    def reconstruct(self, name: str) -> ReconstructOutput:
        reconstructor = self._pluginChooser.currentPlugin.strategy
        parameters = self._prepareInputData()

        tic = time.perf_counter()
        result = reconstructor.reconstruct(parameters)
        toc = time.perf_counter()
        logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')

        if result.scan is not None:
            self._scanAPI.insertItemIntoRepositoryFromScan(name, result.scan, selectItem=True)

        if result.probeArray is not None:
            self._probeAPI.insertItemIntoRepositoryFromArray(name,
                                                             result.probeArray,
                                                             selectItem=True)

        if result.objectArray is not None:
            self._objectAPI.insertItemIntoRepositoryFromArray(name,
                                                              result.objectArray,
                                                              selectItem=True)

        return result

    @property
    def isTrainable(self) -> bool:
        reconstructor = self._pluginChooser.currentPlugin.strategy
        return isinstance(reconstructor, TrainableReconstructor)

    def ingest(self) -> None:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            parameters = self._prepareInputData()
            tic = time.perf_counter()
            reconstructor.ingest(parameters)
            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds.')
        else:
            logger.error('Reconstructor is not trainable!')

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
