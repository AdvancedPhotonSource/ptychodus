from __future__ import annotations
from collections.abc import Iterable, Sequence
from pathlib import Path
import logging
import time

import numpy

from ...api.object import Object
from ...api.observer import Observable, Observer
from ...api.visualize import Plot2D
from ...api.plugins import PluginChooser
from ...api.probe import Probe
from ...api.reconstructor import (NullReconstructor, ReconstructInput, ReconstructOutput,
                                  Reconstructor, ReconstructorLibrary, TrainableReconstructor)
from ...api.scan import ScanPoint
from ..scan import ScanIndexFilter
from ..patterns import ActiveDiffractionDataset
from .settings import ReconstructorSettings

logger = logging.getLogger(__name__)


class ActiveReconstructor(Observable, Observer):

    def __init__(self, settings: ReconstructorSettings,
                 diffractionDataset: ActiveDiffractionDataset, reinitObservable: Observable,
                 pluginChooser: PluginChooser[Reconstructor]) -> None:
        super().__init__()
        self._settings = settings
        self._diffractionDataset = diffractionDataset
        self._reinitObservable = reinitObservable
        self._pluginChooser = pluginChooser

    @classmethod
    def createInstance(cls, settings: ReconstructorSettings,
                       diffractionDataset: ActiveDiffractionDataset,
                       libraries: Iterable[ReconstructorLibrary],
                       reinitObservable: Observable) -> ActiveReconstructor:
        pluginChooser = PluginChooser[Reconstructor]()

        for library in libraries:
            for reconstructor in library:
                pluginChooser.registerPlugin(
                    reconstructor,
                    displayName=f'{library.name}/{reconstructor.name}',
                )

        if not pluginChooser:
            pluginChooser.registerPlugin(NullReconstructor('None'), displayName='None/None')

        activeReconstructor = cls(settings, diffractionDataset, reinitObservable, pluginChooser)
        reinitObservable.addObserver(activeReconstructor)
        activeReconstructor._syncFromSettings()
        return activeReconstructor

    def getReconstructorList(self) -> Sequence[str]:
        return self._pluginChooser.getDisplayNameList()

    def _prepareInputData(self, indexFilter: ScanIndexFilter) -> ReconstructInput:
        selectedScan = self._scanAPI.getSelectedScan()

        if selectedScan is None:
            raise ValueError('No scan is selected!')

        dataIndexes = self._diffractionDataset.getAssembledIndexes()
        scanIndexes = [point.index for point in selectedScan if indexFilter(point.index)]
        commonIndexes = sorted(set(dataIndexes).intersection(scanIndexes))

        diffractionPatternArray = numpy.take(
            self._diffractionDataset.getAssembledData(),
            commonIndexes,
            axis=0,
        )

        pointList: list[ScanPoint] = list()
        pointIter = iter(selectedScan)

        for index in commonIndexes:
            while True:
                point = next(pointIter)

                if point.index == index:
                    pointList.append(point)
                    break

        return ReconstructInput(
            diffractionPatternArray=diffractionPatternArray,
            scan=pointList,
            probeArray=self._probeAPI.getSelectedProbe().getArray(),
            # TODO vvv generalize when able vvv
            objectInterpolator=self._objectAPI.getSelectedThinObjectInterpolator(),
        )

    @property
    def name(self) -> str:
        return self._pluginChooser.currentPlugin.displayName

    def reconstruct(self, label: str, indexFilter: ScanIndexFilter, *,
                    selectResults: bool) -> ReconstructOutput:
        reconstructor = self._pluginChooser.currentPlugin.strategy
        parameters = self._prepareInputData(indexFilter)

        tic = time.perf_counter()
        result = reconstructor.reconstruct(parameters)
        toc = time.perf_counter()
        logger.info(f'Reconstruction time {toc - tic:.4f} seconds.')

        if result.scan is not None:
            self._scanAPI.insertItemIntoRepositoryFromScan(label,
                                                           result.scan,
                                                           selectItem=selectResults)

        if result.probeArray is not None:
            self._probeAPI.insertItemIntoRepository(label,
                                                    Probe(result.probeArray),
                                                    selectItem=selectResults)

        if result.objectArray is not None:
            self._objectAPI.insertItemIntoRepository(label,
                                                     Object(result.objectArray),
                                                     selectItem=selectResults)

        return result

    @property
    def isTrainable(self) -> bool:
        reconstructor = self._pluginChooser.currentPlugin.strategy
        return isinstance(reconstructor, TrainableReconstructor)

    def ingestTrainingData(self) -> None:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Preparing input data...')
            tic = time.perf_counter()
            parameters = self._prepareInputData(ScanIndexFilter.ALL)
            toc = time.perf_counter()
            logger.info(f'Data preparation time {toc - tic:.4f} seconds.')

            logger.info('Ingesting...')
            tic = time.perf_counter()
            reconstructor.ingestTrainingData(parameters)
            toc = time.perf_counter()
            logger.info(f'Ingest time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def getSaveFileFilterList(self) -> Sequence[str]:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveFileFilterList()
        else:
            logger.warning('Reconstructor is not trainable!')

        return list()

    def getSaveFileFilter(self) -> str:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            return reconstructor.getSaveFileFilter()
        else:
            logger.warning('Reconstructor is not trainable!')

        return str()

    def saveTrainingData(self, filePath: Path) -> None:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Saving...')
            tic = time.perf_counter()
            reconstructor.saveTrainingData(filePath)
            toc = time.perf_counter()
            logger.info(f'Save time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def train(self) -> Plot2D:
        reconstructor = self._pluginChooser.currentPlugin.strategy
        plot2D = Plot2D.createNull()

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Training...')
            tic = time.perf_counter()
            plot2D = reconstructor.train()
            toc = time.perf_counter()
            logger.info(f'Training time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

        return plot2D

    def clearTrainingData(self) -> None:
        reconstructor = self._pluginChooser.currentPlugin.strategy

        if isinstance(reconstructor, TrainableReconstructor):
            logger.info('Resetting...')
            tic = time.perf_counter()
            reconstructor.clearTrainingData()
            toc = time.perf_counter()
            logger.info(f'Reset time {toc - tic:.4f} seconds.')
        else:
            logger.warning('Reconstructor is not trainable!')

    def selectReconstructor(self, name: str) -> None:
        self._pluginChooser.setCurrentPluginByName(name)
        self._settings.algorithm.value = self._pluginChooser.currentPlugin.simpleName
        self.notifyObservers()

    def _syncFromSettings(self) -> None:
        self.selectReconstructor(self._settings.algorithm.value)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()
