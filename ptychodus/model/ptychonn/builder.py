from __future__ import annotations
from abc import ABC, abstractmethod
import logging

import numpy
import numpy.typing

from ...api.data import DiffractionPatternArrayType
from ...api.object import ObjectArrayType, ObjectPhaseCenteringStrategy
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..scan import ScanAPI

logger = logging.getLogger(__name__)


class ReconstructorTrainer(ABC):

    @abstractmethod
    def train(self, diffractionPatterns: DiffractionPatternArrayType,
              objectPatches: ObjectArrayType) -> None:
        pass


class PtychoNNTrainingDatasetBuilder(Observable, Observer):

    def __init__(self, phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                 diffractionDataset: ActiveDiffractionDataset, scanAPI: ScanAPI,
                 objectAPI: ObjectAPI) -> None:
        super().__init__()
        self._diffractionDataset = diffractionDataset
        self._scanAPI = scanAPI
        self._objectAPI = objectAPI
        self._trainer: ReconstructorTrainer | None = None
        self._diffractionPatternsArray: DiffractionPatternArrayType | None = None
        self._objectPatchesArray: ObjectArrayType | None = None

    def _appendDiffractionPatterns(self, array: DiffractionPatternArrayType) -> None:
        if self._diffractionPatternsArray is None:
            self._diffractionPatternsArray = array
        else:
            self._diffractionPatternsArray = numpy.concatenate(self._diffractionPatternsArray,
                                                               array)

    def _appendObjectPatches(self, array: ObjectArrayType) -> None:
        if self._objectPatchesArray is None:
            self._objectPatchesArray = array
        else:
            self._objectPatchesArray = numpy.concatenate(self._objectPatchesArray, array)

    def incorporateTrainingDataFromActiveItems(self) -> None:
        selectedScan = self._scanAPI.getSelectedScan()

        if selectedScan is None:
            raise ValueError('No scan is selected!')

        # FIXME objectPhaseCenteringStrategy = \
        # FIXME         self._phaseCenteringStrategyChooser.getCurrentStrategy()
        # FIXME object_ = objectPhaseCenteringStrategy(selectedObjectArray)

        diffractionPatterns = self._diffractionDataset.getAssembledData()
        objectPatchesList: list[ObjectArrayType] = list()

        for index in self._diffractionDataset.getAssembledIndexes():
            try:
                point = selectedScan[index]
            except KeyError:
                continue

            objectPatch = self._objectAPI.getObjectPatch(point)
            objectPatchesList.append(objectPatch)

        self._appendDiffractionPatterns(diffractionPatterns)
        self._appendObjectPatches(numpy.concatenate(objectPatchesList, axis=0))

    def train(self) -> None:
        if self._trainer is None:
            logger.error('Trainer not found!')
        elif self._diffractionPatternsArray is None:
            logger.error('Null diffraction patterns!')
        elif self._objectPatchesArray is None:
            logger.error('Null object patches!')
        else:
            # NOTE PtychoNN writes training outputs using internal mechanisms
            self._trainer.train(self._diffractionPatternsArray, self._objectPatchesArray)

    def clear(self) -> None:
        self._diffractionPatternsArray = None
        self._objectPatchesArray = None
