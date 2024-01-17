from __future__ import annotations
from collections.abc import Sequence
import logging

from scipy.ndimage import map_coordinates
import numpy

from ...api.geometry import Point2D
from ...api.object import (ObjectArrayType, ObjectGrid, ObjectInterpolator, ObjectPatch,
                           ObjectPatchGrid, ObjectPhaseCenteringStrategy)
from ...api.observer import Observable, Observer
from ...api.patterns import ImageExtent
from ...api.plugins import PluginChooser
from .settings import ObjectSettings
from .sizer import ObjectSizer

logger = logging.getLogger(__name__)


class ObjectLinearInterpolator(ObjectInterpolator):

    def __init__(self, grid: ObjectGrid, array: ObjectArrayType) -> None:
        self._grid = grid
        self._array = array

    def getGrid(self) -> ObjectGrid:
        return self._grid

    def getArray(self) -> ObjectArrayType:
        return self._array

    def getPatch(self, patchCenter: Point2D, patchExtent: ImageExtent) -> ObjectPatch:
        grid = ObjectPatchGrid.createInstance(self._grid, patchCenter, patchExtent)
        yy, xx = numpy.meshgrid(grid.axisY.getObjectCoordinates(),
                                grid.axisX.getObjectCoordinates(),
                                indexing='ij')
        array = map_coordinates(self._array, (yy, xx), order=1)
        return ObjectPatch(grid=grid, array=array.reshape(patchExtent.shape))


class ObjectInterpolatorFactory(Observable, Observer):

    def __init__(self, settings: ObjectSettings, sizer: ObjectSizer,
                 phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                 reinitObservable: Observable) -> None:
        super().__init__()
        self._settings = settings
        self._sizer = sizer
        self._phaseCenteringStrategyChooser = phaseCenteringStrategyChooser
        self._reinitObservable = reinitObservable

    @classmethod
    def createInstance(cls, settings: ObjectSettings, sizer: ObjectSizer,
                       phaseCenteringStrategyChooser: PluginChooser[ObjectPhaseCenteringStrategy],
                       reinitObservable: Observable) -> ObjectInterpolatorFactory:
        factory = cls(settings, sizer, phaseCenteringStrategyChooser, reinitObservable)
        reinitObservable.addObserver(factory)
        factory._syncFromSettings()
        return factory

    def createInterpolator(self, objectArray: ObjectArrayType,
                           objectCentroid: Point2D) -> ObjectInterpolator:
        centerPhase = self._phaseCenteringStrategyChooser.currentPlugin.strategy
        objectExtent = ImageExtent(
            widthInPixels=objectArray.shape[-1],
            heightInPixels=objectArray.shape[-2],
        )
        pixelGeometry = self._sizer.getPixelGeometry()
        objectGrid = ObjectGrid.createInstance(objectCentroid, objectExtent, pixelGeometry)

        return ObjectLinearInterpolator(
            grid=objectGrid,
            array=centerPhase(objectArray),
        )

    def getPhaseCenteringStrategyList(self) -> Sequence[str]:
        return self._phaseCenteringStrategyChooser.getDisplayNameList()

    def getPhaseCenteringStrategy(self) -> str:
        return self._phaseCenteringStrategyChooser.currentPlugin.displayName

    def setPhaseCenteringStrategy(self, name: str) -> None:
        self._phaseCenteringStrategyChooser.setCurrentPluginByName(name)
        simpleName = self._phaseCenteringStrategyChooser.currentPlugin.simpleName
        self._settings.phaseCenteringStrategy.value = simpleName

    def _syncFromSettings(self) -> None:
        simpleName = self._settings.phaseCenteringStrategy.value
        self._phaseCenteringStrategyChooser.setCurrentPluginByName(simpleName)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()
