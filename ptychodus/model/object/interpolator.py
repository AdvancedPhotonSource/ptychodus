from __future__ import annotations
from collections.abc import Sequence
import logging

from scipy.ndimage import map_coordinates
import numpy

from ...api.image import ImageExtent
from ...api.object import (ObjectArrayType, ObjectAxis, ObjectGrid, ObjectInterpolator,
                           ObjectPatch, ObjectPatchGrid, ObjectPhaseCenteringStrategy)
from ...api.observer import Observable, Observer
from ...api.plugins import PluginChooser
from ...api.scan import ScanPoint
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

    def getPatch(self, patchCenter: ScanPoint, patchExtent: ImageExtent) -> ObjectPatch:
        grid = ObjectPatchGrid.createInstance(self._grid, patchCenter, patchExtent)
        yy, xx = numpy.meshgrid(grid.axisY.getPixelScanCoordinates(),
                                grid.axisX.getPixelScanCoordinates(),
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
                           objectCentroid: ScanPoint) -> ObjectInterpolator:
        centerPhase = self._phaseCenteringStrategyChooser.currentPlugin.strategy

        objectGrid = ObjectGrid(
            axisX=ObjectAxis(
                centerInMeters=float(objectCentroid.x),
                pixelSizeInMeters=float(self._sizer.getPixelSizeXInMeters()),
                numberOfPixels=objectArray.shape[-1],
            ),
            axisY=ObjectAxis(
                centerInMeters=float(objectCentroid.y),
                pixelSizeInMeters=float(self._sizer.getPixelSizeYInMeters()),
                numberOfPixels=objectArray.shape[-2],
            ),
        )

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
