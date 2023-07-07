from __future__ import annotations
from collections.abc import Sequence
import logging

from scipy.ndimage import map_coordinates
import numpy

from ...api.image import ImageExtent
from ...api.object import (ObjectArrayType, ObjectAxis, ObjectGrid, ObjectInterpolator,
                           ObjectPatch, ObjectPatchAxis, ObjectPhaseCenteringStrategy)
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
        axisX = ObjectPatchAxis(self._grid.axisX, float(patchCenter.x), patchExtent.width)
        axisY = ObjectPatchAxis(self._grid.axisY, float(patchCenter.y), patchExtent.height)
        xx, yy = numpy.meshgrid(axisX.getPixelScanCoordinates(), axisY.getPixelScanCoordinates())
        array = map_coordinates(self._array, (yy, xx), order=1)
        return ObjectPatch(axisX=axisX, axisY=axisY, array=array)


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
        centerPhase = self._phaseCenteringStrategyChooser.getCurrentStrategy()

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
        return self._phaseCenteringStrategyChooser.getCurrentDisplayName()

    def setPhaseCenteringStrategy(self, name: str) -> None:
        self._phaseCenteringStrategyChooser.setFromDisplayName(name)
        simpleName = self._phaseCenteringStrategyChooser.getCurrentSimpleName()
        self._settings.phaseCenteringStrategy.value = simpleName

    def _syncFromSettings(self) -> None:
        simpleName = self._settings.phaseCenteringStrategy.value
        self._phaseCenteringStrategyChooser.setFromSimpleName(simpleName)

    def update(self, observable: Observable) -> None:
        if observable is self._reinitObservable:
            self._syncFromSettings()
