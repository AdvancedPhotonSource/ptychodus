from __future__ import annotations
from decimal import Decimal
from typing import TypeAlias

from scipy.interpolate import RegularGridInterpolator
import numpy
import numpy.typing

from ...api.object import ObjectArrayType, ObjectPoint
from ...api.observer import Observable, Observer
from ...api.scan import ScanPoint
from .selected import SelectedObject
from .sizer import ObjectSizer

FloatArrayType: TypeAlias = numpy.typing.NDArray[numpy.float_]


class ObjectInterpolator(Observer):

    def __init__(self, sizer: ObjectSizer, object_: SelectedObject) -> None:
        super().__init__()
        self._sizer = sizer
        self._object = object_
        self._probeGridXInMeters: FloatArrayType = numpy.array([])
        self._probeGridYInMeters: FloatArrayType = numpy.array([])
        self._interpolator = RegularGridInterpolator([], [])

    @classmethod
    def createInstance(cls, sizer: ObjectSizer, object_: SelectedObject) -> ObjectInterpolator:
        presenter = cls(sizer, object_)
        sizer.addObserver(presenter)
        object_.addObserver(presenter)
        presenter._resetInterpolator()
        return presenter

    def mapScanPointToObjectPoint(self, point: ScanPoint) -> ObjectPoint:
        selectedItem = self._object.getSelectedItem()

        if selectedItem is None:
            raise ValueError('No object is selected!')

        objectExtent = selectedItem.getExtentInPixels()
        dx = self._sizer.getPixelSizeXInMeters()
        dy = self._sizer.getPixelSizeYInMeters()
        center = self._sizer.getCentroidInMeters()

        return ObjectPoint(
            x=(point.x - center.x) / dx + Decimal(objectExtent.width) / 2,
            y=(point.y - center.y) / dy + Decimal(objectExtent.height) / 2,
        )

    def mapObjectPointToScanPoint(self, point: ObjectPoint) -> ScanPoint:
        selectedItem = self._object.getSelectedItem()

        if selectedItem is None:
            raise ValueError('No object is selected!')

        objectExtent = selectedItem.getExtentInPixels()
        dx = self._sizer.getPixelSizeXInMeters()
        dy = self._sizer.getPixelSizeYInMeters()
        center = self._sizer.getCentroidInMeters()

        return ScanPoint(
            x=center.x + dx * (point.x - Decimal(objectExtent.width) / 2),
            y=center.y + dy * (point.y - Decimal(objectExtent.height) / 2),
        )

    @staticmethod
    def _createAxis(ticks: int, tickSize: float, center: float) -> FloatArrayType:
        axis = numpy.arange(ticks) * tickSize
        axis += center - axis.mean()
        return axis

    def _resetInterpolator(self) -> None:
        selectedItem = self._object.getSelectedItem()

        if selectedItem is None:
            # FIXME raise ValueError('No object is selected!')
            return

        objectExtent = selectedItem.getExtentInPixels()
        dy = float(self._sizer.getPixelSizeYInMeters())
        dx = float(self._sizer.getPixelSizeXInMeters())
        center = self._sizer.getCentroidInMeters()

        objectYInMeters = self._createAxis(objectExtent.height, dy, float(center.y))
        objectXInMeters = self._createAxis(objectExtent.width, dx, float(center.x))
        self._interp = RegularGridInterpolator((objectYInMeters, objectXInMeters),
                                               selectedItem.getArray(),
                                               method='pchip')

        probeExtent = self._sizer.getProbeExtent()
        self._probeGridYInMeters, self._probeGridXInMeters = numpy.meshgrid(
            self._createAxis(probeExtent.height, dy, 0.),
            self._createAxis(probeExtent.width, dx, 0.),
        )

    def getPatch(self, point: ObjectPoint) -> ObjectArrayType:
        y = float(point.y) + self._probeGridYInMeters
        x = float(point.x) + self._probeGridXInMeters
        return self._interpolator((y, x))

    def update(self, observable: Observable) -> None:
        if observable in (self._sizer, self._object):
            self._resetInterpolator()
