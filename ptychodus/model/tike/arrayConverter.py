from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

import numpy
import numpy.typing

from ...api.data import DiffractionPatternData
from ...api.object import ObjectArrayType
from ...api.probe import ProbeArrayType
from ...api.scan import Scan, ScanPoint, TabularScan
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..probe import Probe
from ..scan import ScanAPI

ScanArrayType = numpy.typing.NDArray[numpy.floating[Any]]


@dataclass(frozen=True)
class TikeArrays:
    indexes: tuple[int, ...]
    scan: ScanArrayType
    probe: ProbeArrayType
    object_: ObjectArrayType


class TikeArrayConverter:
    PAD_WIDTH: Final[int] = 2

    def __init__(self, scanAPI: ScanAPI, probe: Probe, objectAPI: ObjectAPI,
                 diffractionDataset: ActiveDiffractionDataset) -> None:
        self._scanAPI = scanAPI
        self._probe = probe
        self._objectAPI = objectAPI
        self._diffractionDataset = diffractionDataset

    def getDiffractionData(self) -> DiffractionPatternData:
        data = self._diffractionDataset.getAssembledData()
        return numpy.fft.ifftshift(data, axes=(-2, -1))

    def exportToTike(self) -> TikeArrays:
        assembledIndexes = self._diffractionDataset.getAssembledIndexes()
        pixelSizeXInMeters = self._objectAPI.getPixelSizeXInMeters()
        pixelSizeYInMeters = self._objectAPI.getPixelSizeYInMeters()

        scanBoundingBoxInMeters = self._scanAPI.getBoundingBoxInMeters()
        xMinInMeters = scanBoundingBoxInMeters.rangeX.lower
        yMinInMeters = scanBoundingBoxInMeters.rangeY.lower

        selectedScan = self._scanAPI.getSelectedScan()
        indexes: list[int] = list()
        scanX: list[float] = list()
        scanY: list[float] = list()

        for index in assembledIndexes:
            try:
                point = selectedScan[index]
            except KeyError:
                continue

            indexes.append(index)
            scanX.append(self.PAD_WIDTH + float((point.x - xMinInMeters) / pixelSizeXInMeters))
            scanY.append(self.PAD_WIDTH + float((point.y - yMinInMeters) / pixelSizeYInMeters))

        probe = self._probe.getArray()
        object_ = self._objectAPI.getSelectedObjectArray()
        tikeObject = numpy.pad(object_, self.PAD_WIDTH, mode='constant', constant_values=0)

        return TikeArrays(
            indexes=tuple(indexes),
            scan=numpy.column_stack((scanY, scanX)).astype('float32'),
            probe=probe[numpy.newaxis, numpy.newaxis, ...].astype('complex64'),
            object_=tikeObject.astype('complex64'),
        )

    def importFromTike(self, arrays: TikeArrays) -> None:
        # TODO only update scan/probe/object if correction enabled
        pixelSizeXInMeters = self._objectAPI.getPixelSizeXInMeters()
        pixelSizeYInMeters = self._objectAPI.getPixelSizeYInMeters()

        scanBoundingBoxInMeters = self._scanAPI.getBoundingBoxInMeters()
        xMinInMeters = scanBoundingBoxInMeters.rangeX.lower
        yMinInMeters = scanBoundingBoxInMeters.rangeY.lower

        pointDict: dict[int, ScanPoint] = dict()

        for index, xy in zip(arrays.indexes, arrays.scan):
            uxInPixels = xy[1] - self.PAD_WIDTH
            uyInPixels = xy[0] - self.PAD_WIDTH
            xInMeters = xMinInMeters + Decimal(repr(uxInPixels)) * pixelSizeXInMeters
            yInMeters = yMinInMeters + Decimal(repr(uyInPixels)) * pixelSizeYInMeters
            pointDict[index] = ScanPoint(xInMeters, yInMeters)

        tabularScan = TabularScan('Tike', pointDict)
        self._scanAPI.insertScanIntoRepository(tabularScan, None)
        self._probe.setArray(arrays.probe[0, 0])
        self._objectAPI.insertObjectIntoRepository('Tike', arrays.object_, None)
