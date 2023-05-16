from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Final

import numpy
import numpy.typing

from ...api.data import DiffractionPatternData
from ...api.object import ObjectArrayType
from ...api.probe import ProbeArrayType
from ...api.scan import ScanPoint, TabularScan
from ..data import ActiveDiffractionDataset
from ..object import ObjectAPI
from ..probe import ProbeAPI
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

    def __init__(self, scanAPI: ScanAPI, probeAPI: ProbeAPI, objectAPI: ObjectAPI,
                 diffractionDataset: ActiveDiffractionDataset) -> None:
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
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

        if selectedScan is None:
            raise ValueError('No scan is selected!')

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

        selectedProbe = self._probeAPI.getSelectedProbeArray()

        if selectedProbe is None:
            raise ValueError('No probe is selected!')

        selectedObject = self._objectAPI.getSelectedObjectArray()

        if selectedObject is None:
            raise ValueError('No object is selected!')

        tikeObject = numpy.pad(selectedObject, self.PAD_WIDTH, mode='constant', constant_values=0)

        return TikeArrays(
            indexes=tuple(indexes),
            scan=numpy.column_stack((scanY, scanX)).astype('float32'),
            probe=selectedProbe[numpy.newaxis, numpy.newaxis, ...].astype('complex64'),
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

        tabularScan = TabularScan(pointDict)
        self._scanAPI.insertItemIntoRepositoryFromScan('Tike', tabularScan)
        self._probeAPI.insertItemIntoRepositoryFromArray('Tike', arrays.probe[0, 0])
        self._objectAPI.insertItemIntoRepositoryFromArray('Tike', arrays.object_)
