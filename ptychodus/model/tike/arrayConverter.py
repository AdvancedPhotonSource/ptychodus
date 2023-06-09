from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy
import numpy.typing

from ...api.data import DiffractionPatternData
from ...api.object import ObjectArrayType, ObjectPoint
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
        selectedScan = self._scanAPI.getSelectedScan()

        if selectedScan is None:
            raise ValueError('No scan is selected!')

        selectedProbe = self._probeAPI.getSelectedProbeArray()

        if selectedProbe is None:
            raise ValueError('No probe is selected!')

        selectedObject = self._objectAPI.getSelectedObjectArray()

        if selectedObject is None:
            raise ValueError('No object is selected!')

        indexes: list[int] = list()
        scanX: list[float] = list()
        scanY: list[float] = list()

        for index in self._diffractionDataset.getAssembledIndexes():
            try:
                objectPoint = self._objectAPI.mapScanPointToObjectPoint(selectedScan[index])
            except KeyError:
                continue

            indexes.append(index)
            scanX.append(float(objectPoint.x))
            scanY.append(float(objectPoint.y))

        return TikeArrays(
            indexes=tuple(indexes),
            scan=numpy.column_stack((scanY, scanX)).astype('float32'),
            probe=selectedProbe[numpy.newaxis, numpy.newaxis, ...].astype('complex64'),
            object_=selectedObject.astype('complex64'),
        )

    def importFromTike(self, arrays: TikeArrays) -> None:
        # TODO only update scan/probe/object if correction enabled
        pointDict: dict[int, ScanPoint] = dict()

        for index, xy in zip(arrays.indexes, arrays.scan):
            objectPoint = ObjectPoint(
                x=Decimal.from_float(xy[1]),
                y=Decimal.from_float(xy[0]),
            )
            pointDict[index] = self._objectAPI.mapObjectPointToScanPoint(objectPoint)

        # FIXME extract to reconstructor module: pass in copy of selected scan/probe/object
        # to reconstructor so correct items are in restart file and on the monitor screen
        tabularScan = TabularScan(pointDict)
        scanName = self._scanAPI.insertItemIntoRepositoryFromScan('Tike', tabularScan)

        if scanName:
            self._scanAPI.selectItem(scanName)

        probeName = self._probeAPI.insertItemIntoRepositoryFromArray('Tike', arrays.probe[0, 0])

        if probeName:
            self._probeAPI.selectItem(probeName)

        objectName = self._objectAPI.insertItemIntoRepositoryFromArray('Tike', arrays.object_)

        if objectName:
            self._objectAPI.selectItem(objectName)
