from dataclasses import dataclass
import logging

import numpy
import numpy.typing

from ...api.data import DataArrayType, DataFile
from ...api.object import ObjectArrayType
from ...api.probe import ProbeArrayType
from ...api.scan import ScanPoint
from ..object import Object
from ..probe import Apparatus, Probe
from ..scan import Scan, ScanInitializerFactory, ScanRepository

logger = logging.getLogger(__name__)

ScanArrayType = numpy.typing.NDArray[numpy.floating]


@dataclass(frozen=True)
class TikeArrays:
    scan: ScanArrayType
    probe: ProbeArrayType
    object_: ObjectArrayType


class TikeArrayConverter:

    def __init__(self, apparatus: Apparatus, scan: Scan, probe: Probe, object_: Object,
                 dataFile: DataFile) -> None:
        self._apparatus = apparatus
        self._scan = scan
        self._probe = probe
        self._object = object_
        self._dataFile = dataFile

    def getDiffractionData(self) -> DataArrayType:
        data = self._dataFile.getDiffractionData()
        return numpy.fft.ifftshift(data, axes=(-2, -1))

    def exportToTike(self) -> TikeArrays:
        pixelSizeXInMeters = self._apparatus.getObjectPlanePixelSizeXInMeters()
        pixelSizeYInMeters = self._apparatus.getObjectPlanePixelSizeYInMeters()

        scanX: list[float] = list()
        scanY: list[float] = list()

        for point in self._scan:
            scanX.append(float(point.x / pixelSizeXInMeters))
            scanY.append(float(point.y / pixelSizeYInMeters))

        probe = self._probe.getArray()
        padX = probe.shape[-1] // 2
        padY = probe.shape[-2] // 2

        shiftX = padX - min(scanX)
        shiftY = padY - min(scanY)

        for n in range(len(self._scan)):
            scanX[n] += shiftX
            scanY[n] += shiftY

        logger.debug(f'Scan {min(scanX),min(scanY)} -> {max(scanX),max(scanY)}')

        spanX = max(scanX)
        spanY = max(scanY)

        tikeObjectShapeX = probe.shape[-1] + int(spanX) + padX
        tikeObjectShapeY = probe.shape[-2] + int(spanY) + padY
        tikeObject = numpy.ones(shape=(tikeObjectShapeY, tikeObjectShapeX), dtype='complex64')
        logger.debug(f'Tike object: {tikeObjectShapeY,tikeObjectShapeX}')

        object_ = self._object.getArray()
        logger.debug(f'Ptychodus object: {object_.shape}')

        tikeObject[padY:padY + object_.shape[-2], padX:padX + object_.shape[-1]] = object_

        return TikeArrays(
            scan=numpy.column_stack((scanY, scanX)).astype('float32'),
            probe=probe[numpy.newaxis, numpy.newaxis, ...].astype('complex64'),
            object_=tikeObject,
        )

    def importFromTike(self, arrays: TikeArrays) -> None:
        pixelSizeXInMeters = self._apparatus.getObjectPlanePixelSizeXInMeters()
        pixelSizeYInMeters = self._apparatus.getObjectPlanePixelSizeYInMeters()

        probe = self._probe.getArray()
        padX = probe.shape[-1] // 2
        padY = probe.shape[-2] // 2

        shiftX = padX - min(float(point.x / pixelSizeXInMeters) for point in self._scan)
        shiftY = padY - min(float(point.y / pixelSizeYInMeters) for point in self._scan)

        pointList: list[ScanPoint] = list()

        for xy in arrays.scan:
            x_m = (xy[1] - shiftX) * pixelSizeXInMeters
            y_m = (xy[0] - shiftY) * pixelSizeYInMeters
            pointList.append(ScanPoint(x_m, y_m))

        # FIXME finish; only update stuff if correction enabled
        #tabularScan = self._scanInitializerFactory.createTabularInitializer(pointList, None)
        #self._scanRepository.insertScan(tabularScan)

        self._probe.setArray(arrays.probe[0, 0])

        object_ = self._object.getArray()
        self._object.setArray(arrays.object_[padY:padY + object_.shape[-2],
                                             padX:padX + object_.shape[-1]])
