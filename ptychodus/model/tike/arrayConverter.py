from dataclasses import dataclass
import logging

import numpy
import numpy.typing

from ...api.data import DataArrayType, DataFile
from ...api.object import ObjectArrayType
from ...api.probe import ProbeArrayType
from ..object import Object
from ..probe import Apparatus, Probe
from ..scan import Scan

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

    @property
    def numberOfFrames(self) -> int:
        return min(len(self._scan), len(self._dataFile))

    def getDiffractionData(self) -> DataArrayType:
        numFrames = self.numberOfFrames
        data = self._dataFile.getDiffractionData()
        return numpy.fft.ifftshift(data[:numFrames, ...], axes=(-2, -1))

    def exportToTike(self) -> TikeArrays:
        numFrames = self.numberOfFrames

        pixelSizeXInMeters = self._apparatus.getObjectPlanePixelSizeXInMeters()
        pixelSizeYInMeters = self._apparatus.getObjectPlanePixelSizeYInMeters()

        scanX: list[float] = list()
        scanY: list[float] = list()

        for n, point in enumerate(self._scan):
            scanX.append(float(point.x / pixelSizeXInMeters))
            scanY.append(float(point.y / pixelSizeYInMeters))

            if n > numFrames:
                break

        probe = self._probe.getArray()
        padX = probe.shape[-1] // 2
        padY = probe.shape[-2] // 2

        shiftX = padX - min(scanX)
        shiftY = padY - min(scanY)

        for n in range(numFrames):
            scanX[n] += shiftX
            scanY[n] += shiftY

        spanX = max(scanX)
        spanY = max(scanY)

        object_ = self._object.getArray()
        logger.debug(f'Ptychodus object: {object_.shape}')

        tikeObjectShapeX = probe.shape[-1] + int(spanX) + padX
        tikeObjectShapeY = probe.shape[-2] + int(spanY) + padY
        logger.debug(f'Tike object: ({tikeObjectShapeY,tikeObjectShapeX})')

        # vvv FIXME vvv
        objectDiffX = tikeObjectShapeX - object_.shape[-1]
        objectDiffY = tikeObjectShapeY - object_.shape[-2]

        objectPadX = (padX, objectDiffX - padX)  # TODO verify offset
        objectPadY = (padY, objectDiffY - padY)  # TODO verify offset

        return TikeArrays(
            scan=numpy.column_stack((scanY, scanX)).astype('float32'),
            probe=probe[numpy.newaxis, numpy.newaxis, ...].astype('complex64'),
            object_=numpy.pad(object_, (objectPadY, objectPadX)).astype('complex64'),
        )

    def importFromTike(self, arrays: TikeArrays) -> None:
        # TODO only update if correction enabled
        # FIXME shift and scale self._scan.setScanPoints(...) using arrays.scan
        self._object.setArray(arrays.object_)  # FIXME trim padding
        self._probe.setArray(arrays.probe[0, 0])
