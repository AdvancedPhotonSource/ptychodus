import numpy

from ptychodus.api.object import Object, ObjectArrayType, ObjectGeometry
from ptychodus.api.scan import ScanPoint


class ObjectStitcher:
    def __init__(self, geometry: ObjectGeometry) -> None:
        self._geometry = geometry
        self._weights = numpy.zeros((geometry.heightInPixels, geometry.widthInPixels))
        self._array: ObjectArrayType = numpy.zeros_like(self._weights, dtype=complex)

    def _addPatchPart(
        self, ixSlice: slice, iySlice: slice, weight: float, patchArray: ObjectArrayType
    ) -> None:
        idx = numpy.s_[iySlice, ixSlice]
        self._weights[idx] += weight
        self._array[idx] += (patchArray - self._array[idx]) * weight / self._weights[idx]

    def addPatch(self, patchCenter: ScanPoint, patchArray: ObjectArrayType) -> None:
        geometry = self._geometry

        patchWidth = patchArray.shape[-1]
        patchRadiusXInMeters = geometry.pixelWidthInMeters * patchWidth / 2
        patchMinimumXInMeters = patchCenter.positionXInMeters - patchRadiusXInMeters
        ixBeginF, xi = divmod(
            patchMinimumXInMeters - geometry.minimumXInMeters,
            geometry.pixelWidthInMeters,
        )
        ixBegin = int(ixBeginF)
        ixEnd = ixBegin + patchWidth
        ixSlice0 = slice(ixBegin, ixEnd)
        ixSlice1 = slice(ixBegin + 1, ixEnd + 1)

        patchHeight = patchArray.shape[-2]
        patchRadiusYInMeters = geometry.pixelHeightInMeters * patchHeight / 2
        patchMinimumYInMeters = patchCenter.positionYInMeters - patchRadiusYInMeters
        iyBeginF, eta = divmod(
            patchMinimumYInMeters - geometry.minimumYInMeters,
            geometry.pixelHeightInMeters,
        )
        iyBegin = int(iyBeginF)
        iyEnd = iyBegin + patchHeight
        iySlice0 = slice(iyBegin, iyEnd)
        iySlice1 = slice(iyBegin + 1, iyEnd + 1)

        xiC = 1.0 - xi
        etaC = 1.0 - eta

        self._addPatchPart(ixSlice0, iySlice0, xiC * etaC, patchArray)
        self._addPatchPart(ixSlice1, iySlice0, xi * etaC, patchArray)
        self._addPatchPart(ixSlice0, iySlice1, xiC * eta, patchArray)
        self._addPatchPart(ixSlice1, iySlice1, xi * eta, patchArray)

    def build(self) -> Object:
        return Object(
            array=self._array,
            pixelGeometry=self._geometry.getPixelGeometry(),
            centerXInMeters=self._geometry.centerXInMeters,
            centerYInMeters=self._geometry.centerYInMeters,
        )
