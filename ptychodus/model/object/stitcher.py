import numpy

from ...api.geometry import Point2D
from ...api.object import Object, ObjectArrayType, ObjectGeometry
from ...api.patterns import ImageExtent


class ObjectStitcher:

    def __init__(self, geometry: ObjectGeometry) -> None:
        self._geometry = geometry
        self._weights = numpy.zeros((geometry.heightInPixels, geometry.widthInPixels))
        self._array: ObjectArrayType = numpy.zeros_like(self._weights, dtype=complex)

    def addPatch(self, patchCenter: Point2D, patchArray: ObjectArrayType) -> None:
        geometry = self._geometry

        patchWidth = patchArray.shape[-1]
        patchRadiusXInMeters = geometry.pixelWidthInMeters * patchWidth / 2
        patchMinimumXInMeters = patchCenter.x - patchRadiusXInMeters
        ixBeginF, xi = divmod(patchMinimumXInMeters - geometry.minimumXInMeters,
                              geometry.pixelWidthInMeters)
        ixBegin = int(ixBeginF)
        ixEnd = ixBegin + patchWidth + 1
        ixSlice0 = slice(ixBegin, ixEnd)
        ixSlice1 = slice(ixBegin + 1, ixEnd + 1)

        patchHeight = patchArray.shape[-2]
        patchRadiusYInMeters = geometry.pixelHeightInMeters * patchHeight / 2
        patchMinimumYInMeters = patchCenter.y - patchRadiusYInMeters
        iyBeginF, eta = divmod(patchMinimumYInMeters - geometry.minimumYInMeters,
                               geometry.pixelHeightInMeters)
        iyBegin = int(iyBeginF)
        iyEnd = iyBegin + patchHeight + 1
        iySlice0 = slice(iyBegin, iyEnd)
        iySlice1 = slice(iyBegin + 1, iyEnd + 1)

        xiC = 1. - xi
        etaC = 1. - eta

        w00 = xiC * etaC
        w01 = xi * etaC
        w10 = xiC * eta
        w11 = xi * eta

        # FIXME online update for weighted sum (see Welford's online algorithm)
        self._array[:, iySlice0, ixSlice0] += w00 * patchArray
        self._array[:, iySlice0, ixSlice1] += w01 * patchArray
        self._array[:, iySlice1, ixSlice0] += w10 * patchArray
        self._array[:, iySlice1, ixSlice1] += w11 * patchArray

        self._weights[:, iySlice0, ixSlice0] += w00
        self._weights[:, iySlice0, ixSlice1] += w01
        self._weights[:, iySlice1, ixSlice0] += w10
        self._weights[:, iySlice1, ixSlice1] += w11

    def build(self) -> Object:
        return Object(
            array=self._array,
            pixelWidthInMeters=self._geometry.pixelWidthInMeters,
            pixelHeightInMeters=self._geometry.pixelHeightInMeters,
            centerXInMeters=self._geometry.centerXInMeters,
            centerYInMeters=self._geometry.centerYInMeters,
        )
