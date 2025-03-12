import numpy

from ptychodus.api.object import Object, ObjectArrayType, ObjectGeometry
from ptychodus.api.scan import ScanPoint


class ObjectStitcher:  # XXX
    def __init__(self, geometry: ObjectGeometry) -> None:
        self._geometry = geometry
        self._weights = numpy.zeros((geometry.height_px, geometry.width_px))
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
        patchRadiusXInMeters = geometry.pixel_width_m * patchWidth / 2
        patchMinimumXInMeters = patchCenter.position_x_m - patchRadiusXInMeters
        ixBeginF, xi = divmod(
            patchMinimumXInMeters - geometry.minimum_x_m,
            geometry.pixel_width_m,
        )
        ixBegin = int(ixBeginF)
        ixEnd = ixBegin + patchWidth
        ixSlice0 = slice(ixBegin, ixEnd)
        ixSlice1 = slice(ixBegin + 1, ixEnd + 1)

        patchHeight = patchArray.shape[-2]
        patchRadiusYInMeters = geometry.pixel_height_m * patchHeight / 2
        patchMinimumYInMeters = patchCenter.position_y_m - patchRadiusYInMeters
        iyBeginF, eta = divmod(
            patchMinimumYInMeters - geometry.minimum_y_m,
            geometry.pixel_height_m,
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

    def build(self) -> Object:  # FIXME multilayer objects?
        return Object(
            array=self._array,
            pixel_geometry=self._geometry.get_pixel_geometry(),
            center=self._geometry.get_center(),
        )
