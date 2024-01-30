from ...api.geometry import Point2D
from ...api.object import Object, ObjectInterpolator
from ...api.patterns import ImageExtent


class ObjectLinearInterpolator(ObjectInterpolator):

    def __init__(self, object_: Object) -> None:
        self._object = object_

    def getPatch(self, patchCenter: Point2D, patchExtent: ImageExtent) -> Object:
        geometry = self._object.getGeometry()

        patchWidth = patchExtent.widthInPixels
        patchRadiusXInMeters = geometry.pixelWidthInMeters * patchWidth / 2
        patchMinimumXInMeters = patchCenter.x - patchRadiusXInMeters
        ixBeginF, xi = divmod(patchMinimumXInMeters - geometry.minimumXInMeters,
                              geometry.pixelWidthInMeters)
        xiC = 1. - xi
        ixBegin = int(ixBeginF)
        ixEnd = ixBegin + patchWidth + 1
        ixSlice0 = slice(ixBegin, ixEnd)
        ixSlice1 = slice(ixBegin + 1, ixEnd + 1)

        patchHeight = patchExtent.heightInPixels
        patchRadiusYInMeters = geometry.pixelHeightInMeters * patchHeight / 2
        patchMinimumYInMeters = patchCenter.y - patchRadiusYInMeters
        iyBeginF, eta = divmod(patchMinimumYInMeters - geometry.minimumYInMeters,
                               geometry.pixelHeightInMeters)
        etaC = 1. - eta
        iyBegin = int(iyBeginF)
        iyEnd = iyBegin + patchHeight + 1
        iySlice0 = slice(iyBegin, iyEnd)
        iySlice1 = slice(iyBegin + 1, iyEnd + 1)

        patch = xiC * etaC * self._object.array[:, iySlice0, ixSlice0]
        patch += xi * etaC * self._object.array[:, iySlice0, ixSlice1]
        patch += xiC * eta * self._object.array[:, iySlice1, ixSlice0]
        patch += xi * eta * self._object.array[:, iySlice1, ixSlice1]

        return Object(
            array=patch,
            layerDistanceInMeters=self._object.layerDistanceInMeters,
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
            centerXInMeters=geometry.centerXInMeters,
            centerYInMeters=geometry.centerYInMeters,
        )
