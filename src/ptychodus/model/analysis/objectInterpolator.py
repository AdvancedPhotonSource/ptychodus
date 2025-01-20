from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import Object, ObjectInterpolator
from ptychodus.api.scan import ScanPoint


class ObjectLinearInterpolator(ObjectInterpolator):
    def __init__(self, object_: Object) -> None:
        self._object = object_

    def get_patch(self, patch_center: ScanPoint, patch_extent: ImageExtent) -> Object:
        geometry = self._object.getGeometry()

        patchWidth = patch_extent.widthInPixels
        patchRadiusXInMeters = geometry.pixelWidthInMeters * patchWidth / 2
        patchMinimumXInMeters = patch_center.positionXInMeters - patchRadiusXInMeters
        ixBeginF, xi = divmod(
            patchMinimumXInMeters - geometry.minimumXInMeters,
            geometry.pixelWidthInMeters,
        )
        ixBegin = int(ixBeginF)
        ixEnd = ixBegin + patchWidth
        ixSlice0 = slice(ixBegin, ixEnd)
        ixSlice1 = slice(ixBegin + 1, ixEnd + 1)

        patchHeight = patch_extent.heightInPixels
        patchRadiusYInMeters = geometry.pixelHeightInMeters * patchHeight / 2
        patchMinimumYInMeters = patch_center.positionYInMeters - patchRadiusYInMeters
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

        w00 = xiC * etaC
        w01 = xi * etaC
        w10 = xiC * eta
        w11 = xi * eta

        objectArray = self._object.getArray()
        patch = w00 * objectArray[:, iySlice0, ixSlice0]
        patch += w01 * objectArray[:, iySlice0, ixSlice1]
        patch += w10 * objectArray[:, iySlice1, ixSlice0]
        patch += w11 * objectArray[:, iySlice1, ixSlice1]

        return Object(  # FIXME multilayer objects
            array=patch,
            layerDistanceInMeters=self._object.layerDistanceInMeters,
            pixelGeometry=geometry.getPixelGeometry(),
            center=geometry.getCenter(),
        )
