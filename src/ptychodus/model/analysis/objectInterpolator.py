from ptychodus.api.geometry import ImageExtent
from ptychodus.api.object import Object
from ptychodus.api.scan import ScanPoint


class ObjectLinearInterpolator:  # XXX
    def __init__(self, object_: Object) -> None:
        self._object = object_

    def get_patch(self, patch_center: ScanPoint, patch_extent: ImageExtent) -> Object:
        geometry = self._object.get_geometry()

        patchWidth = patch_extent.width_px
        patchRadiusXInMeters = geometry.pixel_width_m * patchWidth / 2
        patchMinimumXInMeters = patch_center.position_x_m - patchRadiusXInMeters
        ixBeginF, xi = divmod(
            patchMinimumXInMeters - geometry.minimum_x_m,
            geometry.pixel_width_m,
        )
        ixBegin = int(ixBeginF)
        ixEnd = ixBegin + patchWidth
        ixSlice0 = slice(ixBegin, ixEnd)
        ixSlice1 = slice(ixBegin + 1, ixEnd + 1)

        patchHeight = patch_extent.height_px
        patchRadiusYInMeters = geometry.pixel_height_m * patchHeight / 2
        patchMinimumYInMeters = patch_center.position_y_m - patchRadiusYInMeters
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

        w00 = xiC * etaC
        w01 = xi * etaC
        w10 = xiC * eta
        w11 = xi * eta

        objectArray = self._object.get_array()
        patch = w00 * objectArray[:, iySlice0, ixSlice0]
        patch += w01 * objectArray[:, iySlice0, ixSlice1]
        patch += w10 * objectArray[:, iySlice1, ixSlice0]
        patch += w11 * objectArray[:, iySlice1, ixSlice1]

        return Object(  # FIXME multilayer objects
            array=patch,
            layer_distance_m=self._object.layer_distance_m,
            pixel_geometry=geometry.get_pixel_geometry(),
            center=geometry.get_center(),
        )
