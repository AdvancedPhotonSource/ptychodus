import numpy

from ptychodus.api.scan import Scan

from .builder import FromMemoryScanBuilder
from .cartesian import CartesianScanBuilder, CartesianScanVariant
from .item import ScanRepositoryItem
from .transform import ScanPointTransform


class ScanRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator) -> None:
        self._rng = rng

    def createDefault(self) -> ScanRepositoryItem:
        builder = CartesianScanBuilder(CartesianScanVariant.RECTANGULAR_RASTER)
        transform = ScanPointTransform(self._rng)
        return ScanRepositoryItem(builder, transform)

    def create(self, scan: Scan) -> ScanRepositoryItem:
        builder = FromMemoryScanBuilder(scan)
        transform = ScanPointTransform(self._rng)
        return ScanRepositoryItem(builder, transform)
