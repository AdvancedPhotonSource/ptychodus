from ...api.scan import Scan
from .builder import FromMemoryScanBuilder
from .item import ScanRepositoryItem
from .transform import ScanPointTransformFactory


class ScanRepositoryItemFactory:

    def __init__(self, transformFactory: ScanPointTransformFactory) -> None:
        self._transformFactory = transformFactory

    def create(self, scan: Scan) -> ScanRepositoryItem:
        builder = FromMemoryScanBuilder(scan)
        transform = self._transformFactory.createDefaultTransform()
        return ScanRepositoryItem(builder, transform)
