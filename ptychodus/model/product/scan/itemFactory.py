import numpy

from ptychodus.api.scan import Scan

from .builder import FromMemoryScanBuilder
from .builderFactory import ScanBuilderFactory
from .item import ScanRepositoryItem
from .settings import ScanSettings
from .transform import ScanPointTransform


class ScanRepositoryItemFactory:

    def __init__(self, rng: numpy.random.Generator, settings: ScanSettings,
                 builderFactory: ScanBuilderFactory) -> None:
        self._rng = rng
        self._settings = settings
        self._builderFactory = builderFactory

    def createDefault(self) -> ScanRepositoryItem:
        builder = self._builderFactory.createDefault()
        transform = ScanPointTransform(self._rng, self._settings)
        return ScanRepositoryItem(self._settings, builder, transform)

    def create(self, scan: Scan) -> ScanRepositoryItem:
        builder = FromMemoryScanBuilder(scan)
        transform = ScanPointTransform(self._rng, self._settings)
        return ScanRepositoryItem(self._settings, builder, transform)
