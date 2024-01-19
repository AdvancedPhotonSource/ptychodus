import logging

from ...api.geometry import Box2D
from ...api.observer import Observable
from ...api.parametric import ParameterRepository
from ...api.scan import Scan
from .builder import ScanBuilder
from .transform import ScanPointTransform

logger = logging.getLogger(__name__)


class ScanRepositoryItem(ParameterRepository):

    def __init__(self, builder: ScanBuilder, transform: ScanPointTransform) -> None:
        super().__init__('Scan')
        self._builder = builder
        self._transform = transform
        self._scan = builder.build()
        self._metrics = builder.getScanMetrics()  # FIXME metrics should reflect transform

        self._addParameterRepository(self._builder)
        self._addParameterRepository(self._transform)

        self._rebuild()

    # FIXME getTransformed and getUntransformed
    def getScan(self) -> Scan:
        return self._transformedScan

    def _setScan(self, scan: Scan) -> None:
        # FIXME apply transform (per point)
        # FIXME calculate metrics (per point)
        self._scan = scan
        self.notifyObservers()

    def getBuilder(self) -> ScanBuilder:
        return self._builder

    def setBuilder(self, builder: ScanBuilder) -> None:
        self._removeParameterRepository(self._builder)
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._addParameterRepository(self._builder)
        self._rebuild()

    def getBoundingBoxInMeters(self) -> Box2D | None:
        return self._metrics.getBoundingBoxInMeters()

    def _rebuild(self) -> None:
        try:
            scan = self._builder.build()
        except Exception:
            logger.exception('Failed to reinitialize scan!')
        else:
            self._setScan(scan)

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
