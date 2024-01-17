import logging
import threading

from ...api.geometry import Box2D
from ...api.observer import Observable, Observer
from ...api.scan import Scan
from .builder import FromMemoryScanBuilder, ScanBuilder
from .metrics import ScanMetrics
from .transform import ScanPointTransform

logger = logging.getLogger(__name__)


class ScanRepositoryItem(Observable, Observer):

    def __init__(self, scan: Scan) -> None:
        super().__init__()
        self._builder: ScanBuilder = FromMemoryScanBuilder(scan)
        self._scan = scan
        self._scanLock = threading.Lock()
        self._scanChanged = threading.Event()
        self._metrics = ScanMetrics()  # FIXME
        self._transform = ScanPointTransform.PXPY  # FIXME
        self._rebuild()

    # FIXME getTransformed and getUntransformed
    def getScan(self) -> Scan:
        with self._scanLock:
            return self._scan

    def getBuilder(self) -> ScanBuilder:
        return self._builder

    def setBuilder(self, builder: ScanBuilder) -> None:
        self._builder.removeObserver(self)
        self._builder = builder
        self._builder.addObserver(self)
        self._rebuild()

    def getBoundingBoxInMeters(self) -> Box2D | None:
        return self._metrics.getBoundingBoxInMeters()

    def notifyObserversIfChanged(self) -> None:
        if self._scanChanged.is_set():
            self._scanChanged.clear()
            self.notifyObservers()

    def _rebuild(self) -> None:
        try:
            scan = self._builder.build()
        except Exception:
            logger.exception('Failed to reinitialize scan!')
        else:
            with self._scanLock:
                self._scan = scan

            self._scanChanged.set()

    def update(self, observable: Observable) -> None:
        if observable is self._builder:
            self._rebuild()
