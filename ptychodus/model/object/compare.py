import logging

from ...api.object import Object
from ...api.observer import Observable
from ...api.visualize import FourierRingCorrelation
from ..product import ObjectRepository

logger = logging.getLogger(__name__)


class CompareObjectBuilder(Observable):

    def __init__(self, repository: ObjectRepository) -> None:
        super().__init__()
        self._repository = repository
        self._productIndex1 = 0
        self._productIndex2 = 0

    def subtract(self) -> Object:
        object1 = self._repository[self._productIndex1].getObject()
        object2 = self._repository[self._productIndex2].getObject()
        geometry = object2.getGeometry()

        # FIXME object consistency checks
        # FIXME what if pixel geometry differs?

        return Object(
            array=object1.array - object2.array,
            layerDistanceInMeters=object2.layerDistanceInMeters,
            pixelWidthInMeters=geometry.pixelWidthInMeters,
            pixelHeightInMeters=geometry.pixelHeightInMeters,
            centerXInMeters=geometry.centerXInMeters,
            centerYInMeters=geometry.centerYInMeters,
        )

    def getFourierRingCorrelation(self) -> FourierRingCorrelation:
        object1 = self._repository[self._productIndex1].getObject()
        object2 = self._repository[self._productIndex2].getObject()

        # FIXME object consistency checks
        # FIXME what if pixel geometry differs?

        # TODO support multiple layers
        return FourierRingCorrelation.calculate(object1.getLayer(0), object2.getLayer(0),
                                                object2.getPixelGeometry())
