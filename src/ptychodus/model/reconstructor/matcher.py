from enum import auto, Enum
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.scan import Scan, ScanPoint

from ..patterns import AssembledDiffractionDataset
from ..product import ProductRepository

logger = logging.getLogger(__name__)


class ScanIndexFilter(Enum):
    """filters scan points by index"""

    ALL = auto()
    ODD = auto()
    EVEN = auto()

    def __call__(self, index: int) -> bool:
        """include scan point if true, exclude otherwise"""
        if self is ScanIndexFilter.ODD:
            return index & 1 != 0
        elif self is ScanIndexFilter.EVEN:
            return index & 1 == 0

        return True


class DiffractionPatternPositionMatcher:
    def __init__(
        self,
        diffractionDataset: AssembledDiffractionDataset,
        productRepository: ProductRepository,
    ) -> None:
        self._diffractionDataset = diffractionDataset
        self._productRepository = productRepository

    def getProductName(self, inputProductIndex: int) -> str:
        inputProductItem = self._productRepository[inputProductIndex]
        return inputProductItem.getName()

    def getObjectPlanePixelGeometry(self, inputProductIndex: int) -> PixelGeometry:
        inputProductItem = self._productRepository[inputProductIndex]
        objectGeometry = inputProductItem.getGeometry().getObjectGeometry()
        return objectGeometry.getPixelGeometry()

    def matchDiffractionPatternsWithPositions(
        self, inputProductIndex: int, indexFilter: ScanIndexFilter = ScanIndexFilter.ALL
    ) -> ReconstructInput:
        inputProductItem = self._productRepository[inputProductIndex]
        inputProduct = inputProductItem.getProduct()
        dataIndexes = self._diffractionDataset.get_assembled_indexes()
        scanIndexes = [point.index for point in inputProduct.scan if indexFilter(point.index)]
        commonIndexes = sorted(set(dataIndexes).intersection(scanIndexes))

        patterns = numpy.take(
            self._diffractionDataset.get_assembled_patterns(),
            commonIndexes,
            axis=0,
        )

        pointList: list[ScanPoint] = list()
        pointIter = iter(inputProduct.scan)

        for index in commonIndexes:
            while True:
                point = next(pointIter)

                if point.index == index:
                    pointList.append(point)
                    break

        probe = inputProduct.probe  # TODO remap if needed

        product = Product(
            metadata=inputProduct.metadata,
            scan=Scan(pointList),
            probe=probe,
            object_=inputProduct.object_,
            costs=inputProduct.costs,
        )

        return ReconstructInput(
            patterns, self._diffractionDataset.get_processed_bad_pixels(), product
        )
