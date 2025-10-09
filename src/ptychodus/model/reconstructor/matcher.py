from enum import auto, Enum
import logging

import numpy

from ptychodus.api.product import Product
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.scan import PositionSequence, ScanPoint

from ..diffraction import AssembledDiffractionDataset
from ..product import ProductRepositoryItem

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
        dataset: AssembledDiffractionDataset,
    ) -> None:
        self._dataset = dataset

    def match_diffraction_patterns_with_positions(
        self,
        product_item: ProductRepositoryItem,
        index_filter: ScanIndexFilter = ScanIndexFilter.ALL,
    ) -> ReconstructInput:
        product = product_item.get_product()
        pattern_indexes = [int(index) for index in self._dataset.get_assembled_indexes()]
        logger.debug(f'{pattern_indexes=}')
        position_indexes = [
            int(point.index) for point in product.positions if index_filter(point.index)
        ]
        logger.debug(f'{position_indexes=}')
        common_indexes = sorted(set(pattern_indexes).intersection(position_indexes))
        logger.debug(f'{common_indexes=}')

        patterns = numpy.take(
            self._dataset.get_assembled_patterns(),
            common_indexes,
            axis=0,
        )

        point_list: list[ScanPoint] = list()
        point_iterator = iter(product.positions)

        for index in common_indexes:
            while True:
                point = next(point_iterator)

                if point.index == index:
                    point_list.append(point)
                    break

        probe = product.probes  # TODO remap if needed

        product = Product(
            metadata=product.metadata,
            positions=PositionSequence(point_list),
            probes=probe,
            object_=product.object_,
            losses=product.losses,
        )

        bad_pixels = self._dataset.get_bad_pixels()

        if bad_pixels is None:
            raise ValueError('bad_pixels is None!')

        return ReconstructInput(patterns, bad_pixels, product)
