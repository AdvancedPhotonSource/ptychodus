from enum import auto, Enum
import logging

import numpy

from ptychodus.api.geometry import PixelGeometry
from ptychodus.api.product import Product
from ptychodus.api.reconstructor import ReconstructInput
from ptychodus.api.scan import PositionSequence, ScanPoint

from ..diffraction import AssembledDiffractionDataset
from ..product import ProductAPI, ProductRepositoryItem

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
        product_api: ProductAPI,
    ) -> None:
        self._dataset = dataset
        self._product_api = product_api

    def get_product_item(self, input_product_index: int) -> ProductRepositoryItem:
        return self._product_api.get_item(input_product_index)

    def get_object_plane_pixel_geometry(self, input_product_index: int) -> PixelGeometry:
        input_product_item = self._product_api.get_item(input_product_index)
        object_geometry = input_product_item.get_geometry().get_object_geometry()
        return object_geometry.get_pixel_geometry()

    def match_diffraction_patterns_with_positions(
        self, input_product_index: int, index_filter: ScanIndexFilter = ScanIndexFilter.ALL
    ) -> ReconstructInput:
        input_product_item = self._product_api.get_item(input_product_index)
        input_product = input_product_item.get_product()
        pattern_indexes = [int(index) for index in self._dataset.get_assembled_indexes()]
        logger.debug(f'{pattern_indexes=}')
        position_indexes = [
            int(point.index) for point in input_product.positions if index_filter(point.index)
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
        point_iterator = iter(input_product.positions)

        for index in common_indexes:
            while True:
                point = next(point_iterator)

                if point.index == index:
                    point_list.append(point)
                    break

        probe = input_product.probes  # TODO remap if needed

        product = Product(
            metadata=input_product.metadata,
            positions=PositionSequence(point_list),
            probes=probe,
            object_=input_product.object_,
            losses=input_product.losses,
        )

        return ReconstructInput(patterns, self._dataset.get_processed_bad_pixels(), product)
