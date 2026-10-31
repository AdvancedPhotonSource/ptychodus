from __future__ import annotations

from ptychodus.api.xmcd import XMCDResult, estimate_xmcd

from ..product import ProductAPI, ProductRepository


class XMCDAnalyzer:
    def __init__(self, repository: ProductRepository, product_api: ProductAPI) -> None:
        self._repository = repository
        self._product_api = product_api

    def analyze(self, lcp_product_index: int, rcp_product_index: int) -> XMCDResult:
        lcp_product = self._repository[lcp_product_index].get_product()
        rcp_product = self._repository[rcp_product_index].get_product()
        return estimate_xmcd(lcp_product, rcp_product)

    def insert_products(self, result: XMCDResult) -> tuple[int, int]:
        """Add the structural and magnetic products to the repository, returning their indexes."""
        structural_index = self._product_api.insert_product(result.structural_product)
        magnetic_index = self._product_api.insert_product(result.magnetic_product)
        return structural_index, magnetic_index
