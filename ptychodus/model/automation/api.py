from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ptychodus.api.automation import WorkflowAPI, WorkflowProductAPI

from ..patterns import PatternsAPI
from ..product import ProductRepository
from ..workflow import WorkflowCore


class ConcreteWorkflowProductAPI(WorkflowProductAPI):

    def __init__(self, productRepository: ProductRepository, workflowCore: WorkflowCore,
                 productIndex: int) -> None:
        self._productRepository = productRepository
        self._workflowCore = workflowCore
        self._productIndex = productIndex

    def openScan(self, filePath: Path, fileType: str) -> None:
        pass  # FIXME

    def buildProbe(self, builderName: str, builderParameters: Mapping[str, Any]) -> None:
        pass  # FIXME

    def buildObject(self, builderName: str, builderParameters: Mapping[str, Any]) -> None:
        pass  # FIXME

    def reconstruct(self) -> None:
        self._workflowCore.executeWorkflow(self._productIndex)

    def saveProduct(self, filePath: Path, fileType: str) -> None:
        self._productRepository.saveProduct(self._productIndex, filePath, fileType)


class ConcreteWorkflowAPI(WorkflowAPI):

    def __init__(self, patternsAPI: PatternsAPI, productRepository: ProductRepository,
                 workflowCore: WorkflowCore) -> None:
        self._patternsAPI = patternsAPI
        self._productRepository = productRepository
        self._workflowCore = workflowCore

    def _createProductAPI(self, productIndex: int) -> WorkflowProductAPI:
        return ConcreteWorkflowProductAPI(self._productRepository, self._workflowCore,
                                          productIndex)

    def openPatterns(self, filePath: Path, fileType: str) -> None:
        self._patternsAPI.openPatterns(filePath, fileType)

    def openProduct(self, filePath: Path, fileType: str) -> WorkflowProductAPI:
        productIndex = self._productRepository.openProduct(filePath, fileType)
        return self._createProductAPI(productIndex)

    def createProduct(self, name: str) -> WorkflowProductAPI:
        productIndex = self._productRepository.createNewProduct(name)
        return self._createProductAPI(productIndex)
