from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ptychodus.api.automation import WorkflowAPI, WorkflowProductAPI

from ..patterns import PatternsAPI
from ..product import ProductAPI
from ..workflow import WorkflowCore


class ConcreteWorkflowProductAPI(WorkflowProductAPI):

    def __init__(self, productAPI: ProductAPI, workflowCore: WorkflowCore,
                 productIndex: int) -> None:
        self._productAPI = productAPI
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
        self._productAPI.saveProduct(self._productIndex, filePath, fileType)


class ConcreteWorkflowAPI(WorkflowAPI):

    def __init__(self, patternsAPI: PatternsAPI, productAPI: ProductAPI,
                 workflowCore: WorkflowCore) -> None:
        self._patternsAPI = patternsAPI
        self._productAPI = productAPI
        self._workflowCore = workflowCore

    def _createProductAPI(self, productIndex: int) -> WorkflowProductAPI:
        return ConcreteWorkflowProductAPI(self._productAPI, self._workflowCore, productIndex)

    def openPatterns(self, filePath: Path, fileType: str) -> None:
        self._patternsAPI.openPatterns(filePath, fileType)

    def openProduct(self, filePath: Path, fileType: str) -> WorkflowProductAPI:
        productIndex = self._productAPI.openProduct(filePath, fileType)
        return self._createProductAPI(productIndex)

    def createProduct(self, name: str) -> WorkflowProductAPI:
        productIndex = self._productAPI.createNewProduct(name)
        return self._createProductAPI(productIndex)
