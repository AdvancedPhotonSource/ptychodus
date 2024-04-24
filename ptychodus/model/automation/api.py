from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ptychodus.api.automation import WorkflowAPI, WorkflowProductAPI

from ..patterns import PatternsAPI
from ..product import ProductAPI
from ..product.object import ObjectAPI
from ..product.probe import ProbeAPI
from ..product.scan import ScanAPI
from ..workflow import WorkflowCore


class ConcreteWorkflowProductAPI(WorkflowProductAPI):

    def __init__(self, productAPI: ProductAPI, scanAPI: ScanAPI, probeAPI: ProbeAPI,
                 objectAPI: ObjectAPI, workflowCore: WorkflowCore, productIndex: int) -> None:
        self._productAPI = productAPI
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._workflowCore = workflowCore
        self._productIndex = productIndex

    def openScan(self, filePath: Path, fileType: str) -> None:
        self._scanAPI.openScan(self._productIndex, filePath, fileType)

    def buildScan(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        self._scanAPI.buildScan(self._productIndex, builderName, builderParameters)

    def openProbe(self, filePath: Path, fileType: str) -> None:
        self._probeAPI.openProbe(self._productIndex, filePath, fileType)

    def buildProbe(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        self._probeAPI.buildProbe(self._productIndex, builderName, builderParameters)

    def openObject(self, filePath: Path, fileType: str) -> None:
        self._objectAPI.openObject(self._productIndex, filePath, fileType)

    def buildObject(self, builderName: str, builderParameters: Mapping[str, Any] = {}) -> None:
        self._objectAPI.buildObject(self._productIndex, builderName, builderParameters)

    def reconstruct(self) -> None:
        self._workflowCore.executeWorkflow(self._productIndex)

    def saveProduct(self, filePath: Path, fileType: str) -> None:
        self._productAPI.saveProduct(self._productIndex, filePath, fileType)


class ConcreteWorkflowAPI(WorkflowAPI):

    def __init__(self, patternsAPI: PatternsAPI, productAPI: ProductAPI, scanAPI: ScanAPI,
                 probeAPI: ProbeAPI, objectAPI: ObjectAPI, workflowCore: WorkflowCore) -> None:
        self._patternsAPI = patternsAPI
        self._productAPI = productAPI
        self._scanAPI = scanAPI
        self._probeAPI = probeAPI
        self._objectAPI = objectAPI
        self._workflowCore = workflowCore

    def _createProductAPI(self, productIndex: int) -> WorkflowProductAPI:
        return ConcreteWorkflowProductAPI(self._productAPI, self._scanAPI, self._probeAPI,
                                          self._objectAPI, self._workflowCore, productIndex)

    def openPatterns(self, filePath: Path, fileType: str) -> None:
        self._patternsAPI.openPatterns(filePath, fileType)

    def openProduct(self, filePath: Path, fileType: str) -> WorkflowProductAPI:
        productIndex = self._productAPI.openProduct(filePath, fileType)
        return self._createProductAPI(productIndex)

    def createProduct(self, name: str) -> WorkflowProductAPI:
        productIndex = self._productAPI.createNewProduct(name)
        return self._createProductAPI(productIndex)
