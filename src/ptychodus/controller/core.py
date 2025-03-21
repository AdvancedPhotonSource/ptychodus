from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QAction

from ..model import ModelCore
from ..view import ViewCore
from .agent import AgentChatController, AgentController
from .automation import AutomationController
from .data import FileDialogFactory
from .image import ImageController
from .memory import MemoryController
from .object import ObjectController
from .patterns import PatternsController
from .probe import ProbeController
from .product import ProductController
from .ptychi import PtyChiViewControllerFactory
from .ptychonn import PtychoNNViewControllerFactory
from .reconstructor import ReconstructorController
from .scan import ScanController
from .settings import SettingsController
from .tike import TikeViewControllerFactory
from .workflow import WorkflowController


class ControllerCore:
    def __init__(self, model: ModelCore, view: ViewCore) -> None:
        self.view = view

        self._memoryController = MemoryController(model.memoryPresenter, view.memory_widget)
        self._fileDialogFactory = FileDialogFactory()
        self._ptyChiViewControllerFactory = PtyChiViewControllerFactory(
            model.ptyChiReconstructorLibrary
        )
        self._ptychonnViewControllerFactory = PtychoNNViewControllerFactory(
            model.ptychonnReconstructorLibrary
        )
        self._tikeViewControllerFactory = TikeViewControllerFactory(model.tikeReconstructorLibrary)
        self._settingsController = SettingsController(
            model.settingsRegistry,
            view.settingsView,
            view.settingsTableView,
            self._fileDialogFactory,
        )
        self._patternsImageController = ImageController.create_instance(
            model.patternVisualizationEngine,
            view.patternsImageView,
            view.statusBar(),
            self._fileDialogFactory,
        )
        self._patternsController = PatternsController(
            model.patterns_core.detectorSettings,
            model.patterns_core.patternSettings,
            model.patterns_core.patternSizer,
            model.patterns_core.patternsAPI,
            model.patterns_core.dataset,
            model.metadataPresenter,
            view.patternsView,
            self._patternsImageController,
            self._fileDialogFactory,
        )
        self._productController = ProductController.create_instance(
            model.patterns_core.dataset,
            model.productRepository,
            model.productAPI,
            view.productView,
            self._fileDialogFactory,
        )
        self._scanController = ScanController.create_instance(
            model.scanRepository,
            model.scanAPI,
            view.scanView,
            view.scanPlotView,
            self._fileDialogFactory,
        )
        self._probeImageController = ImageController.create_instance(
            model.probeVisualizationEngine,
            view.probeImageView,
            view.statusBar(),
            self._fileDialogFactory,
        )
        self._probeController = ProbeController.create_instance(
            model.probeRepository,
            model.probeAPI,
            self._probeImageController,
            model.probePropagator,
            model.probePropagatorVisualizationEngine,
            model.stxmSimulator,
            model.stxmVisualizationEngine,
            model.exposureAnalyzer,
            model.exposureVisualizationEngine,
            model.fluorescenceEnhancer,
            model.fluorescenceVisualizationEngine,
            view.probeView,
            self._fileDialogFactory,
        )
        self._objectImageController = ImageController.create_instance(
            model.objectVisualizationEngine,
            view.objectImageView,
            view.statusBar(),
            self._fileDialogFactory,
        )
        self._objectController = ObjectController.create_instance(
            model.objectRepository,
            model.objectAPI,
            self._objectImageController,
            model.fourierRingCorrelator,
            model.xmcdAnalyzer,
            model.xmcdVisualizationEngine,
            view.objectView,
            self._fileDialogFactory,
        )
        self._reconstructorController = ReconstructorController(
            model.reconstructorPresenter,
            model.productRepository,
            view.reconstructorView,
            view.reconstructorPlotView,
            self._productController.table_model,
            self._fileDialogFactory,
            [
                self._ptyChiViewControllerFactory,
                self._ptychonnViewControllerFactory,
                self._tikeViewControllerFactory,
            ],
        )
        self._workflowController = WorkflowController(
            model.workflowParametersPresenter,
            model.workflowAuthorizationPresenter,
            model.workflowStatusPresenter,
            model.workflowExecutionPresenter,
            view.workflowParametersView,
            view.workflowTableView,
            self._productController.table_model,
        )
        self._automationController = AutomationController.create_instance(
            model._automationCore,
            model.automationPresenter,
            model.automationProcessingPresenter,
            view.automationView,
            self._fileDialogFactory,
        )
        self._agentController = AgentController(
            model.argoSettings, model.agentPresenter, view.agentView
        )
        self._agentChatController = AgentChatController(
            model.chatHistory, model.agentPresenter, view.agentChatView
        )

        self._refreshDataTimer = QTimer()
        self._refreshDataTimer.timeout.connect(model.refreshActiveDataset)
        self._refreshDataTimer.start(1000)  # TODO make configurable

        view.workflowAction.setVisible(model.areWorkflowsSupported)

        self.swapCentralWidgets(view.patternsAction)
        view.patternsAction.setChecked(True)
        view.navigationActionGroup.triggered.connect(lambda action: self.swapCentralWidgets(action))

        view.agentAction.setVisible(model.agentPresenter.is_agent_supported)

    def showMainWindow(self, windowTitle: str) -> None:
        self.view.setWindowTitle(windowTitle)
        self.view.show()

    def swapCentralWidgets(self, action: QAction) -> None:
        index = action.data()
        self.view.left_panel.setCurrentIndex(index)
        self.view.right_panel.setCurrentIndex(index)
