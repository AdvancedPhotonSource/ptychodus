from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QAction

from ..model import ModelCore
from ..view import ViewCore
from .automation import AutomationController
from .data import FileDialogFactory
from .memory import MemoryController
from .object import ObjectController
from .patterns import PatternsController
from .probe import ProbeController
from .product import ProductController
from .ptychonn import PtychoNNViewControllerFactory
from .reconstructor import ReconstructorController
from .scan import ScanController
from .settings import SettingsController
from .tike import TikeViewControllerFactory
from .workflow import WorkflowController


class ControllerCore:

    def __init__(self, model: ModelCore, view: ViewCore) -> None:
        self.view = view

        self._memoryController = MemoryController.createInstance(model.memoryPresenter,
                                                                 view.memoryProgressBar)
        self._fileDialogFactory = FileDialogFactory()
        self._ptychonnViewControllerFactory = PtychoNNViewControllerFactory(
            model.ptychonnReconstructorLibrary, self._fileDialogFactory)
        self._tikeViewControllerFactory = TikeViewControllerFactory(model.tikeReconstructorLibrary)
        self._settingsController = SettingsController.createInstance(model.settingsRegistry,
                                                                     view.settingsParametersView,
                                                                     view.settingsEntryView,
                                                                     self._fileDialogFactory)
        self._patternsController = PatternsController.createInstance(
            model.detectorPresenter, model.diffractionDatasetInputOutputPresenter,
            model.diffractionMetadataPresenter, model.diffractionDatasetPresenter,
            model.patternPresenter, model.detectorImagePresenter, view.patternsView,
            view.patternsImageView, self._fileDialogFactory)
        self._productController = ProductController.createInstance(model.productRepository,
                                                                   view.productView,
                                                                   self._fileDialogFactory)
        self._scanController = ScanController.createInstance(model.scanRepository, view.scanView,
                                                             view.scanPlotView,
                                                             self._fileDialogFactory)
        self._probeController = ProbeController.createInstance(
            model.probeRepository, model.probeImagePresenter, model.probePropagator,
            model.probePropagatorImagePresenter, view.probeView, view.probeImageView,
            view.statusBar(), self._fileDialogFactory)
        self._objectController = ObjectController.createInstance(
            model.objectRepository, model.objectImagePresenter, model.fourierRingCorrelator,
            model.dichroicAnalyzer, model.dichroicImagePresenter, view.objectView,
            view.objectImageView, view.statusBar(), self._fileDialogFactory)
        self._reconstructorParametersController = ReconstructorController.createInstance(
            model.reconstructorPresenter,
            model.productRepository,
            view.reconstructorParametersView,
            view.reconstructorPlotView,
            self._fileDialogFactory,
            self._productController.tableModel,
            [self._ptychonnViewControllerFactory, self._tikeViewControllerFactory],
        )
        self._workflowController = WorkflowController.createInstance(
            model.workflowParametersPresenter, model.workflowAuthorizationPresenter,
            model.workflowStatusPresenter, model.workflowExecutionPresenter,
            view.workflowParametersView, view.workflowTableView,
            self._productController.tableModel)
        self._automationController = AutomationController.createInstance(
            model._automationCore, model.automationPresenter, model.automationProcessingPresenter,
            view.automationView, self._fileDialogFactory)
        self._refreshDataTimer = QTimer()
        self._automationTimer = QTimer()
        self._processMessagesTimer = QTimer()

    @classmethod
    def createInstance(cls, model: ModelCore, view: ViewCore) -> ControllerCore:
        controller = cls(model, view)

        view.navigationActionGroup.triggered.connect(
            lambda action: controller.swapCentralWidgets(action))

        view.workflowAction.setVisible(model.areWorkflowsSupported)

        controller._refreshDataTimer.timeout.connect(model.refreshActiveDataset)
        controller._refreshDataTimer.start(1000)  # TODO make configurable

        controller._automationTimer.timeout.connect(model.refreshAutomationDatasets)
        controller._automationTimer.start(1000)  # TODO make configurable

        return controller

    def showMainWindow(self, windowTitle: str) -> None:
        self.view.setWindowTitle(windowTitle)
        self.view.show()

    def swapCentralWidgets(self, action: QAction) -> None:
        index = action.data()
        self.view.parametersWidget.setCurrentIndex(index)
        self.view.contentsWidget.setCurrentIndex(index)
