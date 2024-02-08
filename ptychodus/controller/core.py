from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QAction

from ..model import ModelCore
from ..view import ViewCore
from .automation import AutomationController
from .data import FileDialogFactory
from .product import ProductController
from .memory import MemoryController
#from .object import ObjectImageController, ObjectController
from .patterns import PatternsController
#from .probe import ProbeImageController, ProbeController
#from .ptychonn import PtychoNNViewControllerFactory
#from .ptychopy import PtychoPyViewControllerFactory
#from .reconstructor import ReconstructorParametersController
#from .scan import ScanController
from .settings import SettingsController
#from .tike import TikeViewControllerFactory
from .workflow import WorkflowController


class ControllerCore:

    def __init__(self, model: ModelCore, view: ViewCore) -> None:
        self.view = view

        self._memoryController = MemoryController.createInstance(model.memoryPresenter,
                                                                 view.memoryProgressBar)
        self._fileDialogFactory = FileDialogFactory()
        # FIXME self._ptychopyViewControllerFactory = PtychoPyViewControllerFactory(model.ptychopyReconstructorLibrary)
        # FIXME self._ptychonnViewControllerFactory = PtychoNNViewControllerFactory(model.ptychonnReconstructorLibrary, self._fileDialogFactory)
        # FIXME self._tikeViewControllerFactory = TikeViewControllerFactory(model.tikeReconstructorLibrary)
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
        # FIXME self._scanController = ScanController.createInstance(model.scanRepositoryPresenter, view.scanView, view.scanPlotView, self._fileDialogFactory)
        # FIXME self._probeController = ProbeController.createInstance(model.apparatusPresenter, model.probeRepositoryPresenter, model.probeImagePresenter, view.probeView, view.probeImageView, self._fileDialogFactory)
        # FIXME self._objectController = ObjectController.createInstance(model.apparatusPresenter, model.objectRepositoryPresenter, model.objectImagePresenter, view.objectView, view.objectImageView, self._fileDialogFactory)
        # FIXME self._reconstructorParametersController = ReconstructorParametersController.createInstance(model.reconstructorPresenter, model.scanPresenter, model.probePresenter, model.objectPresenter, view.reconstructorParametersView, view.reconstructorPlotView, self._fileDialogFactory, [self._ptychopyViewControllerFactory, self._ptychonnViewControllerFactory, self._tikeViewControllerFactory], )
        self._workflowController = WorkflowController.createInstance(
            model.workflowParametersPresenter, model.workflowAuthorizationPresenter,
            model.workflowStatusPresenter, model.workflowExecutionPresenter,
            view.workflowParametersView, view.workflowTableView)
        self._automationController = AutomationController.createInstance(
            model._automationCore, model.automationPresenter, model.automationProcessingPresenter,
            view.automationView, self._fileDialogFactory)
        # FIXME self._monitorProbeController = ProbeImageController.createInstance(model.apparatusPresenter, model.probePresenter, model.probeImagePresenter, view.monitorProbeView.imageView, self._fileDialogFactory)
        # FIXME self._monitorObjectController = ObjectImageController.createInstance(model.apparatusPresenter, model.objectPresenter, model.objectImagePresenter, view.monitorObjectView.imageView, self._fileDialogFactory)
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
