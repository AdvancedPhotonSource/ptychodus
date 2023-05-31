from __future__ import annotations

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QAction

from ..model import ModelCore
from ..view import ViewCore
from .automation import AutomationController
from .data import DataParametersController, FileDialogFactory
from .detector import DetectorController
from .object import ObjectImageController, ObjectController
from .probe import ProbeImageController, ProbeController
from .ptychonn import PtychoNNViewControllerFactory
from .ptychopy import PtychoPyViewControllerFactory
from .reconstructor import ReconstructorParametersController, ReconstructorPlotController
from .scan import ScanController
from .settings import SettingsController
from .tike import TikeViewControllerFactory
from .workflow import WorkflowController


class ControllerCore:

    def __init__(self, model: ModelCore, view: ViewCore) -> None:
        self.model = model
        self.view = view

        self._fileDialogFactory = FileDialogFactory()

        self._ptychopyViewControllerFactory = PtychoPyViewControllerFactory(
            model.ptychopyReconstructorLibrary)
        self._ptychonnViewControllerFactory = PtychoNNViewControllerFactory(
            model.ptychonnReconstructorLibrary, self._fileDialogFactory)
        self._tikeViewControllerFactory = TikeViewControllerFactory(model.tikeReconstructorLibrary)

        self._settingsController = SettingsController.createInstance(model.settingsRegistry,
                                                                     view.settingsParametersView,
                                                                     view.settingsEntryView,
                                                                     self._fileDialogFactory)
        self._detectorController = DetectorController.createInstance(
            model.detectorPresenter, model.apparatusPresenter, model.diffractionDatasetPresenter,
            model.detectorImagePresenter, view.detectorView, view.detectorImageView,
            self._fileDialogFactory)
        self._scanController = ScanController.createInstance(model.scanRepositoryPresenter,
                                                             view.scanView, view.scanPlotView,
                                                             self._fileDialogFactory)
        self._probeController = ProbeController.createInstance(model.apparatusPresenter,
                                                               model.probeRepositoryPresenter,
                                                               model.probeImagePresenter,
                                                               view.probeView, view.probeImageView,
                                                               self._fileDialogFactory)
        self._objectController = ObjectController.createInstance(
            model.apparatusPresenter, model.objectRepositoryPresenter, model.objectImagePresenter,
            view.objectView, view.objectImageView, self._fileDialogFactory)
        self._dataParametersController = DataParametersController.createInstance(
            model.settingsRegistry, model.diffractionDatasetInputOutputPresenter,
            model.diffractionDatasetPresenter, model.metadataPresenter, model.patternPresenter,
            view.dataParametersView, view.dataTableView, self._fileDialogFactory)
        self._reconstructorParametersController = ReconstructorParametersController.createInstance(
            model.reconstructorPresenter,
            model.reconstructorPlotPresenter,
            model.scanPresenter,
            model.probePresenter,
            model.objectPresenter,
            view.reconstructorParametersView,
            [
                self._ptychopyViewControllerFactory, self._ptychonnViewControllerFactory,
                self._tikeViewControllerFactory
            ],
        )
        self._reconstructorPlotController = ReconstructorPlotController.createInstance(
            model.reconstructorPlotPresenter, view.reconstructorPlotView)
        self._workflowController = WorkflowController.createInstance(
            model.workflowParametersPresenter, model.workflowAuthorizationPresenter,
            model.workflowStatusPresenter, model.workflowExecutionPresenter,
            view.workflowParametersView, view.workflowTableView)
        self._automationController = AutomationController.createInstance(
            model._automationCore, model.automationPresenter, model.automationProcessingPresenter,
            view.automationView, self._fileDialogFactory)
        self._monitorProbeController = ProbeImageController.createInstance(
            model.probePresenter, model.probeImagePresenter, view.monitorProbeView.imageView,
            self._fileDialogFactory)
        self._monitorObjectController = ObjectImageController.createInstance(
            model.objectPresenter, model.objectImagePresenter, view.monitorObjectView.imageView,
            self._fileDialogFactory)
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

        if model.rpcMessageService and model.rpcMessageService.isActive:
            controller._processMessagesTimer.timeout.connect(
                model.rpcMessageService.processMessages)
            controller._processMessagesTimer.start(1000)  # TODO make configurable

        return controller

    def swapCentralWidgets(self, action: QAction) -> None:
        index = action.data()
        self.view.parametersWidget.setCurrentIndex(index)
        self.view.contentsWidget.setCurrentIndex(index)
