from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QAction

from ..model import ModelCore
from ..view import ViewCore
from .data import DataParametersController, FileDialogFactory
from .detector import (CropController, DatasetImageController, DatasetParametersController,
                       DetectorController)
from .object import ObjectImageController, ObjectParametersController
from .probe import ProbeImageController, ProbeParametersController
from .ptychopy import PtychoPyViewControllerFactory
from .reconstructor import ReconstructorParametersController, ReconstructorPlotController
from .scan import ScanController
from .settings import SettingsController, SettingsImportController
from .tike import TikeViewControllerFactory
from .workflow import WorkflowController


class ControllerCore:

    def __init__(self, model: ModelCore, view: ViewCore) -> None:
        self.model = model
        self.view = view

        self._fileDialogFactory = FileDialogFactory()

        self._ptychopyViewControllerFactory = PtychoPyViewControllerFactory(model.ptychopyBackend)
        self._tikeViewControllerFactory = TikeViewControllerFactory(model.tikeBackend)

        self._settingsImportController = SettingsImportController.createInstance(
            model.probePresenter, model.objectPresenter, model.velociprobePresenter,
            view.settingsParametersView.importDialog)
        self._settingsController = SettingsController.createInstance(model.settingsRegistry,
                                                                     view.settingsParametersView,
                                                                     view.settingsEntryView,
                                                                     self._fileDialogFactory)
        self._detectorController = DetectorController.createInstance(
            model.detectorPresenter, view.detectorParametersView.detectorView)
        self._datasetParametersController = DatasetParametersController.createInstance(
            model.dataFilePresenter, model.diffractionDatasetPresenter,
            view.detectorParametersView.datasetView)
        self._cropController = CropController.createInstance(
            model.cropPresenter, view.detectorParametersView.imageCropView)
        self._datasetImageController = DatasetImageController.createInstance(
            model.diffractionDatasetPresenter, model.detectorImagePresenter,
            view.detectorImageView, self._fileDialogFactory)
        self._probeParametersController = ProbeParametersController.createInstance(
            model.probePresenter, view.probeParametersView, model.probeImagePresenter,
            view.probeImageView, self._fileDialogFactory)
        self._scanController = ScanController.createInstance(model.scanPresenter,
                                                             view.scanParametersView,
                                                             view.scanPlotView,
                                                             self._fileDialogFactory)
        self._objectParametersController = ObjectParametersController.createInstance(
            model.objectPresenter, view.objectParametersView, self._fileDialogFactory)
        self._objectImageController = ObjectImageController.createInstance(
            model.objectPresenter, model.objectImagePresenter, view.objectImageView,
            self._fileDialogFactory)
        self._dataParametersController = DataParametersController.createInstance(
            model.dataFilePresenter, view.dataParametersView, view.dataTableView,
            self._fileDialogFactory)
        self._reconstructorParametersController = ReconstructorParametersController.createInstance(
            model.reconstructorPresenter, view.reconstructorParametersView,
            [self._ptychopyViewControllerFactory, self._tikeViewControllerFactory])
        self._reconstructorPlotController = ReconstructorPlotController.createInstance(
            model.reconstructorPlotPresenter, view.reconstructorPlotView)
        self._workflowController = WorkflowController.createInstance(model.workflowPresenter,
                                                                     view.workflowParametersView,
                                                                     view.workflowTableView)
        self._monitorProbeController = ProbeImageController.createInstance(
            model.probePresenter, model.probeImagePresenter, view.monitorProbeView.imageView,
            self._fileDialogFactory)
        self._monitorObjectController = ObjectImageController.createInstance(
            model.objectPresenter, model.objectImagePresenter, view.monitorObjectView.imageView,
            self._fileDialogFactory)
        self._processMessagesTimer = QTimer()

    @classmethod
    def createInstance(cls, model: ModelCore, view: ViewCore):
        controller = cls(model, view)

        view.navigationActionGroup.triggered.connect(
            lambda action: controller.swapCentralWidgets(action))

        if model.rpcMessageService:
            controller._processMessagesTimer.timeout.connect(
                model.rpcMessageService.processMessages)
            controller._processMessagesTimer.start(1000)  # TODO make configurable

        return controller

    def swapCentralWidgets(self, action: QAction) -> None:
        index = action.data()
        self.view.parametersWidget.setCurrentIndex(index)
        self.view.contentsWidget.setCurrentIndex(index)
