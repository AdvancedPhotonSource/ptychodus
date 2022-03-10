from PyQt5.QtWidgets import QApplication, QAction

from ..model import ModelCore
from ..view import ViewCore
from .data_file import *
from .detector import *
from .object import *
from .probe import *
from .ptychopy import PtychoPyViewControllerFactory
from .reconstructor import *
from .scan import *
from .settings import *
from .tike import TikeViewControllerFactory


class ControllerCore:
    def __init__(self, model: ModelCore, view: ViewCore) -> None:
        self.model = model
        self.view = view

        self._fileDialogFactory = FileDialogFactory()

        self._ptychopyViewControllerFactory = PtychoPyViewControllerFactory(model.ptychopyBackend)
        self._tikeViewControllerFactory = TikeViewControllerFactory(model.tikeBackend)

        self._importSettingsController = ImportSettingsController.createInstance(
            model.velociprobePresenter, view.importSettingsDialog)
        self._settingsController = SettingsController.createInstance(model.settingsRegistry,
                                                                     model.settingsPresenter,
                                                                     view.settingsGroupView,
                                                                     view.settingsEntryView,
                                                                     self._fileDialogFactory)
        self._detectorController = DetectorController.createInstance(
            model.detectorPresenter, view.detectorParametersView.detectorView)
        self._datasetController = DatasetController.createInstance(
            model.velociprobePresenter, model.detectorImagePresenter,
            view.detectorParametersView.datasetView)
        self._cropController = CropController.createInstance(
            model.cropPresenter, view.detectorParametersView.imageCropView)
        self._detectorImageController = DetectorImageController.createInstance(
            model.detectorImagePresenter, view.detectorImageView, self._fileDialogFactory)
        self._probeParametersController = ProbeParametersController.createInstance(
            model.probePresenter, view.probeParametersView, self._fileDialogFactory)
        self._probeImageController = ProbeImageController.createInstance(
            model.probePresenter, view.probeImageView, self._fileDialogFactory)
        self._scanParametersController = ScanParametersController.createInstance(
            model.scanPresenter, view.scanParametersView.scanView, self._fileDialogFactory)
        self._scanPlotController = ScanPlotController.createInstance(model.scanPresenter,
                                                                     view.scanPlotView)
        self._objectParametersController = ObjectParametersController.createInstance(
            model.objectPresenter, view.objectParametersView, self._fileDialogFactory)
        self._objectImageController = ObjectImageController.createInstance(
            model.objectPresenter, view.objectImageView, self._fileDialogFactory)
        self._dataFileController = DataFileController.createInstance(model.dataFilePresenter,
                                                                     model.h5FileTreeReader,
                                                                     view.dataFileTreeView,
                                                                     view.dataFileTableView,
                                                                     self._fileDialogFactory)
        self._reconstructorParametersController = ReconstructorParametersController.createInstance(
            model.reconstructorPresenter, view.reconstructorParametersView,
            [self._ptychopyViewControllerFactory, self._tikeViewControllerFactory])
        self._reconstructorPlotController = ReconstructorPlotController.createInstance(
            model.reconstructorPresenter, view.reconstructorPlotView)
        self._monitorProbeController = ProbeImageController.createInstance(
            model.probePresenter, view.monitorProbeView.imageView, self._fileDialogFactory)
        self._monitorObjectController = ObjectImageController.createInstance(
            model.objectPresenter, view.monitorObjectView.imageView, self._fileDialogFactory)

    @classmethod
    def createInstance(cls, model: ModelCore, view: ViewCore):
        controller = cls(model, view)

        view.navigationActionGroup.triggered.connect(
            lambda action: controller.swapCentralWidgets(action))
        view.openSettingsAction.triggered.connect(
            lambda checked: controller._settingsController.openSettings())
        view.saveSettingsAction.triggered.connect(
            lambda checked: controller._settingsController.saveSettings())
        view.openDataFileAction.triggered.connect(
            lambda checked: controller._dataFileController.openDataFile())
        view.openScanAction.triggered.connect(
            lambda checked: controller._scanParametersController.openScan())
        view.saveScanAction.triggered.connect(
            lambda checked: controller._scanParametersController.saveScan())
        view.openProbeAction.triggered.connect(
            lambda checked: controller._probeParametersController.openProbe())
        view.saveProbeAction.triggered.connect(
            lambda checked: controller._probeParametersController.saveProbe())
        view.openObjectAction.triggered.connect(
            lambda checked: controller._objectParametersController.openObject())
        view.saveObjectAction.triggered.connect(
            lambda checked: controller._objectParametersController.saveObject())
        #view.exitAction.triggered.connect(
        #        lambda checked: QApplication.quit())

        return controller

    def swapCentralWidgets(self, action: QAction) -> None:
        index = action.data()
        self.view.parametersWidget.setCurrentIndex(index)
        self.view.contentsWidget.setCurrentIndex(index)
