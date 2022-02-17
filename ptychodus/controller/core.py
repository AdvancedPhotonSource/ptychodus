from pathlib import Path

from PyQt5.QtWidgets import QApplication, QAction

from ..model import ModelCore
from ..view import ViewCore

from .data_file import *
from .detector import *
from .object import *
from .probe import *
from .reconstructor import *
from .scan import *
from .settings import *
from .tike import TikeViewControllerFactory


class ControllerCore:
    def __init__(self, model: ModelCore, view: ViewCore) -> None:
        self.model = model
        self.view = view

        self._tikeViewControllerFactory = TikeViewControllerFactory(model.tikeBackend)

        self._importSettingsController = ImportSettingsController.createInstance(
                model.importSettingsPresenter, view.importSettingsDialog)
        self._settingsController = SettingsController.createInstance(
                model.settingsRegistry, model.settingsPresenter,
                view.settingsGroupView, view.settingsEntryView)
        self._detectorParametersController = DetectorParametersController.createInstance(
                model.detectorParametersPresenter, view.detectorParametersView.detectorView)
        self._detectorDatasetController = DetectorDatasetController.createInstance(
                model.detectorDatasetPresenter, model.detectorImagePresenter,
                view.detectorParametersView.datasetView)
        self._detectorImageCropController = DetectorImageCropController.createInstance(
                model.detectorParametersPresenter, view.detectorParametersView.imageCropView)
        self._detectorImageController = DetectorImageController.createInstance(
                model.detectorImagePresenter, view.detectorImageView)
        self._probeParametersController = ProbeParametersController.createInstance(
                model.probePresenter, view.probeParametersView)
        self._probeImageController = ProbeImageController.createInstance(
                model.probePresenter, view.probeImageView)
        self._scanParametersController = ScanParametersController.createInstance(
                model.scanPresenter, view.scanParametersView)
        self._scanPlotController = ScanPlotController.createInstance(
                model.scanPresenter, view.scanPlotView)
        self._objectParametersController = ObjectParametersController.createInstance(
                model.objectPresenter, view.objectParametersView)
        self._objectImageController = ObjectImageController.createInstance(
                model.objectPresenter, view.objectImageView)
        self._dataFileController = DataFileController.createInstance(
                model.dataFilePresenter, model.h5FileTreeReader,
                view.dataFileTreeView, view.dataFileTableView)
        self._reconstructorParametersController = ReconstructorParametersController.createInstance(
                model.reconstructorPresenter, view.reconstructorParametersView,
                [ self._tikeViewControllerFactory ])
        self._reconstructorPlotController = ReconstructorPlotController.createInstance(
                model.reconstructorPresenter, view.reconstructorPlotView)
        self._monitorProbeController = ProbeImageController.createInstance(
                model.probePresenter, view.monitorProbeView.imageView)
        self._monitorObjectController = ObjectImageController.createInstance(
                model.objectPresenter, view.monitorObjectView.imageView)

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

