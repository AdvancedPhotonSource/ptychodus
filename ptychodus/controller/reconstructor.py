from __future__ import annotations
from abc import ABC, abstractmethod
import logging
import traceback

from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox, QWidget

from ..api.observer import Observable, Observer
from ..api.reconstructor import ReconstructResult
from ..model.object import ObjectPresenter
from ..model.probe import ProbePresenter
from ..model.reconstructor import ReconstructorPlotPresenter, ReconstructorPresenter
from ..model.scan import ScanPresenter
from ..view import ReconstructorParametersView, ReconstructorPlotView

logger = logging.getLogger(__name__)


class ReconstructorViewControllerFactory(ABC):

    @property
    @abstractmethod
    def backendName(self) -> str:
        pass

    @abstractmethod
    def createViewController(self, reconstructorName: str) -> QWidget:
        pass


class ReconstructorParametersController(Observer):

    def __init__(
        self,
        presenter: ReconstructorPresenter,
        plotPresenter: ReconstructorPlotPresenter,
        scanPresenter: ScanPresenter,
        probePresenter: ProbePresenter,
        objectPresenter: ObjectPresenter,
        view: ReconstructorParametersView,
        viewControllerFactoryList: list[ReconstructorViewControllerFactory],
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._plotPresenter = plotPresenter
        self._scanPresenter = scanPresenter
        self._probePresenter = probePresenter
        self._objectPresenter = objectPresenter
        self._view = view
        self._viewControllerFactoryDict: dict[str, ReconstructorViewControllerFactory] = \
                { vcf.backendName: vcf for vcf in viewControllerFactoryList }
        self._scanListModel = QStringListModel()
        self._probeListModel = QStringListModel()
        self._objectListModel = QStringListModel()

    @classmethod
    def createInstance(
        cls,
        presenter: ReconstructorPresenter,
        plotPresenter: ReconstructorPlotPresenter,
        scanPresenter: ScanPresenter,
        probePresenter: ProbePresenter,
        objectPresenter: ObjectPresenter,
        view: ReconstructorParametersView,
        viewControllerFactoryList: list[ReconstructorViewControllerFactory],
    ) -> ReconstructorParametersController:
        controller = cls(presenter, plotPresenter, scanPresenter, probePresenter, objectPresenter,
                         view, viewControllerFactoryList)
        presenter.addObserver(controller)
        scanPresenter.addObserver(controller)
        probePresenter.addObserver(controller)
        objectPresenter.addObserver(controller)

        for name in presenter.getReconstructorList():
            controller._addReconstructor(name)

        view.reconstructorView.algorithmComboBox.currentTextChanged.connect(
            presenter.setReconstructor)
        view.reconstructorView.algorithmComboBox.currentIndexChanged.connect(
            view.stackedWidget.setCurrentIndex)

        view.reconstructorView.scanComboBox.currentTextChanged.connect(scanPresenter.selectScan)
        view.reconstructorView.scanComboBox.setModel(controller._scanListModel)

        view.reconstructorView.probeComboBox.currentTextChanged.connect(probePresenter.selectProbe)
        view.reconstructorView.probeComboBox.setModel(controller._probeListModel)

        view.reconstructorView.objectComboBox.currentTextChanged.connect(
            objectPresenter.selectObject)
        view.reconstructorView.objectComboBox.setModel(controller._objectListModel)

        view.reconstructorView.reconstructButton.clicked.connect(controller._reconstruct)

        controller._syncAlgorithmToView()
        controller._syncScanToView()
        controller._syncProbeToView()
        controller._syncObjectToView()

        return controller

    def _addReconstructor(self, name: str) -> None:
        backendName, reconstructorName = name.split('/')  # TODO REDO
        self._view.reconstructorView.algorithmComboBox.addItem(
            name, self._view.reconstructorView.algorithmComboBox.count())

        if backendName in self._viewControllerFactoryDict:
            viewControllerFactory = self._viewControllerFactoryDict[backendName]
            widget = viewControllerFactory.createViewController(reconstructorName)
        else:
            widget = QLabel(f'{backendName} not found!')
            widget.setAlignment(Qt.AlignCenter)

        self._view.stackedWidget.addWidget(widget)

    def _reconstruct(self) -> None:
        result = ReconstructResult(-1, [[]])

        try:
            result = self._presenter.reconstruct()
        except Exception as err:
            logger.exception(err)

            msgBox = QMessageBox()
            msgBox.setWindowTitle('Exception Dialog')
            msgBox.setIcon(QMessageBox.Critical)
            msgBox.setText(f'The reconstructor raised a {err.__class__.__name__}!')
            msgBox.setInformativeText(str(err))
            msgBox.setDetailedText(traceback.format_exc())
            _ = msgBox.exec_()
        else:
            self._plotPresenter.setEnumeratedYValues(result.objective)

        print(result.result)  # TODO

    def _syncScanToView(self) -> None:
        self._view.reconstructorView.scanComboBox.blockSignals(True)
        self._scanListModel.setStringList(self._scanPresenter.getSelectableNames())
        self._view.reconstructorView.scanComboBox.setCurrentText(
            self._scanPresenter.getSelectedScan())
        self._view.reconstructorView.scanComboBox.blockSignals(False)

        isValid = self._scanPresenter.isSelectedScanValid()
        validationPixmap = self._getValidationPixmap(isValid)
        self._view.reconstructorView.scanValidationLabel.setPixmap(validationPixmap)

    def _syncProbeToView(self) -> None:
        self._view.reconstructorView.probeComboBox.blockSignals(True)
        self._probeListModel.setStringList(['Current Probe'])  # TODO
        self._view.reconstructorView.probeComboBox.blockSignals(False)

        isValid = self._probePresenter.isSelectedProbeValid()
        validationPixmap = self._getValidationPixmap(isValid)
        self._view.reconstructorView.probeValidationLabel.setPixmap(validationPixmap)

    def _syncObjectToView(self) -> None:
        self._view.reconstructorView.objectComboBox.blockSignals(True)
        self._objectListModel.setStringList(self._objectPresenter.getSelectableNames())
        self._view.reconstructorView.objectComboBox.setCurrentText(
            self._objectPresenter.getSelectedObject())
        self._view.reconstructorView.objectComboBox.blockSignals(False)

        isValid = self._objectPresenter.isSelectedObjectValid()
        validationPixmap = self._getValidationPixmap(isValid)
        self._view.reconstructorView.objectValidationLabel.setPixmap(validationPixmap)

    def _getValidationPixmap(self, isValid: bool) -> QPixmap:
        pixmap = QPixmap(':/icons/check' if isValid else ':/icons/xmark')
        return pixmap.scaledToHeight(24)

    def _syncAlgorithmToView(self) -> None:
        self._view.reconstructorView.algorithmComboBox.setCurrentText(
            self._presenter.getReconstructor())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncAlgorithmToView()
        elif observable is self._scanPresenter:
            self._syncScanToView()
        elif observable is self._probePresenter:
            self._syncProbeToView()
        elif observable is self._objectPresenter:
            self._syncObjectToView()


class ReconstructorPlotController(Observer):

    def __init__(self, presenter: ReconstructorPlotPresenter, view: ReconstructorPlotView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ReconstructorPlotPresenter,
                       view: ReconstructorPlotView) -> ReconstructorPlotController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        x = self._presenter.xvalues
        y = self._presenter.yvalues

        self._view.axes.clear()
        self._view.axes.semilogy(x, y, '.-', linewidth=1.5)
        self._view.axes.grid(True)
        self._view.axes.set_xlabel(self._presenter.xlabel)
        self._view.axes.set_ylabel(self._presenter.ylabel)
        self._view.figureCanvas.draw()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
