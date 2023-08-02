from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
import logging

from PyQt5.QtCore import QStringListModel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QWidget

from ..api.observer import Observable, Observer
from ..api.reconstructor import ReconstructOutput
from ..model.object import ObjectPresenter
from ..model.probe import ProbePresenter
from ..model.reconstructor import ReconstructorPlotPresenter, ReconstructorPresenter
from ..model.scan import ScanPresenter
from ..view.reconstructor import ReconstructorParametersView, ReconstructorPlotView
from ..view.widgets import ExceptionDialog

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
        viewControllerFactoryList: Iterable[ReconstructorViewControllerFactory],
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
        view.reconstructorView.ingestButton.clicked.connect(controller._ingest)
        view.reconstructorView.trainButton.clicked.connect(controller._train)
        view.reconstructorView.resetButton.clicked.connect(controller._reset)

        controller._syncModelToView()
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
        result = ReconstructOutput.createNull()

        try:
            result = self._presenter.reconstruct()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reconstructor', err)
        else:
            self._plotPresenter.setEnumeratedYValues(result.objective)

        logger.info(result.result)  # TODO

    def _ingest(self) -> None:
        try:
            self._presenter.ingest()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Ingester', err)

        logger.info('Ingestion complete.')

    def _train(self) -> None:
        try:
            self._presenter.train()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Trainer', err)

        logger.info('Training complete.')

    def _reset(self) -> None:
        try:
            self._presenter.reset()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reset', err)

        logger.info('Reset complete.')

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
        self._probeListModel.setStringList(self._probePresenter.getSelectableNames())
        self._view.reconstructorView.probeComboBox.setCurrentText(
            self._probePresenter.getSelectedProbe())
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

    def _syncModelToView(self) -> None:
        self._view.reconstructorView.algorithmComboBox.setCurrentText(
            self._presenter.getReconstructor())

        isTrainable = self._presenter.isTrainable
        self._view.reconstructorView.ingestButton.setVisible(isTrainable)
        self._view.reconstructorView.trainButton.setVisible(isTrainable)
        self._view.reconstructorView.resetButton.setVisible(isTrainable)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
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
