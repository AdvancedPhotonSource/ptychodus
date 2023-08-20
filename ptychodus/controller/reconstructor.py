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
from ..model.reconstructor import ReconstructorPresenter
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
        scanPresenter: ScanPresenter,
        probePresenter: ProbePresenter,
        objectPresenter: ObjectPresenter,
        view: ReconstructorParametersView,
        plotView: ReconstructorPlotView,
        viewControllerFactoryList: Iterable[ReconstructorViewControllerFactory],
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._scanPresenter = scanPresenter
        self._probePresenter = probePresenter
        self._objectPresenter = objectPresenter
        self._view = view
        self._plotView = plotView
        self._viewControllerFactoryDict: dict[str, ReconstructorViewControllerFactory] = \
                { vcf.backendName: vcf for vcf in viewControllerFactoryList }
        self._scanListModel = QStringListModel()
        self._probeListModel = QStringListModel()
        self._objectListModel = QStringListModel()

    @classmethod
    def createInstance(
        cls,
        presenter: ReconstructorPresenter,
        scanPresenter: ScanPresenter,
        probePresenter: ProbePresenter,
        objectPresenter: ObjectPresenter,
        view: ReconstructorParametersView,
        plotView: ReconstructorPlotView,
        viewControllerFactoryList: list[ReconstructorViewControllerFactory],
    ) -> ReconstructorParametersController:
        controller = cls(presenter, scanPresenter, probePresenter, objectPresenter, view, plotView,
                         viewControllerFactoryList)
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

        logger.info(result.result)  # TODO

    def _ingest(self) -> None:
        try:
            self._presenter.ingest()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Ingester', err)

    def _train(self) -> None:
        try:
            self._presenter.train()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Trainer', err)

    def _reset(self) -> None:
        try:
            self._presenter.reset()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reset', err)

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

    def _redrawPlot(self) -> None:
        plot2D = self._presenter.getPlot()
        axisX = plot2D.axisX
        axisY = plot2D.axisY

        ax = self._plotView.axes
        ax.clear()

        if len(axisX.series) == len(axisY.series):
            for sx, sy in zip(axisX.series, axisY.series):
                ax.plot(sx.values, sy.values, '.-', label=sy.label, linewidth=1.5)
        elif len(axisX.series) == 1:
            sx = axisX.series[0]

            for sy in axisY.series:
                ax.plot(sx.values, sy.values, '.-', label=sy.label, linewidth=1.5)
        else:
            logger.error('Failed to broadcast plot series!')

        ax.set_xlabel(axisX.label)
        ax.set_ylabel(axisY.label)
        ax.grid(True)

        if len(axisX.series) > 0:
            ax.legend()

        self._plotView.figureCanvas.draw()

    def _syncModelToView(self) -> None:
        self._view.reconstructorView.algorithmComboBox.setCurrentText(
            self._presenter.getReconstructor())

        isTrainable = self._presenter.isTrainable
        self._view.reconstructorView.ingestButton.setVisible(isTrainable)
        self._view.reconstructorView.trainButton.setVisible(isTrainable)
        self._view.reconstructorView.resetButton.setVisible(isTrainable)

        self._redrawPlot()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
        elif observable is self._scanPresenter:
            self._syncScanToView()
        elif observable is self._probePresenter:
            self._syncProbeToView()
        elif observable is self._objectPresenter:
            self._syncObjectToView()
