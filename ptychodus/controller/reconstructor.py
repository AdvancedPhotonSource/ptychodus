from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable
import logging
import itertools

from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QWidget

from ..api.observer import Observable, Observer
from ..model.object import ObjectPresenter
from ..model.probe import ProbePresenter
from ..model.reconstructor import ReconstructorPresenter
from ..model.scan import ScanPresenter
from ..view.reconstructor import ReconstructorParametersView, ReconstructorPlotView
from ..view.widgets import ExceptionDialog
from .data import FileDialogFactory

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

    def __init__(self, presenter: ReconstructorPresenter, scanPresenter: ScanPresenter,
                 probePresenter: ProbePresenter, objectPresenter: ObjectPresenter,
                 view: ReconstructorParametersView, plotView: ReconstructorPlotView,
                 fileDialogFactory: FileDialogFactory,
                 viewControllerFactoryList: Iterable[ReconstructorViewControllerFactory]) -> None:
        super().__init__()
        self._presenter = presenter
        self._scanPresenter = scanPresenter
        self._probePresenter = probePresenter
        self._objectPresenter = objectPresenter
        self._view = view
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._viewControllerFactoryDict: dict[str, ReconstructorViewControllerFactory] = \
                { vcf.backendName: vcf for vcf in viewControllerFactoryList }
        self._scanListModel = QStringListModel()
        self._probeListModel = QStringListModel()
        self._objectListModel = QStringListModel()

    @classmethod
    def createInstance(
        cls, presenter: ReconstructorPresenter, scanPresenter: ScanPresenter,
        probePresenter: ProbePresenter, objectPresenter: ObjectPresenter,
        view: ReconstructorParametersView, plotView: ReconstructorPlotView,
        fileDialogFactory: FileDialogFactory,
        viewControllerFactoryList: list[ReconstructorViewControllerFactory]
    ) -> ReconstructorParametersController:
        controller = cls(presenter, scanPresenter, probePresenter, objectPresenter, view, plotView,
                         fileDialogFactory, viewControllerFactoryList)
        presenter.addObserver(controller)
        scanPresenter.addObserver(controller)
        probePresenter.addObserver(controller)
        objectPresenter.addObserver(controller)

        for name in presenter.getReconstructorList():
            controller._addReconstructor(name)

        view.reconstructorView.algorithmComboBox.textActivated.connect(presenter.setReconstructor)
        view.reconstructorView.algorithmComboBox.currentIndexChanged.connect(
            view.stackedWidget.setCurrentIndex)

        view.reconstructorView.scanComboBox.textActivated.connect(scanPresenter.selectScan)
        view.reconstructorView.scanComboBox.setModel(controller._scanListModel)

        view.reconstructorView.probeComboBox.textActivated.connect(probePresenter.selectProbe)
        view.reconstructorView.probeComboBox.setModel(controller._probeListModel)

        view.reconstructorView.objectComboBox.textActivated.connect(objectPresenter.selectObject)
        view.reconstructorView.objectComboBox.setModel(controller._objectListModel)

        view.reconstructorView.reconstructButton.clicked.connect(controller._reconstruct)
        view.reconstructorView.reconstructSplitButton.clicked.connect(controller._reconstructSplit)
        view.reconstructorView.ingestButton.clicked.connect(controller._ingestTrainingData)
        view.reconstructorView.saveButton.clicked.connect(controller._saveTrainingData)
        view.reconstructorView.trainButton.clicked.connect(controller._train)
        view.reconstructorView.clearButton.clicked.connect(controller._clearTrainingData)

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
        try:
            self._presenter.reconstruct()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reconstructor', err)

    def _reconstructSplit(self) -> None:
        try:
            self._presenter.reconstructSplit()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Split Reconstructor', err)

    def _ingestTrainingData(self) -> None:
        try:
            self._presenter.ingestTrainingData()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Ingester', err)

    def _saveTrainingData(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Training Data',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            try:
                self._presenter.saveTrainingData(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File writer', err)

    def _train(self) -> None:
        try:
            self._presenter.train()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Trainer', err)

    def _clearTrainingData(self) -> None:
        try:
            self._presenter.clearTrainingData()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Clear', err)

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
        ax.set_xlabel(axisX.label)
        ax.set_ylabel(axisY.label)
        ax.grid(True)

        if (
            (len(axisX.series) == len(axisY.series)) or
            (len(axisX.series) == 1 and len(axisY.series) >= 1)
        ):
            for sx, sy in zip(itertools.cycle(axisX.series), axisY.series):
                ax.plot(sx.values, sy.values, '.-', label=sy.label, linewidth=1.5)
                if hasattr(sy, 'hi') and hasattr(sy, 'lo'):
                    ax.fill_between(sx.values, sy.lo, sy.hi, alpha=0.2)
        else:
            logger.error('Failed to broadcast plot series!')

        if len(axisX.series) > 0:
            ax.legend(loc='upper right')

        self._plotView.figureCanvas.draw()

    def _syncModelToView(self) -> None:
        self._view.reconstructorView.algorithmComboBox.setCurrentText(
            self._presenter.getReconstructor())

        isTrainable = self._presenter.isTrainable
        self._view.reconstructorView.ingestButton.setVisible(isTrainable)
        self._view.reconstructorView.saveButton.setVisible(isTrainable)
        self._view.reconstructorView.trainButton.setVisible(isTrainable)
        self._view.reconstructorView.clearButton.setVisible(isTrainable)

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
