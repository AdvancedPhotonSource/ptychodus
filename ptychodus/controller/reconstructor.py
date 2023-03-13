from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Optional
import logging
import traceback

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QMessageBox, QWidget

from ..api.observer import Observable, Observer
from ..api.reconstructor import ReconstructResult
from ..model.object import ObjectPresenter
from ..model.probe import ProbePresenter
from ..model.reconstructor import ReconstructorPlotPresenter, ReconstructorPresenter
from ..model.scan import ScanPresenter, ScanRepositoryItem
from ..view import ReconstructorParametersView, ReconstructorPlotView, resources

logger = logging.getLogger(__name__)


class ReconstructorViewControllerFactory(ABC):

    @abstractproperty
    def backendName(self) -> str:
        pass

    @abstractmethod
    def createViewController(self, reconstructorName: str) -> QWidget:
        pass


@dataclass(frozen=True)
class ScanRepositoryKeyAndValue:
    name: str
    item: ScanRepositoryItem


class ScanListModel(QAbstractListModel):

    def __init__(self, presenter: ScanPresenter, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter
        self._scanList: list[ScanRepositoryKeyAndValue] = list()

    def refresh(self) -> None:
        self.beginResetModel()
        self._scanList = [
            ScanRepositoryKeyAndValue(name, item)
            for name, item in self._presenter.getScanRepositoryContents()
            if self._presenter.canActivateScan(name)
        ]
        self.endResetModel()

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid() and role == Qt.DisplayRole:
            entry = self._scanList[index.row()]
            value = QVariant(entry.name)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._scanList)


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
        self._scanListModel = ScanListModel(scanPresenter)

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

        view.reconstructorView.scanComboBox.currentTextChanged.connect(scanPresenter.setActiveScan)
        view.reconstructorView.scanComboBox.setModel(controller._scanListModel)

        view.reconstructorView.probeComboBox.addItem('Current Probe')  # FIXME
        view.reconstructorView.objectComboBox.addItem('Current Object')  # FIXME
        view.reconstructorView.reconstructButton.clicked.connect(controller._reconstruct)

        controller._refreshScanListModel()
        controller._syncModelToView()

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
            retval = msgBox.exec_()
        else:
            self._plotPresenter.setEnumeratedYValues(result.objective)

        print(result.result)  # TODO

    def _refreshScanListModel(self) -> None:
        self._view.reconstructorView.scanComboBox.blockSignals(True)
        self._scanListModel.refresh()
        self._view.reconstructorView.scanComboBox.setCurrentText(
            self._scanPresenter.getActiveScan())
        self._view.reconstructorView.scanComboBox.blockSignals(False)

    def _getValidationPixmap(self, isValid: bool) -> QPixmap:
        pixmap = QPixmap(':/icons/check' if isValid else ':/icons/xmark')
        return pixmap.scaledToHeight(24)

    def _syncModelToView(self) -> None:
        self._view.reconstructorView.algorithmComboBox.setCurrentText(
            self._presenter.getReconstructor())

        isScanValid = self._scanPresenter.isActiveScanValid()
        isProbeValid = self._probePresenter.isActiveProbeValid()
        isObjectValid = self._objectPresenter.isActiveObjectValid()

        self._view.reconstructorView.scanValidationLabel.setPixmap(
            self._getValidationPixmap(isScanValid))
        self._view.reconstructorView.probeValidationLabel.setPixmap(
            self._getValidationPixmap(isProbeValid))
        self._view.reconstructorView.objectValidationLabel.setPixmap(
            self._getValidationPixmap(isObjectValid))

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
        elif observable is self._scanPresenter:
            self._refreshScanListModel()
            self._syncModelToView()
        elif observable is self._probePresenter:
            self._syncModelToView()
        elif observable is self._objectPresenter:
            self._syncModelToView()


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
