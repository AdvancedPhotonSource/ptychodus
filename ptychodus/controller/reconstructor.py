from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty
import logging
import traceback

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QLabel, QMessageBox, QWidget

from ..model import (Observable, Observer, ReconstructorPlotPresenter, ReconstructorPresenter)
from ..view import ReconstructorParametersView, ReconstructorPlotView

logger = logging.getLogger(__name__)


class ReconstructorViewControllerFactory(ABC):

    @abstractproperty
    def backendName(self) -> str:
        pass

    @abstractmethod
    def createViewController(self, reconstructorName: str) -> QWidget:
        pass


class ReconstructorParametersController(Observer):

    def __init__(self, presenter: ReconstructorPresenter, view: ReconstructorParametersView,
                 viewControllerFactoryList: list[ReconstructorViewControllerFactory]) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view
        self._algorithmComboBoxModel = QStandardItemModel()
        self._viewControllerFactoryDict: dict[str, ReconstructorViewControllerFactory] = \
                { vcf.backendName: vcf for vcf in viewControllerFactoryList }

    @classmethod
    def createInstance(cls, presenter: ReconstructorPresenter, view: ReconstructorParametersView,
            viewControllerFactoryList: list[ReconstructorViewControllerFactory]) \
            -> ReconstructorParametersController:
        controller = cls(presenter, view, viewControllerFactoryList)
        presenter.addObserver(controller)

        view.algorithmComboBox.setModel(controller._algorithmComboBoxModel)

        for reconstructorName, backendName in presenter.getAlgorithmDict().items():
            controller._addReconstructor(reconstructorName, backendName)

        view.algorithmComboBox.currentTextChanged.connect(presenter.setAlgorithm)
        view.algorithmComboBox.currentIndexChanged.connect(
            view.reconstructorStackedWidget.setCurrentIndex)
        view.reconstructButton.clicked.connect(controller._reconstruct)

        controller._syncModelToView()

        return controller

    def _addReconstructor(self, reconstructorName: str, backendName: str) -> None:
        row = QStandardItem(reconstructorName)
        row.setData(self._algorithmComboBoxModel.rowCount())
        self._algorithmComboBoxModel.appendRow(row)

        if backendName in self._viewControllerFactoryDict:
            viewControllerFactory = self._viewControllerFactoryDict[backendName]
            widget = viewControllerFactory.createViewController(reconstructorName)
        else:
            widget = QLabel(f'{backendName} not found!')
            widget.setAlignment(Qt.AlignCenter)

        self._view.reconstructorStackedWidget.addWidget(widget)

    def _reconstruct(self) -> None:
        result = -1

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

        print(result)  # TODO

    def _syncModelToView(self) -> None:
        self._view.algorithmComboBox.setCurrentText(self._presenter.getAlgorithm())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ReconstructorPlotController(Observer):

    def __init__(self, presenter: ReconstructorPlotPresenter, view: ReconstructorPlotView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ReconstructorPlotPresenter, view: ReconstructorPlotView):
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
