from __future__ import annotations
from abc import ABC, abstractmethod, abstractproperty

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from PyQt5.QtWidgets import QWidget, QLabel, QFileDialog

from ..model import Observable, Observer, Reconstructor, ReconstructorPresenter
from ..view import ReconstructorParametersView, ReconstructorPlotView


class ReconstructorViewControllerFactory(ABC):
    @abstractproperty
    def backendName(self):
        pass

    @abstractmethod
    def createViewController(self, reconstructorName: str) -> QWidget:
        pass


class ReconstructorParametersController(Observer):
    def __init__(self, presenter: ReconstructorPresenter, view: ReconstructorParametersView,
                 viewControllerFactoryList: list[ReconstructorViewControllerFactory]) -> None:
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

        view.algorithmComboBox.currentTextChanged.connect(presenter.setCurrentAlgorithm)
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
            widget.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)

        self._view.reconstructorStackedWidget.addWidget(widget)

    def _reconstruct(self) -> None:
        result = self._presenter.reconstruct()
        print(result)  # TODO

    def _syncModelToView(self) -> None:
        self._view.algorithmComboBox.setCurrentText(self._presenter.getCurrentAlgorithm())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ReconstructorPlotController(Observer):
    def __init__(self, presenter: ReconstructorPresenter, view: ReconstructorPlotView) -> None:
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ReconstructorPresenter, view: ReconstructorPlotView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        pass  # TODO

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
