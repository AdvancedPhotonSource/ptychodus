from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import logging

from PyQt5.QtCore import Qt, QAbstractItemModel, QTimer
from PyQt5.QtWidgets import QLabel, QWidget

from ptychodus.api.observer import Observable, Observer

from ..model.product import (
    ProductRepository,
    ProductRepositoryItem,
    ProductRepositoryObserver,
)
from ..model.product.metadata import MetadataRepositoryItem
from ..model.product.object import ObjectRepositoryItem
from ..model.product.probe import ProbeRepositoryItem
from ..model.product.scan import ScanRepositoryItem
from ..model.reconstructor import ReconstructorPresenter
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


class ReconstructorController(ProductRepositoryObserver, Observer):
    def __init__(
        self,
        presenter: ReconstructorPresenter,
        productRepository: ProductRepository,
        view: ReconstructorParametersView,
        plotView: ReconstructorPlotView,
        fileDialogFactory: FileDialogFactory,
        viewControllerFactoryList: Iterable[ReconstructorViewControllerFactory],
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._productRepository = productRepository
        self._view = view
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._viewControllerFactoryDict: dict[str, ReconstructorViewControllerFactory] = {
            vcf.backendName: vcf for vcf in viewControllerFactoryList
        }
        self._timer = QTimer()
        self._timer.timeout.connect(self._processResults)
        self._timer.start(5 * 1000)  # TODO customize (in milliseconds)

    @classmethod
    def createInstance(
        cls,
        presenter: ReconstructorPresenter,
        productRepository: ProductRepository,
        view: ReconstructorParametersView,
        plotView: ReconstructorPlotView,
        fileDialogFactory: FileDialogFactory,
        productTableModel: QAbstractItemModel,
        viewControllerFactoryList: list[ReconstructorViewControllerFactory],
    ) -> ReconstructorController:
        controller = cls(
            presenter,
            productRepository,
            view,
            plotView,
            fileDialogFactory,
            viewControllerFactoryList,
        )
        presenter.addObserver(controller)
        productRepository.addObserver(controller)

        for name in presenter.getReconstructorList():
            controller._addReconstructor(name)

        view.reconstructorView.algorithmComboBox.textActivated.connect(presenter.setReconstructor)
        view.reconstructorView.algorithmComboBox.currentIndexChanged.connect(
            view.stackedWidget.setCurrentIndex
        )

        view.reconstructorView.productComboBox.textActivated.connect(controller._redrawPlot)
        view.reconstructorView.productComboBox.setModel(productTableModel)

        openModelAction = view.reconstructorView.modelMenu.addAction('Open...')
        openModelAction.triggered.connect(controller._openModel)
        saveModelAction = view.reconstructorView.modelMenu.addAction('Save...')
        saveModelAction.triggered.connect(controller._saveModel)

        openTrainingDataAction = view.reconstructorView.trainerMenu.addAction(
            'Open Training Data...'
        )
        openTrainingDataAction.triggered.connect(controller._openTrainingData)
        saveTrainingDataAction = view.reconstructorView.trainerMenu.addAction(
            'Save Training Data...'
        )
        saveTrainingDataAction.triggered.connect(controller._saveTrainingData)
        ingestTrainingDataAction = view.reconstructorView.trainerMenu.addAction(
            'Ingest Training Data'
        )
        ingestTrainingDataAction.triggered.connect(controller._ingestTrainingData)
        clearTrainingDataAction = view.reconstructorView.trainerMenu.addAction(
            'Clear Training Data'
        )
        clearTrainingDataAction.triggered.connect(controller._clearTrainingData)
        view.reconstructorView.trainerMenu.addSeparator()
        trainAction = view.reconstructorView.trainerMenu.addAction('Train')
        trainAction.triggered.connect(controller._train)

        reconstructSplitAction = view.reconstructorView.reconstructorMenu.addAction(
            'Reconstruct Odd/Even Split'
        )
        reconstructSplitAction.triggered.connect(controller._reconstructSplit)
        reconstructAction = view.reconstructorView.reconstructorMenu.addAction('Reconstruct')
        reconstructAction.triggered.connect(controller._reconstruct)

        controller._syncAlgorithmToView()

        return controller

    def _processResults(self) -> None:
        self._presenter.processResults(block=False)

    def _addReconstructor(self, name: str) -> None:
        backendName, reconstructorName = name.split('/')  # TODO REDO
        self._view.reconstructorView.algorithmComboBox.addItem(
            name, self._view.reconstructorView.algorithmComboBox.count()
        )

        if backendName in self._viewControllerFactoryDict:
            viewControllerFactory = self._viewControllerFactoryDict[backendName]
            widget = viewControllerFactory.createViewController(reconstructorName)
        else:
            widget = QLabel(f'{backendName} not found!')
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._view.stackedWidget.addWidget(widget)

    def _reconstruct(self) -> None:
        inputProductIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            return

        try:
            self._presenter.reconstruct(inputProductIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reconstructor', err)

    def _reconstructSplit(self) -> None:
        inputProductIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            return

        try:
            self._presenter.reconstructSplit(inputProductIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Split Reconstructor', err)

    def _openModel(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Model',
            nameFilters=self._presenter.getOpenModelFileFilterList(),
            selectedNameFilter=self._presenter.getOpenModelFileFilter(),
        )

        if filePath:
            try:
                self._presenter.openModel(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Model Reader', err)

    def _saveModel(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Model',
            nameFilters=self._presenter.getSaveModelFileFilterList(),
            selectedNameFilter=self._presenter.getSaveModelFileFilter(),
        )

        if filePath:
            try:
                self._presenter.saveModel(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Model Writer', err)

    def _openTrainingData(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Training Data',
            nameFilters=self._presenter.getOpenTrainingDataFileFilterList(),
            selectedNameFilter=self._presenter.getOpenTrainingDataFileFilter(),
        )

        if filePath:
            try:
                self._presenter.openTrainingData(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Training Data Reader', err)

    def _saveTrainingData(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Training Data',
            nameFilters=self._presenter.getSaveTrainingDataFileFilterList(),
            selectedNameFilter=self._presenter.getSaveTrainingDataFileFilter(),
        )

        if filePath:
            try:
                self._presenter.saveTrainingData(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Training Data Writer', err)

    def _ingestTrainingData(self) -> None:
        inputProductIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            return

        try:
            self._presenter.ingestTrainingData(inputProductIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Ingester', err)

    def _clearTrainingData(self) -> None:
        try:
            self._presenter.clearTrainingData()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Clear', err)

    def _train(self) -> None:
        try:
            self._presenter.train()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Trainer', err)

    def _redrawPlot(self) -> None:
        productIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if productIndex < 0:
            self._plotView.axes.clear()
            return

        try:
            item = self._productRepository[productIndex]
        except IndexError as err:
            logger.exception(err)
            return

        ax = self._plotView.axes
        ax.clear()
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.grid(True)
        ax.plot(item.getCosts(), '.-', label='Cost', linewidth=1.5)
        self._plotView.figureCanvas.draw()

    def _syncAlgorithmToView(self) -> None:
        self._view.reconstructorView.algorithmComboBox.setCurrentText(
            self._presenter.getReconstructor()
        )

        isTrainable = self._presenter.isTrainable
        self._view.reconstructorView.modelButton.setVisible(isTrainable)
        self._view.reconstructorView.trainerButton.setVisible(isTrainable)

        self._redrawPlot()

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        pass

    def handleScanChanged(self, index: int, item: ScanRepositoryItem) -> None:
        pass

    def handleProbeChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        pass

    def handleObjectChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        pass

    def handleCostsChanged(self, index: int, costs: Sequence[float]) -> None:
        currentIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if index == currentIndex:
            self._redrawPlot()

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncAlgorithmToView()
