from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import logging

from PyQt5.QtCore import Qt, QAbstractItemModel, QTimer
from PyQt5.QtWidgets import QActionGroup, QLabel, QWidget

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
from ..view.reconstructor import ReconstructorView, ReconstructorPlotView
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
        view: ReconstructorView,
        plotView: ReconstructorPlotView,
        productTableModel: QAbstractItemModel,
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

        for name in presenter.getReconstructorList():
            self._addReconstructor(name)

        view.parametersView.algorithmComboBox.textActivated.connect(presenter.setReconstructor)
        view.parametersView.algorithmComboBox.currentIndexChanged.connect(
            view.stackedWidget.setCurrentIndex
        )

        view.parametersView.productComboBox.textActivated.connect(self._redrawPlot)
        view.parametersView.productComboBox.setModel(productTableModel)

        self._progressTimer = QTimer()
        self._progressTimer.timeout.connect(self._updateProgress)
        self._progressTimer.start(5 * 1000)  # TODO customize (in milliseconds)

        view.progressDialog.setModal(True)
        view.progressDialog.setWindowModality(Qt.ApplicationModal)
        view.progressDialog.setWindowFlags(Qt.Window | Qt.WindowTitleHint | Qt.CustomizeWindowHint)
        view.progressDialog.textEdit.setReadOnly(True)

        openModelAction = view.parametersView.reconstructorMenu.addAction('Open Model...')
        openModelAction.triggered.connect(self._openModel)
        saveModelAction = view.parametersView.reconstructorMenu.addAction('Save Model...')
        saveModelAction.triggered.connect(self._saveModel)

        self._modelActionGroup = QActionGroup(view.parametersView.reconstructorMenu)
        self._modelActionGroup.setExclusive(False)
        self._modelActionGroup.addAction(openModelAction)
        self._modelActionGroup.addAction(saveModelAction)
        self._modelActionGroup.addAction(view.parametersView.reconstructorMenu.addSeparator())

        reconstructSplitAction = view.parametersView.reconstructorMenu.addAction(
            'Reconstruct Odd/Even Split'
        )
        reconstructSplitAction.triggered.connect(self._reconstructSplit)
        reconstructAction = view.parametersView.reconstructorMenu.addAction('Reconstruct')
        reconstructAction.triggered.connect(self._reconstruct)

        exportTrainingDataAction = view.parametersView.trainerMenu.addAction(
            'Export Training Data...'
        )
        exportTrainingDataAction.triggered.connect(self._exportTrainingData)
        trainAction = view.parametersView.trainerMenu.addAction('Train')
        trainAction.triggered.connect(self._train)

        presenter.addObserver(self)
        productRepository.addObserver(self)
        self._syncModelToView()

    def _updateProgress(self) -> None:
        isReconstructing = self._presenter.isReconstructing

        for button in self._view.progressDialog.buttonBox.buttons():
            button.setEnabled(not isReconstructing)

        for text in self._presenter.flushLog():
            self._view.progressDialog.textEdit.appendPlainText(text)

        self._presenter.processResults(block=False)

    def _addReconstructor(self, name: str) -> None:
        backendName, reconstructorName = name.split('/')  # TODO REDO
        self._view.parametersView.algorithmComboBox.addItem(
            name, self._view.parametersView.algorithmComboBox.count()
        )

        if backendName in self._viewControllerFactoryDict:
            viewControllerFactory = self._viewControllerFactoryDict[backendName]
            widget = viewControllerFactory.createViewController(reconstructorName)
        else:
            widget = QLabel(f'{backendName} not found!')
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._view.stackedWidget.addWidget(widget)

    def _reconstruct(self) -> None:
        inputProductIndex = self._view.parametersView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            return

        try:
            self._presenter.reconstruct(inputProductIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reconstructor', err)

        self._view.progressDialog.show()

    def _reconstructSplit(self) -> None:
        inputProductIndex = self._view.parametersView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            return

        try:
            self._presenter.reconstructSplit(inputProductIndex)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Split Reconstructor', err)

        self._view.progressDialog.show()

    def _openModel(self) -> None:
        nameFilter = self._presenter.getModelFileFilter()
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view, 'Open Model', nameFilters=[nameFilter], selectedNameFilter=nameFilter
        )

        if filePath:
            try:
                self._presenter.openModel(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Model Reader', err)

    def _saveModel(self) -> None:
        nameFilter = self._presenter.getModelFileFilter()
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Model', nameFilters=[nameFilter], selectedNameFilter=nameFilter
        )

        if filePath:
            try:
                self._presenter.saveModel(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Model Writer', err)

    def _exportTrainingData(self) -> None:
        inputProductIndex = self._view.parametersView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            return

        nameFilter = self._presenter.getTrainingDataFileFilter()
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Export Training Data',
            nameFilters=[nameFilter],
            selectedNameFilter=nameFilter,
        )

        if filePath:
            try:
                self._presenter.exportTrainingData(filePath, inputProductIndex)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Training Data Writer', err)

    def _train(self) -> None:
        dataPath = self._fileDialogFactory.getExistingDirectoryPath(
            self._view,
            'Choose Training Data Directory',
            initialDirectory=self._presenter.getTrainingDataPath(),
        )

        if dataPath:
            try:
                self._presenter.train(dataPath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('Trainer', err)

    def _redrawPlot(self) -> None:
        productIndex = self._view.parametersView.productComboBox.currentIndex()

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

    def _syncModelToView(self) -> None:
        self._view.parametersView.algorithmComboBox.setCurrentText(
            self._presenter.getReconstructor()
        )

        isTrainable = self._presenter.isTrainable
        self._modelActionGroup.setVisible(isTrainable)
        self._view.parametersView.trainerButton.setVisible(isTrainable)

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
        currentIndex = self._view.parametersView.productComboBox.currentIndex()

        if index == currentIndex:
            self._redrawPlot()

    def handleItemRemoved(self, index: int, item: ProductRepositoryItem) -> None:
        pass

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
