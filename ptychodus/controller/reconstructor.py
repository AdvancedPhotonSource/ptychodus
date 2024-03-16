from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
import logging

from PyQt5.QtCore import Qt, QStringListModel
from PyQt5.QtWidgets import QLabel, QWidget

from ..api.observer import Observable, Observer
from ..model.product import ProductRepository, ProductRepositoryItem, ProductRepositoryObserver
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

    def __init__(self, presenter: ReconstructorPresenter, productRepository: ProductRepository,
                 view: ReconstructorParametersView, plotView: ReconstructorPlotView,
                 fileDialogFactory: FileDialogFactory,
                 viewControllerFactoryList: Iterable[ReconstructorViewControllerFactory]) -> None:
        super().__init__()
        self._presenter = presenter
        self._productRepository = productRepository
        self._view = view
        self._plotView = plotView
        self._fileDialogFactory = fileDialogFactory
        self._viewControllerFactoryDict: dict[str, ReconstructorViewControllerFactory] = \
                { vcf.backendName: vcf for vcf in viewControllerFactoryList }
        self._productListModel = QStringListModel()

    @classmethod
    def createInstance(
        cls, presenter: ReconstructorPresenter, productRepository: ProductRepository,
        view: ReconstructorParametersView, plotView: ReconstructorPlotView,
        fileDialogFactory: FileDialogFactory,
        viewControllerFactoryList: list[ReconstructorViewControllerFactory]
    ) -> ReconstructorController:
        controller = cls(presenter, productRepository, view, plotView, fileDialogFactory,
                         viewControllerFactoryList)
        presenter.addObserver(controller)
        productRepository.addObserver(controller)

        for name in presenter.getReconstructorList():
            controller._addReconstructor(name)

        view.reconstructorView.algorithmComboBox.textActivated.connect(presenter.setReconstructor)
        view.reconstructorView.algorithmComboBox.currentIndexChanged.connect(
            view.stackedWidget.setCurrentIndex)

        view.reconstructorView.productComboBox.textActivated.connect(controller._redrawPlot)
        view.reconstructorView.productComboBox.setModel(controller._productListModel)

        view.reconstructorView.reconstructButton.clicked.connect(controller._reconstruct)
        view.reconstructorView.reconstructSplitButton.clicked.connect(controller._reconstructSplit)
        view.reconstructorView.ingestButton.clicked.connect(controller._ingestTrainingData)
        view.reconstructorView.saveButton.clicked.connect(controller._saveTrainingData)
        view.reconstructorView.trainButton.clicked.connect(controller._train)
        view.reconstructorView.clearButton.clicked.connect(controller._clearTrainingData)

        controller._syncAlgorithmToView()
        controller._syncProductToView()

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
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._view.stackedWidget.addWidget(widget)

    def _reconstruct(self) -> None:
        outputProductName = self._presenter.getReconstructor()
        inputProductIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            logger.debug('No current index!')
            return

        try:
            self._presenter.reconstruct(inputProductIndex, outputProductName)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Reconstructor', err)

    def _reconstructSplit(self) -> None:
        outputProductName = self._presenter.getReconstructor()
        inputProductIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            logger.debug('No current index!')
            return

        try:
            self._presenter.reconstructSplit(inputProductIndex, outputProductName)
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Split Reconstructor', err)

    def _ingestTrainingData(self) -> None:
        inputProductIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if inputProductIndex < 0:
            logger.debug('No current index!')
            return

        try:
            self._presenter.ingestTrainingData(inputProductIndex)
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

    def _syncProductToView(self) -> None:
        self._productListModel.setStringList(product.getName()
                                             for product in self._productRepository)

    def _redrawPlot(self) -> None:
        productIndex = self._view.reconstructorView.productComboBox.currentIndex()

        if productIndex < 0:
            logger.debug('No current index!')
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
            self._presenter.getReconstructor())

        isTrainable = self._presenter.isTrainable
        self._view.reconstructorView.ingestButton.setVisible(isTrainable)
        self._view.reconstructorView.saveButton.setVisible(isTrainable)
        self._view.reconstructorView.trainButton.setVisible(isTrainable)
        self._view.reconstructorView.clearButton.setVisible(isTrainable)

        self._redrawPlot()

    def handleItemInserted(self, index: int, item: ProductRepositoryItem) -> None:
        self._syncProductToView()

    def handleMetadataChanged(self, index: int, item: MetadataRepositoryItem) -> None:
        self._syncProductToView()

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
        self._syncProductToView()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncAlgorithmToView()
