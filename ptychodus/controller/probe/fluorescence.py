from typing import Any
import logging

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QStringListModel

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import FluorescenceEnhancer
from ...model.visualization import VisualizationEngine
from ...view.probe import FluorescenceDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class FluorescenceChannelListModel(QAbstractListModel):

    def __init__(self, enhancer: FluorescenceEnhancer, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._enhancer = enhancer

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        # TODO make this a table model and show measured/enhanced count statistics
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            emap = self._enhancer.getMeasuredElementMap(index.row())
            return emap.name

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._enhancer.getNumberOfChannels()


class FluorescenceViewController(Observer):

    def __init__(self, enhancer: FluorescenceEnhancer, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._engine = engine
        self._fileDialogFactory = fileDialogFactory
        self._dialog = FluorescenceDialog()
        self._upscalingModel = QStringListModel()
        self._upscalingModel.setStringList(self._enhancer.getUpscalingStrategyList())
        self._deconvolutionModel = QStringListModel()
        self._deconvolutionModel.setStringList(self._enhancer.getDeconvolutionStrategyList())
        self._channelListModel = FluorescenceChannelListModel(enhancer)

        self._dialog.fluorescenceParametersView.openButton.clicked.connect(
            self._openMeasuredDataset)

        self._dialog.fluorescenceParametersView.upscalingStrategyComboBox.setModel(
            self._upscalingModel)
        self._dialog.fluorescenceParametersView.upscalingStrategyComboBox.textActivated.connect(
            enhancer.setUpscalingStrategy)

        self._dialog.fluorescenceParametersView.deconvolutionStrategyComboBox.setModel(
            self._deconvolutionModel)
        self._dialog.fluorescenceParametersView.deconvolutionStrategyComboBox.textActivated.connect(
            enhancer.setDeconvolutionStrategy)

        self._dialog.fluorescenceChannelListView.setModel(self._channelListModel)
        self._dialog.fluorescenceChannelListView.selectionModel().currentChanged.connect(
            self._updateView)

        self._dialog.fluorescenceParametersView.enhanceButton.clicked.connect(
            self._enhanceFluorescence)
        self._dialog.fluorescenceParametersView.saveButton.clicked.connect(
            self._saveEnhancedDataset)

        self._measuredWidgetController = VisualizationWidgetController(
            engine, self._dialog.measuredWidget, self._dialog.statusBar, fileDialogFactory)
        self._enhancedWidgetController = VisualizationWidgetController(
            engine, self._dialog.enhancedWidget, self._dialog.statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.visualizationParametersView)

        enhancer.addObserver(self)

    def _openMeasuredDataset(self) -> None:
        title = 'Open Measured Fluorescence Dataset'
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._dialog,
            title,
            nameFilters=self._enhancer.getOpenFileFilterList(),
            selectedNameFilter=self._enhancer.getOpenFileFilter())

        if filePath:
            try:
                self._enhancer.openMeasuredDataset(filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException(title, err)

    def _enhanceFluorescence(self) -> None:
        try:
            self._enhancer.enhanceFluorescence()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Enhance Fluorescence', err)

    def launch(self, productIndex: int) -> None:
        self._enhancer.setProduct(productIndex)

        try:
            itemName = self._enhancer.getProductName()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Launch', err)
        else:
            self._dialog.setWindowTitle(f'Enhance Fluorescence: {itemName}')
            self._dialog.open()

    def _saveEnhancedDataset(self) -> None:
        title = 'Save Enhanced Fluorescence Dataset'
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._dialog,
            title,
            nameFilters=self._enhancer.getSaveFileFilterList(),
            selectedNameFilter=self._enhancer.getSaveFileFilter())

        if filePath:
            try:
                self._enhancer.saveEnhancedDataset(filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException(title, err)

    def _syncModelToView(self) -> None:
        self._dialog.fluorescenceParametersView.upscalingStrategyComboBox.setCurrentText(
            self._enhancer.getUpscalingStrategy())
        self._dialog.fluorescenceParametersView.deconvolutionStrategyComboBox.setCurrentText(
            self._enhancer.getDeconvolutionStrategy())

        self._channelListModel.beginResetModel()
        self._channelListModel.endResetModel()

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid():
            self._measuredWidgetController.clearArray()
            self._enhancedWidgetController.clearArray()
            return

        try:
            emap_measured = self._enhancer.getMeasuredElementMap(current.row())
        except Exception as err:
            logger.exception(err)
            self._measuredWidgetController.clearArray()
            ExceptionDialog.showException('Render Measured Element Map', err)
        else:
            self._measuredWidgetController.setArray(emap_measured.counts_per_second,
                                                    self._enhancer.getPixelGeometry())

        try:
            emap_enhanced = self._enhancer.getEnhancedElementMap(current.row())
        except Exception as err:
            logger.exception(err)
            self._enhancedWidgetController.clearArray()
            ExceptionDialog.showException('Render Enhanced Element Map', err)
        else:
            self._enhancedWidgetController.setArray(emap_enhanced.counts_per_second,
                                                    self._enhancer.getPixelGeometry())

    def update(self, observable: Observable) -> None:
        if observable is self._enhancer:
            self._syncModelToView()
