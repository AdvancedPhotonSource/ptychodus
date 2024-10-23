from decimal import Decimal
from typing import Any, Final
import logging

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QStringListModel
from PyQt5.QtWidgets import QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.fluorescence import (
    FluorescenceEnhancer,
    TwoStepFluorescenceEnhancingAlgorithm,
    VSPIFluorescenceEnhancingAlgorithm,
)
from ...model.visualization import VisualizationEngine
from ...view.probe import (
    FluorescenceDialog,
    FluorescenceTwoStepParametersView,
    FluorescenceVSPIParametersView,
)
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import (
    VisualizationParametersController,
    VisualizationWidgetController,
)

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


class FluorescenceTwoStepViewController(Observer):
    def __init__(self, algorithm: TwoStepFluorescenceEnhancingAlgorithm) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._view = FluorescenceTwoStepParametersView()

        self._upscalingModel = QStringListModel()
        self._upscalingModel.setStringList(self._algorithm.getUpscalingStrategyList())
        self._view.upscalingStrategyComboBox.setModel(self._upscalingModel)
        self._view.upscalingStrategyComboBox.textActivated.connect(algorithm.setUpscalingStrategy)

        self._deconvolutionModel = QStringListModel()
        self._deconvolutionModel.setStringList(self._algorithm.getDeconvolutionStrategyList())
        self._view.deconvolutionStrategyComboBox.setModel(self._deconvolutionModel)
        self._view.deconvolutionStrategyComboBox.textActivated.connect(
            algorithm.setDeconvolutionStrategy
        )

        self._syncModelToView()
        algorithm.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._view

    def _syncModelToView(self) -> None:
        self._view.upscalingStrategyComboBox.setCurrentText(self._algorithm.getUpscalingStrategy())
        self._view.deconvolutionStrategyComboBox.setCurrentText(
            self._algorithm.getDeconvolutionStrategy()
        )

    def update(self, observable: Observable) -> None:
        if observable is self._algorithm:
            self._syncModelToView()


class FluorescenceVSPIViewController(Observer):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(self, algorithm: VSPIFluorescenceEnhancingAlgorithm) -> None:
        super().__init__()
        self._algorithm = algorithm
        self._view = FluorescenceVSPIParametersView()

        self._view.dampingFactorLineEdit.valueChanged.connect(self._syncDampingFactorToModel)
        self._view.maxIterationsSpinBox.setRange(1, self.MAX_INT)
        self._view.maxIterationsSpinBox.valueChanged.connect(algorithm.setMaxIterations)

        algorithm.addObserver(self)
        self._syncModelToView()

    def getWidget(self) -> QWidget:
        return self._view

    def _syncDampingFactorToModel(self, value: Decimal) -> None:
        self._algorithm.setDampingFactor(float(value))

    def _syncModelToView(self) -> None:
        self._view.dampingFactorLineEdit.setValue(Decimal(repr(self._algorithm.getDampingFactor())))
        self._view.maxIterationsSpinBox.setValue(self._algorithm.getMaxIterations())

    def update(self, observable: Observable) -> None:
        if observable is self._algorithm:
            self._syncModelToView()


class FluorescenceViewController(Observer):
    def __init__(
        self,
        enhancer: FluorescenceEnhancer,
        engine: VisualizationEngine,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._engine = engine
        self._fileDialogFactory = fileDialogFactory
        self._dialog = FluorescenceDialog()
        self._enhancementModel = QStringListModel()
        self._enhancementModel.setStringList(self._enhancer.getAlgorithmList())
        self._channelListModel = FluorescenceChannelListModel(enhancer)

        self._dialog.fluorescenceParametersView.openButton.clicked.connect(
            self._openMeasuredDataset
        )

        twoStepViewController = FluorescenceTwoStepViewController(
            enhancer.twoStepEnhancingAlgorithm
        )
        self._dialog.fluorescenceParametersView.algorithmComboBox.addItem(
            TwoStepFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
            self._dialog.fluorescenceParametersView.algorithmComboBox.count(),
        )
        self._dialog.fluorescenceParametersView.stackedWidget.addWidget(
            twoStepViewController.getWidget()
        )

        vspiViewController = FluorescenceVSPIViewController(enhancer.vspiEnhancingAlgorithm)
        self._dialog.fluorescenceParametersView.algorithmComboBox.addItem(
            VSPIFluorescenceEnhancingAlgorithm.DISPLAY_NAME,
            self._dialog.fluorescenceParametersView.algorithmComboBox.count(),
        )
        self._dialog.fluorescenceParametersView.stackedWidget.addWidget(
            vspiViewController.getWidget()
        )

        self._dialog.fluorescenceParametersView.algorithmComboBox.textActivated.connect(
            enhancer.setAlgorithm
        )
        self._dialog.fluorescenceParametersView.algorithmComboBox.currentIndexChanged.connect(
            self._dialog.fluorescenceParametersView.stackedWidget.setCurrentIndex
        )
        self._dialog.fluorescenceParametersView.algorithmComboBox.setModel(self._enhancementModel)
        self._dialog.fluorescenceParametersView.algorithmComboBox.textActivated.connect(
            enhancer.setAlgorithm
        )

        self._dialog.fluorescenceParametersView.enhanceButton.clicked.connect(
            self._enhanceFluorescence
        )
        self._dialog.fluorescenceParametersView.saveButton.clicked.connect(
            self._saveEnhancedDataset
        )

        self._dialog.fluorescenceChannelListView.setModel(self._channelListModel)
        self._dialog.fluorescenceChannelListView.selectionModel().currentChanged.connect(
            self._updateView
        )

        self._measuredWidgetController = VisualizationWidgetController(
            engine,
            self._dialog.measuredWidget,
            self._dialog.statusBar,
            fileDialogFactory,
        )
        self._enhancedWidgetController = VisualizationWidgetController(
            engine,
            self._dialog.enhancedWidget,
            self._dialog.statusBar,
            fileDialogFactory,
        )
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.visualizationParametersView
        )

        enhancer.addObserver(self)

    def _openMeasuredDataset(self) -> None:
        title = 'Open Measured Fluorescence Dataset'
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._dialog,
            title,
            nameFilters=self._enhancer.getOpenFileFilterList(),
            selectedNameFilter=self._enhancer.getOpenFileFilter(),
        )

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
            selectedNameFilter=self._enhancer.getSaveFileFilter(),
        )

        if filePath:
            try:
                self._enhancer.saveEnhancedDataset(filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException(title, err)

    def _syncModelToView(self) -> None:
        self._dialog.fluorescenceParametersView.algorithmComboBox.setCurrentText(
            self._enhancer.getAlgorithm()
        )
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
            self._measuredWidgetController.setArray(
                emap_measured.counts_per_second, self._enhancer.getPixelGeometry()
            )

        try:
            emap_enhanced = self._enhancer.getEnhancedElementMap(current.row())
        except Exception as err:
            logger.exception(err)
            self._enhancedWidgetController.clearArray()
            ExceptionDialog.showException('Render Enhanced Element Map', err)
        else:
            self._enhancedWidgetController.setArray(
                emap_enhanced.counts_per_second, self._enhancer.getPixelGeometry()
            )

    def update(self, observable: Observable) -> None:
        if observable is self._enhancer:
            self._syncModelToView()
