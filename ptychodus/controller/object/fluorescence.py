import logging

from ...model.analysis import FluorescenceEnhancer
from ...model.visualization import VisualizationEngine
from ...view.object import FluorescenceDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class FluorescenceViewController:

    def __init__(self, enhancer: FluorescenceEnhancer, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._enhancer = enhancer
        self._engine = engine
        self._fileDialogFactory = fileDialogFactory
        self._dialog = FluorescenceDialog()

        self._dialog.fluorescenceParametersView.openButton.clicked.connect(
            self._openMeasuredDataset)
        # FIXME self._dialog.fluorescenceParametersView.upscalingStrategyComboBox: QComboBox
        # FIXME self._dialog.fluorescenceParametersView.deconvolutionStrategyComboBox: QComboBox
        self._dialog.fluorescenceParametersView.enhanceButton.clicked.connect(
            self._enhanceFluorescence)
        self._dialog.fluorescenceParametersView.saveButton.clicked.connect(
            self._saveEnhancedDataset)

        # FIXME self._dialog.fluorescenceChannelListView: QListView

        self._measuredWidgetController = VisualizationWidgetController(
            engine, self._dialog.measuredWidget, self._dialog.statusBar, fileDialogFactory)
        self._enhancedWidgetController = VisualizationWidgetController(
            engine, self._dialog.enhancedWidget, self._dialog.statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.visualizationParametersView)

    def _updateVisualizations(self) -> None:
        # FIXME self._measuredVisualizationWidgetController.setArray(result.measured, result.pixelGeometry)
        # FIXME self._enhancedVisualizationWidgetController.setArray(result.enhanced), result.pixelGeometry)
        pass

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
            return

    def enhanceFluorescence(self, itemIndex: int) -> None:
        itemName = self._enhancer.enhanceFluorescence(itemIndex)
        self._dialog.setWindowTitle(f'Enhance Fluorescence: {itemName}')
        self._updateVisualizations()
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
