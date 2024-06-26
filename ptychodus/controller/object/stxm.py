import logging

from ptychodus.api.observer import Observable, Observer

from ...model.analysis import STXMSimulator
from ...model.visualization import VisualizationEngine
from ...view.object import STXMDialog
from ...view.widgets import ExceptionDialog
from ..data import FileDialogFactory
from ..visualization import VisualizationParametersController, VisualizationWidgetController

logger = logging.getLogger(__name__)


class STXMViewController(Observer):

    def __init__(self, simulator: STXMSimulator, engine: VisualizationEngine,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._simulator = simulator
        self._fileDialogFactory = fileDialogFactory

        self._dialog = STXMDialog()
        self._dialog.saveButton.clicked.connect(self._saveResult)

        self._visualizationWidgetController = VisualizationWidgetController(
            engine, self._dialog.visualizationWidget, self._dialog.statusBar, fileDialogFactory)
        self._visualizationParametersController = VisualizationParametersController.createInstance(
            engine, self._dialog.visualizationParametersView)

        simulator.addObserver(self)

    def launch(self, productIndex: int) -> None:
        self._simulator.setProduct(productIndex)

        try:
            itemName = self._simulator.getProductName()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Launch', err)
        else:
            self._dialog.setWindowTitle(f'Simulate STXM: {itemName}')
            self._dialog.open()

        self._simulator.simulate()

    def _saveResult(self) -> None:
        title = 'Save STXM Image'
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._dialog,
            title,
            nameFilters=self._simulator.getSaveFileFilterList(),
            selectedNameFilter=self._simulator.getSaveFileFilter())

        if filePath:
            try:
                self._simulator.saveImage(filePath)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException(title, err)

    def _syncModelToView(self) -> None:
        image = self._simulator.getImage()

        try:
            self._visualizationWidgetController.setArray(image.intensity, image.pixel_geometry)
        except ValueError:
            self._visualizationWidgetController.clearArray()
        except Exception as err:
            logger.exception(err)
            ExceptionDialog.showException('Update Views', err)

    def update(self, observable: Observable) -> None:
        if observable is self._simulator:
            self._syncModelToView()
