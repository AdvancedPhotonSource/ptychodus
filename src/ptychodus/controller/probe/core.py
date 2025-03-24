from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ptychodus.api.observer import SequenceObserver

from ...model.analysis import (
    IlluminationMapper,
    ProbePropagator,
    STXMSimulator,
)
from ...model.fluorescence import FluorescenceEnhancer
from ...model.product import ProbeAPI, ProbeRepository
from ...model.product.probe import ProbeRepositoryItem
from ...model.visualization import VisualizationEngine
from ...view.repository import RepositoryTreeView
from ...view.widgets import (
    ComboBoxItemDelegate,
    ExceptionDialog,
    ProgressBarItemDelegate,
)
from ..data import FileDialogFactory
from ..image import ImageController
from .editor_factory import ProbeEditorViewControllerFactory
from .illumination import IlluminationViewController
from .fluorescence import FluorescenceViewController
from .propagator import ProbePropagationViewController
from .stxm import STXMViewController
from .tree_model import ProbeTreeModel

logger = logging.getLogger(__name__)


class ProbeController(SequenceObserver[ProbeRepositoryItem]):
    def __init__(
        self,
        repository: ProbeRepository,
        api: ProbeAPI,
        imageController: ImageController,
        propagator: ProbePropagator,
        propagatorVisualizationEngine: VisualizationEngine,
        stxmSimulator: STXMSimulator,
        stxmVisualizationEngine: VisualizationEngine,
        exposureAnalyzer: IlluminationMapper,
        exposureVisualizationEngine: VisualizationEngine,
        fluorescenceEnhancer: FluorescenceEnhancer,
        fluorescenceVisualizationEngine: VisualizationEngine,
        view: RepositoryTreeView,
        fileDialogFactory: FileDialogFactory,
        treeModel: ProbeTreeModel,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._imageController = imageController
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = treeModel
        self._editorFactory = ProbeEditorViewControllerFactory()

        self._propagationViewController = ProbePropagationViewController(
            propagator, propagatorVisualizationEngine, fileDialogFactory
        )
        self._stxmViewController = STXMViewController(
            stxmSimulator, stxmVisualizationEngine, fileDialogFactory
        )
        self._exposureViewController = IlluminationViewController(
            exposureAnalyzer, exposureVisualizationEngine, fileDialogFactory
        )
        self._fluorescenceViewController = FluorescenceViewController(
            fluorescenceEnhancer, fluorescenceVisualizationEngine, fileDialogFactory
        )

    @classmethod
    def create_instance(
        cls,
        repository: ProbeRepository,
        api: ProbeAPI,
        imageController: ImageController,
        propagator: ProbePropagator,
        propagatorVisualizationEngine: VisualizationEngine,
        stxmSimulator: STXMSimulator,
        stxmVisualizationEngine: VisualizationEngine,
        exposureAnalyzer: IlluminationMapper,
        exposureVisualizationEngine: VisualizationEngine,
        fluorescenceEnhancer: FluorescenceEnhancer,
        fluorescenceVisualizationEngine: VisualizationEngine,
        view: RepositoryTreeView,
        fileDialogFactory: FileDialogFactory,
    ) -> ProbeController:
        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)
        treeModel = ProbeTreeModel(repository, api)
        controller = cls(
            repository,
            api,
            imageController,
            propagator,
            propagatorVisualizationEngine,
            stxmSimulator,
            stxmVisualizationEngine,
            exposureAnalyzer,
            exposureVisualizationEngine,
            fluorescenceEnhancer,
            fluorescenceVisualizationEngine,
            view,
            fileDialogFactory,
            treeModel,
        )
        repository.add_observer(controller)

        builderListModel = QStringListModel()
        builderListModel.setStringList([name for name in api.builder_names()])
        builderItemDelegate = ComboBoxItemDelegate(builderListModel, view.treeView)

        view.treeView.setModel(treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        powerItemDelegate = ProgressBarItemDelegate(view.treeView)
        view.treeView.setItemDelegateForColumn(1, powerItemDelegate)
        view.treeView.setItemDelegateForColumn(2, builderItemDelegate)
        view.treeView.selectionModel().currentChanged.connect(controller._updateView)
        controller._updateView(QModelIndex(), QModelIndex())

        loadFromFileAction = view.buttonBox.loadMenu.addAction('Open File...')
        loadFromFileAction.triggered.connect(controller._loadCurrentProbeFromFile)

        copyAction = view.buttonBox.loadMenu.addAction('Copy...')
        copyAction.triggered.connect(controller._copyToCurrentProbe)

        saveToFileAction = view.buttonBox.save_menu.addAction('Save File...')
        saveToFileAction.triggered.connect(controller._saveCurrentProbeToFile)

        syncToSettingsAction = view.buttonBox.save_menu.addAction('Sync To Settings')
        syncToSettingsAction.triggered.connect(controller._syncCurrentProbeToSettings)

        view.copierDialog.setWindowTitle('Copy Probe')
        view.copierDialog.source_combo_box.setModel(treeModel)
        view.copierDialog.destination_combo_box.setModel(treeModel)
        view.copierDialog.finished.connect(controller._finishCopyingProbe)

        view.buttonBox.edit_button.clicked.connect(controller._editCurrentProbe)

        propagateAction = view.buttonBox.analyze_menu.addAction('Propagate...')
        propagateAction.triggered.connect(controller._propagateProbe)

        stxmAction = view.buttonBox.analyze_menu.addAction('Simulate STXM...')
        stxmAction.triggered.connect(controller._simulateSTXM)

        exposureAction = view.buttonBox.analyze_menu.addAction('Exposure...')
        exposureAction.triggered.connect(controller._analyzeExposure)

        fluorescenceAction = view.buttonBox.analyze_menu.addAction('Enhance Fluorescence...')
        fluorescenceAction.triggered.connect(controller._enhanceFluorescence)

        return controller

    def _getCurrentItemIndex(self) -> int:
        modelIndex = self._view.treeView.currentIndex()

        if modelIndex.isValid():
            parent = modelIndex.parent()

            while parent.isValid():
                modelIndex = parent
                parent = modelIndex.parent()

            return modelIndex.row()

        logger.warning('No current index!')
        return -1

    def _loadCurrentProbeFromFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.get_open_file_path(
            self._view,
            'Open Probe',
            name_filters=[nameFilter for nameFilter in self._api.get_open_file_filters()],
            selected_name_filter=self._api.get_open_file_filter(),
        )

        if filePath:
            try:
                self._api.open_probe(itemIndex, filePath, file_type=nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Reader', err)

    def _copyToCurrentProbe(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex >= 0:
            self._view.copierDialog.destination_combo_box.setCurrentIndex(itemIndex)
            self._view.copierDialog.open()

    def _finishCopyingProbe(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            sourceIndex = self._view.copierDialog.source_combo_box.currentIndex()
            destinationIndex = self._view.copierDialog.destination_combo_box.currentIndex()
            self._api.copy_probe(sourceIndex, destinationIndex)

    def _editCurrentProbe(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        itemName = self._repository.get_name(itemIndex)
        item = self._repository[itemIndex]
        dialog = self._editorFactory.create_editor_dialog(itemName, item, self._view)
        dialog.open()

    def _saveCurrentProbeToFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.get_save_file_path(
            self._view,
            'Save Probe',
            name_filters=[nameFilter for nameFilter in self._api.get_save_file_filters()],
            selected_name_filter=self._api.get_save_file_filter(),
        )

        if filePath:
            try:
                self._api.save_probe(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Writer', err)

    def _syncCurrentProbeToSettings(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[itemIndex]
            item.sync_to_settings()

    def _propagateProbe(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._propagationViewController.launch(itemIndex)

    def _simulateSTXM(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._stxmViewController.simulate(itemIndex)

    def _analyzeExposure(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._exposureViewController.analyze(itemIndex)

    def _enhanceFluorescence(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._fluorescenceViewController.launch(itemIndex)

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.load_button.setEnabled(enabled)
        self._view.buttonBox.save_button.setEnabled(enabled)
        self._view.buttonBox.edit_button.setEnabled(enabled)
        self._view.buttonBox.analyze_button.setEnabled(enabled)

        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            self._imageController.clear_array()
        else:
            try:
                item = self._repository[itemIndex]
            except IndexError:
                logger.warning('Unable to access item for visualization!')
            else:
                probe = item.get_probe()
                array = (
                    probe.get_incoherent_mode(current.row())
                    if current.parent().isValid()
                    else probe.get_incoherent_modes_flattened()
                )
                pixelGeometry = probe.get_pixel_geometry()

                if pixelGeometry is None:
                    logger.warning('Missing probe pixel geometry!')
                else:
                    self._imageController.set_array(array, pixelGeometry)

    def handle_item_inserted(self, index: int, item: ProbeRepositoryItem) -> None:
        self._treeModel.insert_item(index, item)

    def handle_item_changed(self, index: int, item: ProbeRepositoryItem) -> None:
        self._treeModel.update_item(index, item)

        if index == self._getCurrentItemIndex():
            currentIndex = self._view.treeView.currentIndex()
            self._updateView(currentIndex, currentIndex)

    def handle_item_removed(self, index: int, item: ProbeRepositoryItem) -> None:
        self._treeModel.remove_item(index, item)
