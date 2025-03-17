from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ptychodus.api.observer import SequenceObserver

from ...model.analysis import FourierRingCorrelator, XMCDAnalyzer
from ...model.product import ObjectAPI, ObjectRepository
from ...model.product.object import ObjectRepositoryItem
from ...model.visualization import VisualizationEngine
from ...view.repository import RepositoryTreeView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .editor_factory import ObjectEditorViewControllerFactory
from .frc import FourierRingCorrelationViewController
from .tree_model import ObjectTreeModel
from .xmcd import XMCDViewController

logger = logging.getLogger(__name__)


class ObjectController(SequenceObserver[ObjectRepositoryItem]):
    def __init__(
        self,
        repository: ObjectRepository,
        api: ObjectAPI,
        imageController: ImageController,
        correlator: FourierRingCorrelator,
        xmcdAnalyzer: XMCDAnalyzer,
        xmcdVisualizationEngine: VisualizationEngine,
        view: RepositoryTreeView,
        fileDialogFactory: FileDialogFactory,
        treeModel: ObjectTreeModel,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._imageController = imageController
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = treeModel
        self._editorFactory = ObjectEditorViewControllerFactory()

        self._frcViewController = FourierRingCorrelationViewController(correlator, treeModel)
        self._xmcdViewController = XMCDViewController(
            xmcdAnalyzer, xmcdVisualizationEngine, fileDialogFactory, treeModel
        )

    @classmethod
    def create_instance(
        cls,
        repository: ObjectRepository,
        api: ObjectAPI,
        imageController: ImageController,
        correlator: FourierRingCorrelator,
        xmcdAnalyzer: XMCDAnalyzer,
        xmcdVisualizationEngine: VisualizationEngine,
        view: RepositoryTreeView,
        fileDialogFactory: FileDialogFactory,
    ) -> ObjectController:
        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)
        treeModel = ObjectTreeModel(repository, api)
        controller = cls(
            repository,
            api,
            imageController,
            correlator,
            xmcdAnalyzer,
            xmcdVisualizationEngine,
            view,
            fileDialogFactory,
            treeModel,
        )
        repository.add_observer(controller)

        builderListModel = QStringListModel()
        builderListModel.setStringList([name for name in api.builderNames()])
        builderItemDelegate = ComboBoxItemDelegate(builderListModel, view.treeView)

        view.treeView.setModel(treeModel)
        view.treeView.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.treeView.setItemDelegateForColumn(2, builderItemDelegate)
        view.treeView.selectionModel().currentChanged.connect(controller._updateView)
        controller._updateView(QModelIndex(), QModelIndex())

        loadFromFileAction = view.buttonBox.loadMenu.addAction('Open File...')
        loadFromFileAction.triggered.connect(controller._loadCurrentObjectFromFile)

        copyAction = view.buttonBox.loadMenu.addAction('Copy...')
        copyAction.triggered.connect(controller._copyToCurrentObject)

        saveToFileAction = view.buttonBox.saveMenu.addAction('Save File...')
        saveToFileAction.triggered.connect(controller._saveCurrentObjectToFile)

        syncToSettingsAction = view.buttonBox.saveMenu.addAction('Sync To Settings')
        syncToSettingsAction.triggered.connect(controller._syncCurrentObjectToSettings)

        view.copierDialog.setWindowTitle('Copy Object')
        view.copierDialog.sourceComboBox.setModel(treeModel)
        view.copierDialog.destinationComboBox.setModel(treeModel)
        view.copierDialog.finished.connect(controller._finishCopyingObject)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentObject)

        frcAction = view.buttonBox.analyzeMenu.addAction('Fourier Ring Correlation...')
        frcAction.triggered.connect(controller._analyzeFRC)

        xmcdAction = view.buttonBox.analyzeMenu.addAction('XMCD...')
        xmcdAction.triggered.connect(controller._analyzeXMCD)

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

    def _loadCurrentObjectFromFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Object',
            nameFilters=[nameFilter for nameFilter in self._api.getOpenFileFilterList()],
            selectedNameFilter=self._api.getOpenFileFilter(),
        )

        if filePath:
            try:
                self._api.openObject(itemIndex, filePath, file_type=nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Reader', err)

    def _copyToCurrentObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex >= 0:
            self._view.copierDialog.destinationComboBox.setCurrentIndex(itemIndex)
            self._view.copierDialog.open()

    def _finishCopyingObject(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            sourceIndex = self._view.copierDialog.sourceComboBox.currentIndex()
            destinationIndex = self._view.copierDialog.destinationComboBox.currentIndex()
            self._api.copyObject(sourceIndex, destinationIndex)

    def _editCurrentObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        itemName = self._repository.getName(itemIndex)
        item = self._repository[itemIndex]
        dialog = self._editorFactory.createEditorDialog(itemName, item, self._view)
        dialog.open()

    def _saveCurrentObjectToFile(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.get_save_file_path(
            self._view,
            'Save Object',
            name_filters=[nameFilter for nameFilter in self._api.getSaveFileFilterList()],
            selected_name_filter=self._api.getSaveFileFilter(),
        )

        if filePath:
            try:
                self._api.saveObject(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.show_exception('File Writer', err)

    def _syncCurrentObjectToSettings(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            item = self._repository[itemIndex]
            item.syncToSettings()

    def _analyzeFRC(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._frcViewController.analyze(itemIndex, itemIndex)

    def _analyzeXMCD(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._xmcdViewController._analyze(itemIndex, itemIndex)

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.loadButton.setEnabled(enabled)
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.editButton.setEnabled(enabled)
        self._view.buttonBox.analyzeButton.setEnabled(enabled)

        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            self._imageController.clear_array()
        else:
            try:
                item = self._repository[itemIndex]
            except IndexError:
                logger.warning('Unable to access item for visualization!')
            else:
                object_ = item.get_object()
                array = (
                    object_.get_layer(current.row())
                    if current.parent().isValid()
                    else object_.get_layers_flattened()
                )
                pixelGeometry = object_.get_pixel_geometry()

                if pixelGeometry is None:
                    logger.warning('Missing object pixel geometry!')
                else:
                    self._imageController.set_array(array, pixelGeometry)

    def handle_item_inserted(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.insertItem(index, item)

    def handle_item_changed(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.updateItem(index, item)

        if index == self._getCurrentItemIndex():
            currentIndex = self._view.treeView.currentIndex()
            self._updateView(currentIndex, currentIndex)

    def handle_item_removed(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.removeItem(index, item)
