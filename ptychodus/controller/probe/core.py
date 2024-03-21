from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog, QStatusBar

from ...api.observer import SequenceObserver
from ...model.analysis import ProbePropagator
from ...model.image import ImagePresenter
from ...model.product import ProbeRepository
from ...model.product.probe import ProbeRepositoryItem
from ...view.image import ImageView
from ...view.repository import RepositoryTreeView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog, ProgressBarItemDelegate
from ..data import FileDialogFactory
from ..image import ImageController
from .editorFactory import ProbeEditorViewControllerFactory
from .propagator import ProbePropagationViewController
from .treeModel import ProbeTreeModel

logger = logging.getLogger(__name__)


class ProbeController(SequenceObserver[ProbeRepositoryItem]):

    def __init__(self, repository: ProbeRepository, imagePresenter: ImagePresenter,
                 propagator: ProbePropagator, propagatorImagePresenter: ImagePresenter,
                 view: RepositoryTreeView, imageView: ImageView, statusBar: QStatusBar,
                 fileDialogFactory: FileDialogFactory, treeModel: ProbeTreeModel) -> None:
        super().__init__()
        self._repository = repository
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageView = imageView
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = treeModel
        self._editorFactory = ProbeEditorViewControllerFactory()
        self._imageController = ImageController.createInstance(imagePresenter, imageView,
                                                               fileDialogFactory)
        self._propagationViewController = ProbePropagationViewController(
            propagator, propagatorImagePresenter, fileDialogFactory, statusBar, view)

    @classmethod
    def createInstance(cls, repository: ProbeRepository, imagePresenter: ImagePresenter,
                       propagator: ProbePropagator, propagatorImagePresenter: ImagePresenter,
                       view: RepositoryTreeView, imageView: ImageView, statusBar: QStatusBar,
                       fileDialogFactory: FileDialogFactory) -> ProbeController:
        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)
        treeModel = ProbeTreeModel(repository)
        controller = cls(repository, imagePresenter, propagator, propagatorImagePresenter, view,
                         imageView, statusBar, fileDialogFactory, treeModel)
        repository.addObserver(controller)

        builderListModel = QStringListModel()
        builderListModel.setStringList([name for name in repository.builderNames()])
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

        view.copierDialog.setWindowTitle('Copy Probe')
        view.copierDialog.sourceComboBox.setModel(treeModel)
        view.copierDialog.destinationComboBox.setModel(treeModel)
        view.copierDialog.finished.connect(controller._finishCopyingProbe)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentProbe)
        view.buttonBox.saveButton.clicked.connect(controller._saveCurrentProbe)

        propagateAction = view.buttonBox.analyzeMenu.addAction('Propagate...')
        propagateAction.triggered.connect(controller._propagateProbe)

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

        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Probe',
            nameFilters=self._repository.getOpenFileFilterList(),
            selectedNameFilter=self._repository.getOpenFileFilter())

        if filePath:
            try:
                self._repository.openProbe(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Reader', err)

    def _copyToCurrentProbe(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex >= 0:
            self._view.copierDialog.destinationComboBox.setCurrentIndex(itemIndex)
            self._view.copierDialog.open()

    def _finishCopyingProbe(self, result: int) -> None:
        if result == QDialog.DialogCode.Accepted:
            sourceIndex = self._view.copierDialog.sourceComboBox.currentIndex()
            destinationIndex = self._view.copierDialog.destinationComboBox.currentIndex()
            self._repository.copyProbe(sourceIndex, destinationIndex)

    def _editCurrentProbe(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        itemName = self._repository.getName(itemIndex)
        item = self._repository[itemIndex]
        dialog = self._editorFactory.createEditorDialog(itemName, item, self._view)
        dialog.open()

    def _saveCurrentProbe(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Probe',
            nameFilters=self._repository.getSaveFileFilterList(),
            selectedNameFilter=self._repository.getSaveFileFilter())

        if filePath:
            try:
                self._repository.saveProbe(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)

    def _propagateProbe(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._propagationViewController.propagate(itemIndex)

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.loadButton.setEnabled(enabled)
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.editButton.setEnabled(enabled)
        self._view.buttonBox.analyzeButton.setEnabled(enabled)

        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            self._imagePresenter.clearArray()
        else:
            try:
                item = self._repository[itemIndex]
            except IndexError:
                logger.warning('Unable to access item for visualization!')
            else:
                probe = item.getProbe()
                array = probe.getMode(current.row()) if current.parent().isValid() \
                        else probe.getModesFlattened()
                self._imagePresenter.setArray(array, probe.getPixelGeometry())

    def handleItemInserted(self, index: int, item: ProbeRepositoryItem) -> None:
        self._treeModel.insertItem(index, item)

    def handleItemChanged(self, index: int, item: ProbeRepositoryItem) -> None:
        self._treeModel.updateItem(index, item)

        if index == self._getCurrentItemIndex():
            currentIndex = self._view.treeView.currentIndex()
            self._updateView(currentIndex, currentIndex)

    def handleItemRemoved(self, index: int, item: ProbeRepositoryItem) -> None:
        self._treeModel.removeItem(index, item)