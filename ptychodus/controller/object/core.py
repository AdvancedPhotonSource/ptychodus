from __future__ import annotations
import logging

from PyQt5.QtCore import QModelIndex, QStringListModel
from PyQt5.QtWidgets import QAbstractItemView, QDialog

from ptychodus.api.observer import SequenceObserver

from ...model.analysis import (ExposureAnalyzer, FluorescenceEnhancer, FourierRingCorrelator,
                               STXMAnalyzer, XMCDAnalyzer)
from ...model.product import ObjectAPI, ObjectRepository
from ...model.product.object import ObjectRepositoryItem
from ...model.visualization import VisualizationEngine
from ...view.repository import RepositoryTreeView
from ...view.widgets import ComboBoxItemDelegate, ExceptionDialog
from ..data import FileDialogFactory
from ..image import ImageController
from .editorFactory import ObjectEditorViewControllerFactory
from .exposure import ExposureViewController
from .frc import FourierRingCorrelationViewController
from .stxm import STXMViewController
from .treeModel import ObjectTreeModel
from .xmcd import XMCDViewController
from .fluorescence import FluorescenceViewController

logger = logging.getLogger(__name__)


class ObjectController(SequenceObserver[ObjectRepositoryItem]):

    def __init__(self, repository: ObjectRepository, api: ObjectAPI,
                 imageController: ImageController, correlator: FourierRingCorrelator,
                 stxmAnalyzer: STXMAnalyzer, stxmVisualizationEngine: VisualizationEngine,
                 exposureAnalyzer: ExposureAnalyzer,
                 exposureVisualizationEngine: VisualizationEngine,
                 fluorescenceEnhancer: FluorescenceEnhancer,
                 fluorescenceVisualizationEngine: VisualizationEngine, xmcdAnalyzer: XMCDAnalyzer,
                 xmcdVisualizationEngine: VisualizationEngine, view: RepositoryTreeView,
                 fileDialogFactory: FileDialogFactory, treeModel: ObjectTreeModel) -> None:
        super().__init__()
        self._repository = repository
        self._api = api
        self._imageController = imageController
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = treeModel
        self._editorFactory = ObjectEditorViewControllerFactory()

        self._frcViewController = FourierRingCorrelationViewController(correlator, treeModel)
        self._stxmViewController = STXMViewController(stxmAnalyzer, stxmVisualizationEngine,
                                                      fileDialogFactory)
        self._exposureViewController = ExposureViewController(exposureAnalyzer,
                                                              exposureVisualizationEngine,
                                                              fileDialogFactory)
        self._fluorescenceViewController = FluorescenceViewController(
            fluorescenceEnhancer, fluorescenceVisualizationEngine, fileDialogFactory)
        self._xmcdViewController = XMCDViewController(xmcdAnalyzer, xmcdVisualizationEngine,
                                                      fileDialogFactory, treeModel)

    @classmethod
    def createInstance(cls, repository: ObjectRepository, api: ObjectAPI,
                       imageController: ImageController, correlator: FourierRingCorrelator,
                       stxmAnalyzer: STXMAnalyzer, stxmVisualizationEngine: VisualizationEngine,
                       exposureAnalyzer: ExposureAnalyzer,
                       exposureVisualizationEngine: VisualizationEngine,
                       fluorescenceEnhancer: FluorescenceEnhancer,
                       fluorescenceVisualizationEngine: VisualizationEngine,
                       xmcdAnalyzer: XMCDAnalyzer, xmcdVisualizationEngine: VisualizationEngine,
                       view: RepositoryTreeView,
                       fileDialogFactory: FileDialogFactory) -> ObjectController:
        # TODO figure out good fix when saving NPY file without suffix (numpy adds suffix)
        treeModel = ObjectTreeModel(repository, api)
        controller = cls(repository, api, imageController, correlator, stxmAnalyzer,
                         stxmVisualizationEngine, exposureAnalyzer, exposureVisualizationEngine,
                         fluorescenceEnhancer, fluorescenceVisualizationEngine, xmcdAnalyzer,
                         xmcdVisualizationEngine, view, fileDialogFactory, treeModel)
        repository.addObserver(controller)

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

        view.copierDialog.setWindowTitle('Copy Object')
        view.copierDialog.sourceComboBox.setModel(treeModel)
        view.copierDialog.destinationComboBox.setModel(treeModel)
        view.copierDialog.finished.connect(controller._finishCopyingObject)

        view.buttonBox.editButton.clicked.connect(controller._editCurrentObject)
        view.buttonBox.saveButton.clicked.connect(controller._saveCurrentObject)

        frcAction = view.buttonBox.analyzeMenu.addAction('Fourier Ring Correlation...')
        frcAction.triggered.connect(controller._analyzeFRC)

        stxmAction = view.buttonBox.analyzeMenu.addAction('STXM...')
        stxmAction.triggered.connect(controller._analyzeSTXM)

        exposureAction = view.buttonBox.analyzeMenu.addAction('Exposure...')
        exposureAction.triggered.connect(controller._analyzeExposure)

        fluorescenceAction = view.buttonBox.analyzeMenu.addAction('Enhance Fluorescence...')
        fluorescenceAction.triggered.connect(controller._enhanceFluorescence)

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
            nameFilters=self._api.getOpenFileFilterList(),
            selectedNameFilter=self._api.getOpenFileFilter())

        if filePath:
            try:
                self._api.openObject(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Reader', err)

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

    def _saveCurrentObject(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            return

        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Object',
            nameFilters=self._api.getSaveFileFilterList(),
            selectedNameFilter=self._api.getSaveFileFilter())

        if filePath:
            try:
                self._api.saveObject(itemIndex, filePath, nameFilter)
            except Exception as err:
                logger.exception(err)
                ExceptionDialog.showException('File Writer', err)

    def _analyzeFRC(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._frcViewController.analyze(itemIndex, itemIndex)

    def _analyzeSTXM(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._stxmViewController.analyze(itemIndex)

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
            self._fluorescenceViewController.enhanceFluorescence(itemIndex)

    def _analyzeXMCD(self) -> None:
        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            logger.warning('No current item!')
        else:
            self._xmcdViewController.analyze(itemIndex, itemIndex)

    def _updateView(self, current: QModelIndex, previous: QModelIndex) -> None:
        enabled = current.isValid()
        self._view.buttonBox.loadButton.setEnabled(enabled)
        self._view.buttonBox.saveButton.setEnabled(enabled)
        self._view.buttonBox.editButton.setEnabled(enabled)
        self._view.buttonBox.analyzeButton.setEnabled(enabled)

        itemIndex = self._getCurrentItemIndex()

        if itemIndex < 0:
            self._imageController.clearArray()
        else:
            try:
                item = self._repository[itemIndex]
            except IndexError:
                logger.warning('Unable to access item for visualization!')
            else:
                object_ = item.getObject()
                array = object_.getLayer(current.row()) if current.parent().isValid() \
                        else object_.getLayersFlattened()
                self._imageController.setArray(array, object_.getPixelGeometry())

    def handleItemInserted(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.insertItem(index, item)

    def handleItemChanged(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.updateItem(index, item)

        if index == self._getCurrentItemIndex():
            currentIndex = self._view.treeView.currentIndex()
            self._updateView(currentIndex, currentIndex)

    def handleItemRemoved(self, index: int, item: ObjectRepositoryItem) -> None:
        self._treeModel.removeItem(index, item)
