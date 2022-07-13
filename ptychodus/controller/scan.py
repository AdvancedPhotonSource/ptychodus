from __future__ import annotations
from decimal import Decimal

from ..model import Observer, Observable, Scan, ScanPresenter
from ..view import (ScanParametersView, ScanPlotView, ScanEditorView, ScanTransformView)
from .data import FileDialogFactory
from .tree import CheckableTreeModel


class ScanEditorController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanEditorView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter,
                       view: ScanEditorView) -> ScanEditorController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numberOfScanPointsSpinBox.setEnabled(False)
        view.extentXSpinBox.valueChanged.connect(presenter.setExtentX)
        view.extentYSpinBox.valueChanged.connect(presenter.setExtentY)

        view.stepSizeXWidget.lengthChanged.connect(presenter.setStepSizeXInMeters)
        view.stepSizeYWidget.lengthChanged.connect(presenter.setStepSizeYInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfScanPointsSpinBox.blockSignals(True)
        self._view.numberOfScanPointsSpinBox.setRange(
            self._presenter.getNumberOfScanPointsLimits().lower,
            self._presenter.getNumberOfScanPointsLimits().upper)
        self._view.numberOfScanPointsSpinBox.setValue(self._presenter.getNumberOfScanPoints())
        self._view.numberOfScanPointsSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getExtentXLimits().lower,
                                           self._presenter.getExtentXLimits().upper)
        self._view.extentXSpinBox.setValue(self._presenter.getExtentX())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getExtentYLimits().lower,
                                           self._presenter.getExtentYLimits().upper)
        self._view.extentYSpinBox.setValue(self._presenter.getExtentY())
        self._view.extentYSpinBox.blockSignals(False)

        self._view.stepSizeXWidget.setLengthInMeters(self._presenter.getStepSizeXInMeters())
        self._view.stepSizeYWidget.setLengthInMeters(self._presenter.getStepSizeYInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanTransformController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanTransformView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter,
                       view: ScanTransformView) -> ScanTransformController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        for transform in presenter.getTransformList():
            view.transformComboBox.addItem(transform)

        view.transformComboBox.currentTextChanged.connect(presenter.setTransform)
        view.jitterRadiusWidget.lengthChanged.connect(presenter.setJitterRadiusInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.transformComboBox.setCurrentText(self._presenter.getTransform())
        self._view.jitterRadiusWidget.setLengthInMeters(self._presenter.getJitterRadiusInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class ScanParametersController:

    def __init__(self, presenter: ScanPresenter, view: ScanParametersView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._treeModel = CheckableTreeModel(presenter.getScanTree())

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanParametersView,
                       fileDialogFactory: FileDialogFactory) -> ScanParametersController:
        controller = cls(presenter, view, fileDialogFactory)

        view.treeView.setModel(controller._treeModel)
        view.treeView.selectionModel().currentChanged.connect(controller._setActiveScan)

        view.buttonBox.editButton.clicked.connect(controller._editSelectedScan)
        view.buttonBox.saveButton.clicked.connect(controller._saveSelectedScan)
        view.buttonBox.removeButton.clicked.connect(controller._removeSelectedScan)

        # TODO treeview selected/checked

        controller._syncModelToView()

        return controller

    def _openScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Scan',
            nameFilters=self._presenter.getOpenFileFilterList(),
            selectedNameFilter=self._presenter.getOpenFileFilter())

        if filePath:
            self._presenter.openScan(filePath, nameFilter)

    def _getSelectedScan(self) -> str:
        # TODO handle case of no selection
        current = self._view.treeView.selectionModel().currentIndex()
        nodeItem = current.internalPointer()
        return nodeItem.data(0)

    def _saveSelectedScan(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Scan',
            nameFilters=self._presenter.getSaveFileFilterList(),
            selectedNameFilter=self._presenter.getSaveFileFilter())

        if filePath:
            name = self._getSelectedScan()
            self._presenter.saveScan(filePath, nameFilter, name)

    def _editSelectedScan(self) -> None:
        name = self._getSelectedScan()
        print(f'editScan({name})')  # TODO

    def _removeSelectedScan(self) -> None:
        name = self._getSelectedScan()
        self._presenter.removeScan(name)

    def _setButtonsEnabled(self) -> None:
        hasSelection = self._view.treeView.selectionModel().hasSelection()
        self._view.buttonBox.insertButton.setEnabled(True)
        self._view.buttonBox.editButton.setEnabled(hasSelection)
        self._view.buttonBox.saveButton.setEnabled(hasSelection)
        self._view.buttonBox.removeButton.setEnabled(hasSelection)  # TODO and count > 1

    def _setActiveScan(self, current: QModelIndex, previous: QModelIndex) -> None:
        nodeItem = current.internalPointer()
        name = nodeItem.data(0)
        self._presenter.setActiveScan(name)
        self._setButtonsEnabled()

    def _syncModelToView(self) -> None:
        self._view.buttonBox.insertMenu.clear()

        for entry in self._presenter.getAvailableInitializerList():
            self._view.buttonBox.insertMenu.addAction(entry)

        self._setButtonsEnabled()

        # TODO presenter.getActiveScan() to update treeview selection
        # TODO do something with treeview checkboxes
        self._treeModel.setRootNode(self._presenter.getScanTree())


class ScanPlotController(Observer):

    def __init__(self, presenter: ScanPresenter, view: ScanPlotView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: ScanPresenter, view: ScanPlotView) -> ScanPlotController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)
        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        scanPath = self._presenter.getActiveScanPointList()

        x = [point.x for point in scanPath]
        y = [point.y for point in scanPath]

        self._view.axes.clear()
        self._view.axes.plot(x, y, '.-', linewidth=1.5)
        self._view.axes.invert_yaxis()
        self._view.axes.axis('equal')
        self._view.axes.grid(True)
        self._view.axes.set_xlabel('X [m]')
        self._view.axes.set_ylabel('Y [m]')
        self._view.figureCanvas.draw()

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
