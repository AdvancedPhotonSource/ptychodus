from __future__ import annotations
from decimal import Decimal
from pathlib import Path

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFileDialog

from ..api.data import DatasetState
from ..api.observer import Observable, Observer
from ..model import (CropPresenter, DataFilePresenter, DetectorPresenter,
                     DiffractionDatasetPresenter, ImagePresenter)
from ..view import CropView, DatasetView, DetectorView, ImageView
from .data import FileDialogFactory
from .image import ImageController


class DetectorController(Observer):

    def __init__(self, presenter: DetectorPresenter, view: DetectorView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DetectorPresenter,
                       view: DetectorView) -> DetectorController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.numberOfPixelsXSpinBox.valueChanged.connect(presenter.setNumberOfPixelsX)
        view.numberOfPixelsYSpinBox.valueChanged.connect(presenter.setNumberOfPixelsY)
        view.pixelSizeXWidget.lengthChanged.connect(presenter.setPixelSizeXInMeters)
        view.pixelSizeYWidget.lengthChanged.connect(presenter.setPixelSizeYInMeters)
        view.detectorDistanceWidget.lengthChanged.connect(presenter.setDetectorDistanceInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.numberOfPixelsXSpinBox.blockSignals(True)
        self._view.numberOfPixelsXSpinBox.setRange(self._presenter.getMinNumberOfPixelsX(),
                                                   self._presenter.getMaxNumberOfPixelsX())
        self._view.numberOfPixelsXSpinBox.setValue(self._presenter.getNumberOfPixelsX())
        self._view.numberOfPixelsXSpinBox.blockSignals(False)

        self._view.numberOfPixelsYSpinBox.blockSignals(True)
        self._view.numberOfPixelsYSpinBox.setRange(self._presenter.getMinNumberOfPixelsY(),
                                                   self._presenter.getMaxNumberOfPixelsY())
        self._view.numberOfPixelsYSpinBox.setValue(self._presenter.getNumberOfPixelsY())
        self._view.numberOfPixelsYSpinBox.blockSignals(False)

        self._view.pixelSizeXWidget.setLengthInMeters(self._presenter.getPixelSizeXInMeters())
        self._view.pixelSizeYWidget.setLengthInMeters(self._presenter.getPixelSizeYInMeters())
        self._view.detectorDistanceWidget.setLengthInMeters(
            self._presenter.getDetectorDistanceInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class DatasetListModel(QAbstractListModel):

    def __init__(self, presenter: DataFilePresenter, parent: QObject = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = Qt.NoItemFlags

        if index.isValid():
            state = self._presenter.getDatasetState(index.row())
            value = super().flags(index)

            if state != DatasetState.VALID:
                value &= ~Qt.ItemIsSelectable
                value &= ~Qt.ItemIsEnabled

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            if role == Qt.DisplayRole:
                value = self._presenter.getDatasetName(index.row())
            elif role == Qt.FontRole:
                state = self._presenter.getDatasetState(index.row())
                font = QFont()
                font.setItalic(state == DatasetState.EXISTS)
                value = QVariant(font)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.getNumberOfDatasets()


class DatasetParametersController(Observer):

    def __init__(self, dataFilePresenter: DataFilePresenter,
                 datasetPresenter: DiffractionDatasetPresenter, view: DatasetView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._dataFilePresenter = dataFilePresenter
        self._datasetPresenter = datasetPresenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._listModel = DatasetListModel(dataFilePresenter)

    @classmethod
    def createInstance(cls, dataFilePresenter: DataFilePresenter,
                       datasetPresenter: DiffractionDatasetPresenter, view: DatasetView,
                       fileDialogFactory: FileDialogFactory) -> DatasetParametersController:
        controller = cls(dataFilePresenter, datasetPresenter, view, fileDialogFactory)

        view.listView.setModel(controller._listModel)
        dataFilePresenter.addObserver(controller)
        datasetPresenter.addObserver(controller)

        view.listView.selectionModel().currentChanged.connect(
            controller._updateCurrentDatasetIndex)

        view.buttonBox.openButton.clicked.connect(controller._openDataFile)
        view.buttonBox.saveButton.clicked.connect(controller._saveDataFile)

        return controller

    def _openDataFile(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getOpenFilePath(
            self._view,
            'Open Data File',
            nameFilters=self._dataFilePresenter.getOpenFileFilterList(),
            selectedNameFilter=self._dataFilePresenter.getOpenFileFilter())

        if filePath:
            self._dataFilePresenter.openDataFile(filePath, nameFilter)

    def _saveDataFile(self) -> None:
        filePath, nameFilter = self._fileDialogFactory.getSaveFilePath(
            self._view,
            'Save Data File',
            nameFilters=self._dataFilePresenter.getSaveFileFilterList(),
            selectedNameFilter=self._dataFilePresenter.getSaveFileFilter())

        if filePath:
            self._dataFilePresenter.saveDataFile(filePath, nameFilter)

    def _chooseScratchDirectory(self) -> None:
        # TODO unused
        scratchDir = QFileDialog.getExistingDirectory(
            self._view, 'Choose Scratch Directory',
            str(self._dataFilePresenter.getScratchDirectory()))

        if scratchDir:
            self._dataFilePresenter.setScratchDirectory(Path(scratchDir))

    def _updateCurrentDatasetIndex(self, index: QModelIndex) -> None:
        self._datasetPresenter.setCurrentDatasetIndex(index.row())

    def _updateSelection(self) -> None:
        row = self._datasetPresenter.getCurrentDatasetIndex()
        index = self._listModel.index(row, 0)
        self._view.listView.setCurrentIndex(index)

    def update(self, observable: Observable) -> None:
        if observable is self._dataFilePresenter:
            self._listModel.refresh()
        elif observable is self._datasetPresenter:
            self._updateSelection()


class CropController(Observer):

    def __init__(self, presenter: CropPresenter, view: CropView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: CropPresenter, view: CropView) -> CropController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setCropEnabled)

        view.centerXSpinBox.valueChanged.connect(presenter.setCenterXInPixels)
        view.centerYSpinBox.valueChanged.connect(presenter.setCenterYInPixels)
        view.extentXSpinBox.valueChanged.connect(presenter.setExtentXInPixels)
        view.extentYSpinBox.valueChanged.connect(presenter.setExtentYInPixels)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isCropEnabled())

        self._view.centerXSpinBox.blockSignals(True)
        self._view.centerXSpinBox.setRange(self._presenter.getMinCenterXInPixels(),
                                           self._presenter.getMaxCenterXInPixels())
        self._view.centerXSpinBox.setValue(self._presenter.getCenterXInPixels())
        self._view.centerXSpinBox.blockSignals(False)

        self._view.centerYSpinBox.blockSignals(True)
        self._view.centerYSpinBox.setRange(self._presenter.getMinCenterYInPixels(),
                                           self._presenter.getMaxCenterYInPixels())
        self._view.centerYSpinBox.setValue(self._presenter.getCenterYInPixels())
        self._view.centerYSpinBox.blockSignals(False)

        self._view.extentXSpinBox.blockSignals(True)
        self._view.extentXSpinBox.setRange(self._presenter.getMinExtentXInPixels(),
                                           self._presenter.getMaxExtentXInPixels())
        self._view.extentXSpinBox.setValue(self._presenter.getExtentXInPixels())
        self._view.extentXSpinBox.blockSignals(False)

        self._view.extentYSpinBox.blockSignals(True)
        self._view.extentYSpinBox.setRange(self._presenter.getMinExtentYInPixels(),
                                           self._presenter.getMaxExtentYInPixels())
        self._view.extentYSpinBox.setValue(self._presenter.getExtentYInPixels())
        self._view.extentYSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class DatasetImageController(Observer):

    def __init__(self, datasetPresenter: DiffractionDatasetPresenter,
                 imagePresenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._datasetPresenter = datasetPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageController = ImageController.createInstance(imagePresenter, view,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, datasetPresenter: DiffractionDatasetPresenter,
                       imagePresenter: ImagePresenter, view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> DatasetImageController:
        controller = cls(datasetPresenter, imagePresenter, view, fileDialogFactory)
        datasetPresenter.addObserver(controller)
        controller._syncModelToView()
        view.imageRibbon.indexGroupBox.setTitle('Frame')
        view.imageRibbon.indexGroupBox.indexSpinBox.valueChanged.connect(
            controller._renderImageData)
        return controller

    def _renderImageData(self, index: int) -> None:
        array = self._datasetPresenter.getImage(index)
        self._imagePresenter.setArray(array)

    def _syncModelToView(self) -> None:
        numberOfImages = self._datasetPresenter.getNumberOfImages()
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setEnabled(numberOfImages > 0)
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setRange(0, numberOfImages - 1)

        index = self._view.imageRibbon.indexGroupBox.indexSpinBox.value()
        self._renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._datasetPresenter:
            self._syncModelToView()
