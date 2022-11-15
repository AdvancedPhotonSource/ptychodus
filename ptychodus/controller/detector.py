from __future__ import annotations
from decimal import Decimal
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFileDialog

from ..api.data import DiffractionPatternState
from ..api.observer import Observable, Observer
from ..model import (DetectorPresenter, DiffractionDatasetPresenter,
                     ActiveDiffractionPatternPresenter, ImagePresenter)
from ..view import DiffractionPatternView, DetectorView, ImageView
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

    def __init__(self,
                 presenter: DiffractionDatasetPresenter,
                 parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def refresh(self) -> None:
        self.beginResetModel()
        self.endResetModel()

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = Qt.ItemFlags()

        if index.isValid():
            state = self._presenter.getPatternState(index.row())
            value = super().flags(index)

            if state != DiffractionPatternState.LOADED:
                value &= ~Qt.ItemIsSelectable
                value &= ~Qt.ItemIsEnabled

        return value

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            if role == Qt.DisplayRole:
                label = self._presenter.getArrayLabel(index.row())
                value = QVariant(label)
            elif role == Qt.FontRole:
                state = self._presenter.getPatternState(index.row())
                font = QFont()
                font.setItalic(state == DiffractionPatternState.FOUND)
                value = QVariant(font)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.getNumberOfArrays()


class DatasetParametersController(Observer):

    def __init__(self, datasetPresenter: DiffractionDatasetPresenter,
                 activePatternPresenter: ActiveDiffractionPatternPresenter,
                 view: DiffractionPatternView) -> None:
        super().__init__()
        self._datasetPresenter = datasetPresenter
        self._activePatternPresenter = activePatternPresenter
        self._view = view
        self._listModel = DatasetListModel(datasetPresenter)

    @classmethod
    def createInstance(cls, datasetPresenter: DiffractionDatasetPresenter,
                       activePatternPresenter: ActiveDiffractionPatternPresenter,
                       view: DiffractionPatternView) -> DatasetParametersController:
        controller = cls(datasetPresenter, activePatternPresenter, view)

        view.listView.setModel(controller._listModel)
        datasetPresenter.addObserver(controller)
        activePatternPresenter.addObserver(controller)

        view.listView.selectionModel().currentChanged.connect(
            controller._updateCurrentPatternIndex)

        return controller

    def _updateCurrentPatternIndex(self, index: QModelIndex) -> None:
        self._activePatternPresenter.setCurrentPatternIndex(index.row())

    def _updateSelection(self) -> None:
        row = self._activePatternPresenter.getCurrentPatternIndex()
        index = self._listModel.index(row, 0)
        self._view.listView.setCurrentIndex(index)

    def update(self, observable: Observable) -> None:
        if observable is self._datasetPresenter:
            self._listModel.refresh()
        elif observable is self._activePatternPresenter:
            self._updateSelection()


class DatasetImageController(Observer):

    def __init__(self, activePatternPresenter: ActiveDiffractionPatternPresenter,
                 imagePresenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        super().__init__()
        self._activePatternPresenter = activePatternPresenter
        self._imagePresenter = imagePresenter
        self._view = view
        self._imageController = ImageController.createInstance(imagePresenter, view,
                                                               fileDialogFactory)

    @classmethod
    def createInstance(cls, activePatternPresenter: ActiveDiffractionPatternPresenter,
                       imagePresenter: ImagePresenter, view: ImageView,
                       fileDialogFactory: FileDialogFactory) -> DatasetImageController:
        controller = cls(activePatternPresenter, imagePresenter, view, fileDialogFactory)
        activePatternPresenter.addObserver(controller)
        controller._syncModelToView()
        view.imageRibbon.indexGroupBox.setTitle('Frame')
        view.imageRibbon.indexGroupBox.indexSpinBox.valueChanged.connect(
            controller._renderImageData)
        return controller

    def _renderImageData(self, index: int) -> None:
        array = self._activePatternPresenter.getImage(index)
        self._imagePresenter.setArray(array)

    def _syncModelToView(self) -> None:
        numberOfImages = self._activePatternPresenter.getNumberOfImages()
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setEnabled(numberOfImages > 0)
        self._view.imageRibbon.indexGroupBox.indexSpinBox.setRange(0, numberOfImages - 1)

        index = self._view.imageRibbon.indexGroupBox.indexSpinBox.value()
        self._renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._activePatternPresenter:
            self._syncModelToView()
