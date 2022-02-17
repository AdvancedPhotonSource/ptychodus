from decimal import Decimal
from pathlib import Path

from PyQt5.QtCore import QAbstractListModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QFont

from ..model import *
from ..view import *

from .image import ImageController


class DetectorParametersController(Observer):
    def __init__(self, presenter: DetectorParametersPresenter, view: DetectorDetectorView) -> None:
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DetectorParametersPresenter, view: DetectorDetectorView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.pixelSizeXWidget.lengthChanged.connect(presenter.setPixelSizeXInMeters)
        view.pixelSizeYWidget.lengthChanged.connect(presenter.setPixelSizeYInMeters)
        view.detectorDistanceWidget.lengthChanged.connect(presenter.setDetectorDistanceInMeters)
        view.defocusDistanceWidget.lengthChanged.connect(presenter.setDefocusDistanceInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.pixelSizeXWidget.setLengthInMeters(
                self._presenter.getPixelSizeXInMeters())
        self._view.pixelSizeYWidget.setLengthInMeters(
                self._presenter.getPixelSizeYInMeters())
        self._view.detectorDistanceWidget.setLengthInMeters(
                self._presenter.getDetectorDistanceInMeters())
        self._view.defocusDistanceWidget.setLengthInMeters(
                self._presenter.getDefocusDistanceInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class DetectorDatasetListModel(QAbstractListModel):
    def __init__(self, presenter: DetectorDatasetPresenter, parent: QObject = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = Qt.NoItemFlags

        if index.isValid():
            state = self._presenter.getDatasetState(index.row())
            value = super().flags(index)

            if state != DatasetState.VALID:
                value &= ~Qt.ItemIsSelectable
                value &= ~Qt.ItemIsEnabled

        return value

    def data(self, index: QModelIndex, role: Qt.ItemDataRole) -> QVariant:
        value = QVariant()

        if index.isValid():
            if role == Qt.DisplayRole:
                value = self._presenter.getDatasetName(index.row())
            elif role == Qt.FontRole:
                state = self._presenter.getDatasetState(index.row())
                value = QFont()

                if state == DatasetState.FOUND:
                    value.setItalic(True)

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.getNumberOfDatasets()


class DetectorDatasetController(Observer):
    def __init__(self, datasetPresenter: DetectorDatasetPresenter,
            imagePresenter: DetectorImagePresenter, view: DetectorDatasetView) -> None:
        self._datasetPresenter = datasetPresenter
        self._imagePresenter = imagePresenter
        self._listModel = DetectorDatasetListModel(datasetPresenter)
        self._view = view

    @classmethod
    def createInstance(cls, datasetPresenter: DetectorDatasetPresenter,
            imagePresenter: DetectorImagePresenter, view: DetectorDatasetView):
        controller = cls(datasetPresenter, imagePresenter, view)

        view.dataFileListView.setModel(controller._listModel)
        datasetPresenter.addObserver(controller)
        imagePresenter.addObserver(controller)

        view.dataFileListView.selectionModel().currentChanged.connect(
                controller._updateCurrentDatasetIndex)

        return controller

    def _updateCurrentDatasetIndex(self, index: QModelIndex) -> None:
        self._imagePresenter.setCurrentDatasetIndex(index.row())

    def _updateSelection(self) -> None:
        row = self._imagePresenter.getCurrentDatasetIndex()
        index = self._listModel.index(row, 0)
        self._view.dataFileListView.setCurrentIndex(index)

    def update(self, observable: Observable) -> None:
        if observable is self._datasetPresenter:
            index = QModelIndex()
            self._listModel.dataChanged.emit(index, index)
        elif observable is self._imagePresenter:
            self._updateSelection()


class DetectorImageCropController(Observer):
    MAX_INT = 0x7FFFFFFF

    def __init__(self, presenter: DetectorParametersPresenter, view: DetectorImageCropView) -> None:
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DetectorParametersPresenter, view: DetectorImageCropView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.setCheckable(True)
        view.toggled.connect(presenter.setCropEnabled)
        view.centerXSpinBox.setRange(0, DetectorImageCropController.MAX_INT) # TODO image width (move to model)
        view.centerXSpinBox.valueChanged.connect(presenter.setCropCenterXInPixels)
        view.centerYSpinBox.setRange(0, DetectorImageCropController.MAX_INT) # TODO image height (move to model)
        view.centerYSpinBox.valueChanged.connect(presenter.setCropCenterYInPixels)
        view.extentXSpinBox.setRange(0, DetectorImageCropController.MAX_INT) # TODO image width (move to model)
        view.extentXSpinBox.valueChanged.connect(presenter.setCropExtentXInPixels)
        view.extentYSpinBox.setRange(0, DetectorImageCropController.MAX_INT) # TODO image height (move to model)
        view.extentYSpinBox.valueChanged.connect(presenter.setCropExtentYInPixels)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.setChecked(self._presenter.isCropEnabled())
        self._view.centerXSpinBox.setValue(self._presenter.getCropCenterXInPixels())
        self._view.centerYSpinBox.setValue(self._presenter.getCropCenterYInPixels())
        self._view.extentXSpinBox.setValue(self._presenter.getCropExtentXInPixels())
        self._view.extentYSpinBox.setValue(self._presenter.getCropExtentYInPixels())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class DetectorImageController(Observer):
    def __init__(self, presenter: DetectorImagePresenter, view: ImageView) -> None:
       self._presenter = presenter
       self._view = view
       self._image_controller = ImageController.createInstance(view)

    @classmethod
    def createInstance(cls, presenter: DetectorImagePresenter, view: ImageView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        controller.updateView()
        view.imageRibbon.imageSpinBox.valueChanged.connect(controller.renderImageData)

        return controller

    def renderImageData(self, index: int) -> None:
        image = self._presenter.getImage(index)
        self._image_controller.renderImageData(image)

    def updateView(self) -> None:
        numberOfImages = self._presenter.getNumberOfImages()
        self._view.imageRibbon.imageSpinBox.setEnabled(numberOfImages > 0)
        self._view.imageRibbon.imageSpinBox.setRange(0, numberOfImages - 1)

        index = self._view.imageRibbon.imageSpinBox.value()
        self.renderImageData(index)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self.updateView()

