from decimal import Decimal
from pathlib import Path

from PyQt5.QtCore import QAbstractListModel, QModelIndex, QObject, QVariant
from PyQt5.QtGui import QFont

from ..model import *
from ..view import *

from .image import ImageController


class DetectorController(Observer):
    def __init__(self, presenter: DetectorPresenter, view: DetectorView) -> None:
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: DetectorPresenter, view: DetectorView):
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.pixelSizeXWidget.lengthChanged.connect(presenter.setPixelSizeXInMeters)
        view.pixelSizeYWidget.lengthChanged.connect(presenter.setPixelSizeYInMeters)
        view.detectorDistanceWidget.lengthChanged.connect(presenter.setDetectorDistanceInMeters)
        view.defocusDistanceWidget.lengthChanged.connect(presenter.setDefocusDistanceInMeters)

        controller._syncModelToView()

        return controller

    def _syncModelToView(self) -> None:
        self._view.pixelSizeXWidget.setLengthInMeters(self._presenter.getPixelSizeXInMeters())
        self._view.pixelSizeYWidget.setLengthInMeters(self._presenter.getPixelSizeYInMeters())
        self._view.detectorDistanceWidget.setLengthInMeters(
            self._presenter.getDetectorDistanceInMeters())
        self._view.defocusDistanceWidget.setLengthInMeters(
            self._presenter.getDefocusDistanceInMeters())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class DatasetListModel(QAbstractListModel):
    def __init__(self, presenter: DatasetPresenter, parent: QObject = None) -> None:
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

    def reload(self) -> None:
        self.beginResetModel()
        self.endResetModel()


class DatasetController(Observer):
    def __init__(self, datasetPresenter: DatasetPresenter,
                 imagePresenter: DetectorImagePresenter, view: DatasetView) -> None:
        self._datasetPresenter = datasetPresenter
        self._imagePresenter = imagePresenter
        self._listModel = DatasetListModel(datasetPresenter)
        self._view = view

    @classmethod
    def createInstance(cls, datasetPresenter: DatasetPresenter,
                       imagePresenter: DetectorImagePresenter, view: DatasetView):
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
            self._listModel.reload()
        elif observable is self._imagePresenter:
            self._updateSelection()


class CropController(Observer):
    def __init__(self, presenter: CropPresenter, view: CropView) -> None:
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(cls, presenter: CropPresenter, view: CropView):
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
