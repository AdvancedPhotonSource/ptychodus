from PyQt5.QtGui import QDoubleValidator, QImage, QPixmap, QStandardItem, QStandardItemModel

import matplotlib
import numpy

from ..api.image import ScalarTransformation, ComplexToRealStrategy
from ..api.observer import Observable, Observer
from ..model import ImagePresenter
from ..view import ImageView
from .data import FileDialogFactory


class ImageController(Observer):
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, presenter: ImagePresenter, view: ImageView,
                 fileDialogFactory: FileDialogFactory) -> None:
        self._presenter = presenter
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._scalarTransformationModel = QStringListModel()
        self._complexToRealStrategyModel = QStringListModel()
        self._colormapModel = QStringListModel()

    @classmethod
    def createInstance(cls, presenter: ImagePresenter, view: ImageView,
                       fileDialogFactory: FileDialogFactory):
        controller = cls(presenter, view, fileDialogFactory)
        controller._syncModelToView()
        presenter.addObserver(controller)

        view.imageRibbon.saveButton.clicked.connect(controller._saveImage)

        view.imageRibbon.scalarTransformComboBox.setModel(controller._scalarTransformationModel)
        view.imageRibbon.scalarTransformComboBox.currentTextChanged.connect(
            presenter.setScalarTransformation)

        view.imageRibbon.complexComponentComboBox.setModel(controller._complexToRealStrategyModel)
        view.imageRibbon.complexComponentComboBox.currentTextChanged.connect(
            presenter.setComplexToRealStrategy)

        view.imageRibbon.colormapComboBox.setModel(controller._colormapModel)
        view.imageRibbon.colormapComboBox.currentTextChanged.connect(presenter.setColormap)

        view.imageRibbon.vminLineEdit.setValidator(QDoubleValidator())
        view.imageRibbon.vminLineEdit.editingFinished.connect(controller._syncVMinToModel)
        view.imageRibbon.vminAutoCheckBox.toggled.connect(presenter.setAutomaticVMinEnabled)

        view.imageRibbon.vmaxLineEdit.setValidator(QDoubleValidator())
        view.imageRibbon.vmaxLineEdit.editingFinished.connect(controller._syncVMaxToModel)
        view.imageRibbon.vmaxAutoCheckBox.toggled.connect(presenter.setAutomaticVMaxEnabled)

        return controller

    def _saveImage(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Image', mimeTypeFilters=ImageController.MIME_TYPES)

        if filePath:
            pixmap = self._view.imageWidget.getPixmap()
            pixmap.save(str(filePath))

    def _syncVMinToModel(self) -> None:
        self._presenter.setVMinValue(Decimal(self._view.imageRibbon.vminLineEdit.text()))

    def _syncVMaxToModel(self) -> None:
        self._presenter.setVMaxValue(Decimal(self._view.imageRibbon.vmaxLineEdit.text()))

    def _syncModelToView(self) -> None:
        self._scalarTransformationModel.setStringList(
            self._presenter.getScalarTransformationList())
        self._view.imageRibbon.scalarTransformComboBox.setCurrentText(
            self._presenter.getScalarTransformation())

        self._complexToRealStrategyModel.setStringList(
            self._presenter.getComplexToRealStrategyList())
        self._view.imageRibbon.complexComponentComboBox.setCurrentText(
            self._presenter.getComplexToRealStrategy())
        self._view.imageRibbon.complexComponentComboBox.setVisible(True)  # FIXME iscomplexobj

        self._colormapModel.setStringList(self._presenter.getColormapList())
        self._view.imageRibbon.colormapComboBox.setCurrentText(self._presenter.getColormap())

        self._view.imageRibbon.vmaxAutoCheckBox.setChecked(
            self._presenter.isAutomaticVMinEnabled())
        self._view.imageRibbon.vminLineEdit.setEnabled(
            not self._presenter.isAutomaticVMinEnabled())
        self._view.imageRibbon.vminLineEdit.setText(str(self._presenter.getVMinValue()))

        self._view.imageRibbon.vmaxAutoCheckBox.setChecked(
            self._presenter.isAutomaticVMaxEnabled())
        self._view.imageRibbon.vmaxLineEdit.setEnabled(
            not self._presenter.isAutomaticVMaxEnabled())
        self._view.imageRibbon.vmaxLineEdit.setText(str(self._presenter.getVMaxValue()))

        self._renderCachedImageData()

    def _renderCachedImageData(self) -> None:
        realImage = self._presenter.getImage()

        if realImage is None:
            return

        integerImage = numpy.multiply(realImage, 255).astype(numpy.uint8)

        qimage = QImage(integerImage.data, integerImage.shape[1], integerImage.shape[0],
                        integerImage.strides[0], QImage.Format_RGBA8888)
        qpixmap = QPixmap.fromImage(qimage)
        self._view.imageWidget.setPixmap(qpixmap)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()
