from PyQt5.QtGui import QDoubleValidator, QImage, QPixmap, QStandardItem, QStandardItemModel

import matplotlib
import numpy

from ..model import ColorMapListFactory, ScalarTransformation, ComplexToRealStrategy
from ..view import ImageView
from .data import FileDialogFactory


class ImageController:
    MIME_TYPES = ['image/bmp', 'image/jpeg', 'image/png', 'image/x-portable-pixmap']

    def __init__(self, view: ImageView, fileDialogFactory: FileDialogFactory) -> None:
        self._view = view
        self._fileDialogFactory = fileDialogFactory
        self._vmin = 0.
        self._vmax = 1.
        self._image_data = None
        self._acyclicColorMapModel = QStandardItemModel()
        self._cyclicColorMapModel = QStandardItemModel()

    @classmethod
    def createInstance(cls, view: ImageView, fileDialogFactory: FileDialogFactory):
        controller = cls(view, fileDialogFactory)

        view.imageRibbon.saveButton.clicked.connect(controller._saveImage)

        zlf = lambda x: numpy.zeros_like(x, dtype=float)

        view.imageRibbon.scalarTransformComboBox.addItem('Identity',
                                                         ScalarTransformation(lambda x: x))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Square Root',
            ScalarTransformation(lambda x: numpy.sqrt(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Logarithm (Base 2)',
            ScalarTransformation(lambda x: numpy.log2(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Natural Logarithm',
            ScalarTransformation(lambda x: numpy.log(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.addItem(
            'Logarithm (Base 10)',
            ScalarTransformation(lambda x: numpy.log10(x, out=zlf(x), where=(x > 0))))
        view.imageRibbon.scalarTransformComboBox.currentTextChanged.connect(
            lambda text: controller._renderCachedImageData())

        view.imageRibbon.complexComponentComboBox.addItem(
            'Magnitude', ComplexToRealStrategy(numpy.absolute, False))
        view.imageRibbon.complexComponentComboBox.addItem('Phase',
                                                          ComplexToRealStrategy(numpy.angle, True))
        view.imageRibbon.complexComponentComboBox.addItem('Real Part',
                                                          ComplexToRealStrategy(numpy.real, False))
        view.imageRibbon.complexComponentComboBox.addItem('Imaginary Part',
                                                          ComplexToRealStrategy(numpy.imag, False))
        view.imageRibbon.complexComponentComboBox.currentTextChanged.connect(
            lambda text: controller._renderCachedImageData())

        view.imageRibbon.vminLineEdit.setValidator(QDoubleValidator())
        view.imageRibbon.vminLineEdit.editingFinished.connect(
            controller._handleVminEditingFinished)
        view.imageRibbon.vminAutoCheckBox.stateChanged.connect(controller._handleVminAutoToggled)
        controller._updateVminLineEditText()
        controller._updateVminLineEditEnabled()

        view.imageRibbon.vmaxLineEdit.setValidator(QDoubleValidator())
        view.imageRibbon.vmaxLineEdit.editingFinished.connect(
            controller._handleVmaxEditingFinished)
        view.imageRibbon.vmaxAutoCheckBox.stateChanged.connect(controller._handleVmaxAutoToggled)
        controller._updateVmaxLineEditText()
        controller._updateVmaxLineEditEnabled()

        colorMapListFactory = ColorMapListFactory()

        for colorMap in colorMapListFactory.createAcyclicColorMapList():
            row = QStandardItem(colorMap)
            controller._acyclicColorMapModel.appendRow(row)

        for colorMap in colorMapListFactory.createCyclicColorMapList():
            row = QStandardItem(colorMap)
            controller._cyclicColorMapModel.appendRow(row)

        view.imageRibbon.colorMapComboBox.setModel(controller._acyclicColorMapModel)
        view.imageRibbon.colorMapComboBox.currentTextChanged.connect(
            controller._handleColorMapTextChanged)

        return controller

    def _saveImage(self) -> None:
        filePath, _ = self._fileDialogFactory.getSaveFilePath(
            self._view, 'Save Image', mimeTypeFilters=ImageController.MIME_TYPES)

        if filePath:
            pixmap = self._view.imageWidget.getPixmap()
            pixmap.save(str(filePath))

    def _renderCachedImageData(self) -> None:
        image_data = self._image_data

        if image_data is None:
            return

        currentComplexComponentStrategy = self._view.imageRibbon.complexComponentComboBox.currentData(
        )

        if numpy.iscomplexobj(image_data):
            self._view.imageRibbon.complexComponentComboBox.setVisible(True)
            self._view.imageRibbon.colorMapComboBox.setModel(
                self._cyclicColorMapModel if currentComplexComponentStrategy.isCyclic else self.
                _acyclicColorMapModel)
            image_data = currentComplexComponentStrategy.complexToRealFunction(image_data)
        else:
            self._view.imageRibbon.complexComponentComboBox.setVisible(False)
            self._view.imageRibbon.colorMapComboBox.setModel(self._acyclicColorMapModel)

        currentScalarTransformStrategy = self._view.imageRibbon.scalarTransformComboBox.currentData(
        )
        image_data = currentScalarTransformStrategy.transformFunction(image_data)

        if self._view.imageRibbon.vminAutoCheckBox.isChecked():
            self._vmin = image_data.min()
            self._updateVminLineEditText()

        if self._view.imageRibbon.vmaxAutoCheckBox.isChecked():
            self._vmax = image_data.max()
            self._updateVmaxLineEditText()

        cnorm = matplotlib.colors.Normalize(vmin=self._vmin, vmax=self._vmax, clip=False)
        cmap = matplotlib.cm.get_cmap(self._view.imageRibbon.colorMapComboBox.currentText())
        scalarMappable = matplotlib.cm.ScalarMappable(norm=cnorm, cmap=cmap)

        color_image = scalarMappable.to_rgba(image_data)
        color_image = numpy.multiply(color_image, 255).astype(numpy.uint8)

        qimage = QImage(color_image.data, color_image.shape[1], color_image.shape[0],
                        color_image.strides[0], QImage.Format_RGBA8888)
        qpixmap = QPixmap.fromImage(qimage)
        self._view.imageWidget.setPixmap(qpixmap)

    def renderImageData(self, image_data: numpy.ndarray) -> None:
        self._image_data = image_data
        self._renderCachedImageData()

    def _updateVminLineEditText(self) -> None:
        self._view.imageRibbon.vminLineEdit.setText(f'{self._vmin:.4f}')

    def _updateVminLineEditEnabled(self) -> None:
        self._view.imageRibbon.vminLineEdit.setEnabled(
            not self._view.imageRibbon.vminAutoCheckBox.isChecked())

    def _handleVminEditingFinished(self) -> None:
        self._vmin = float(self._view.imageRibbon.vminLineEdit.text())
        self._renderCachedImageData()

    def _handleVminAutoToggled(self, state: int) -> None:
        self._updateVminLineEditEnabled()
        self._renderCachedImageData()

    def _updateVmaxLineEditText(self) -> None:
        self._view.imageRibbon.vmaxLineEdit.setText(f'{self._vmax:.4f}')

    def _updateVmaxLineEditEnabled(self) -> None:
        self._view.imageRibbon.vmaxLineEdit.setEnabled(
            not self._view.imageRibbon.vmaxAutoCheckBox.isChecked())

    def _handleVmaxEditingFinished(self) -> None:
        self._vmax = float(self._view.imageRibbon.vmaxLineEdit.text())
        self._renderCachedImageData()

    def _handleVmaxAutoToggled(self, state: int) -> None:
        self._updateVmaxLineEditEnabled()
        self._renderCachedImageData()

    def _handleColorMapTextChanged(self, text: str) -> None:
        self._renderCachedImageData()
