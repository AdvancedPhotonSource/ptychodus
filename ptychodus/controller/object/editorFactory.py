from PyQt5.QtWidgets import QDialog, QMessageBox, QSpinBox, QWidget

from ptychodus.api.observer import Observable, Observer

from ...model.product.object import ObjectRepositoryItem, RandomObjectBuilder
from ..parametric import ParameterDialogBuilder, ParameterViewController


class MultisliceViewController(ParameterViewController, Observer):

    def __init__(self, item: ObjectRepositoryItem) -> None:
        super().__init__()
        self._item = item
        self._parameter = item.layerDistanceInMeters
        self._widget = QSpinBox()

        self._syncModelToView()
        self._widget.valueChanged.connect(self._syncViewToModel)
        self._parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, numberOfLayers: int) -> None:
        self._item.setNumberOfLayers(numberOfLayers)

    def _syncModelToView(self) -> None:
        self._widget.blockSignals(True)
        self._widget.setRange(1, 99)
        self._widget.setValue(self._item.getNumberOfLayers())
        self._widget.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class ObjectEditorViewControllerFactory:

    def createEditorDialog(self, itemName: str, item: ObjectRepositoryItem,
                           parent: QWidget) -> QDialog:
        objectBuilder = item.getBuilder()
        builderName = objectBuilder.getName()
        firstLayerGroup = 'First Layer'
        additionalLayersGroup = 'Additional Layers'
        title = f'{itemName} [{builderName}]'

        if isinstance(objectBuilder, RandomObjectBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(objectBuilder.extraPaddingX, 'Extra Padding X:',
                                     firstLayerGroup)
            dialogBuilder.addSpinBox(objectBuilder.extraPaddingY, 'Extra Padding Y:',
                                     firstLayerGroup)
            dialogBuilder.addDecimalSlider(objectBuilder.amplitudeMean, 'Amplitude Mean:',
                                           firstLayerGroup)
            dialogBuilder.addDecimalSlider(objectBuilder.amplitudeDeviation,
                                           'Amplitude Deviation:', firstLayerGroup)
            dialogBuilder.addDecimalSlider(objectBuilder.phaseDeviation, 'Phase Deviation:',
                                           firstLayerGroup)
            dialogBuilder.addViewController(MultisliceViewController(item), 'Number of Layers:',
                                            additionalLayersGroup)
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
