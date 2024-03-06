from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from ...model.product.object import ObjectRepositoryItem, RandomObjectBuilder
from ..parametric import ParameterDialogBuilder


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
            dialogBuilder.addSpinBox(objectBuilder.numberOfLayers, 'Number of Layers:',
                                     additionalLayersGroup)
            dialogBuilder.addLengthWidget(objectBuilder.layerDistanceInMeters, 'Layer Distance:',
                                          additionalLayersGroup)
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
