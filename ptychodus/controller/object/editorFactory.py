from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from ...model.object import ObjectRepositoryItem, RandomObjectBuilder
from ..parametric import ParameterDialogBuilder


class ObjectEditorViewControllerFactory:

    def createEditorDialog(self, itemName: str, item: ObjectRepositoryItem,
                           parent: QWidget) -> QDialog:
        objectBuilder = item.getBuilder()
        builderName = objectBuilder.getName()
        title = f'{builderName}: {itemName}'

        if isinstance(objectBuilder, RandomObjectBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox('Number of Layers', objectBuilder.numberOfLayers)
            dialogBuilder.addLengthWidget('Layer Distance', objectBuilder.layerDistanceInMeters)
            dialogBuilder.addSpinBox('Extra Padding X', objectBuilder.extraPaddingX)
            dialogBuilder.addSpinBox('Extra Padding Y', objectBuilder.extraPaddingY)
            dialogBuilder.addDecimalSlider('Amplitude Mean', objectBuilder.amplitudeMean)
            dialogBuilder.addDecimalSlider('Amplitude Deviation', objectBuilder.amplitudeDeviation)
            dialogBuilder.addDecimalSlider('Phase Deviation', objectBuilder.phaseDeviation)
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
