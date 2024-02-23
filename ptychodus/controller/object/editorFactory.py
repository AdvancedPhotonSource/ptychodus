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
            dialogBuilder.addSpinBox(objectBuilder.numberOfLayers, 'Number of Layers')
            dialogBuilder.addLengthWidget(objectBuilder.layerDistanceInMeters, 'Layer Distance')
            dialogBuilder.addSpinBox(objectBuilder.extraPaddingX, 'Extra Padding X')
            dialogBuilder.addSpinBox(objectBuilder.extraPaddingY, 'Extra Padding Y')
            dialogBuilder.addDecimalSlider(objectBuilder.amplitudeMean, 'Amplitude Mean')
            dialogBuilder.addDecimalSlider(objectBuilder.amplitudeDeviation, 'Amplitude Deviation')
            dialogBuilder.addDecimalSlider(objectBuilder.phaseDeviation, 'Phase Deviation')
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
