from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from ...model.probe import ProbeRepositoryItem, RandomProbeBuilder
from ..parametric import ParameterDialogBuilder


class ProbeEditorViewControllerFactory:

    def createEditorDialog(self, itemName: str, item: ProbeRepositoryItem,
                           parent: QWidget) -> QDialog:
        probeBuilder = item.getBuilder()
        builderName = probeBuilder.getName()
        title = f'{builderName}: {itemName}'

        if isinstance(probeBuilder, RandomProbeBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox('Number of Layers', probeBuilder.numberOfLayers)
            dialogBuilder.addLengthWidget('Layer Distance', probeBuilder.layerDistanceInMeters)
            dialogBuilder.addSpinBox('Extra Padding X', probeBuilder.extraPaddingX)
            dialogBuilder.addSpinBox('Extra Padding Y', probeBuilder.extraPaddingY)
            dialogBuilder.addDecimalSlider('Amplitude Mean', probeBuilder.amplitudeMean)
            dialogBuilder.addDecimalSlider('Amplitude Deviation', probeBuilder.amplitudeDeviation)
            dialogBuilder.addDecimalSlider('Phase Deviation', probeBuilder.phaseDeviation)
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
