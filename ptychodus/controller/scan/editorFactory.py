from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from ...model.scan import ScanRepositoryItem
from ..parametric import ParameterDialogBuilder


class ScanEditorViewControllerFactory:

    def createEditorDialog(self, itemName: str, item: ScanRepositoryItem,
                           parent: QWidget) -> QDialog:
        scanBuilder = item.getBuilder()
        builderName = scanBuilder.getName()
        title = f'{builderName}: {itemName}'

        # FIXME if isinstance(scanBuilder, RandomScanBuilder):
        # FIXME     dialogBuilder = ParameterDialogBuilder()
        # FIXME     dialogBuilder.addSpinBox(scanBuilder.numberOfLayers, 'Number of Layers')
        # FIXME     dialogBuilder.addLengthWidget(scanBuilder.layerDistanceInMeters, 'Layer Distance')
        # FIXME     dialogBuilder.addSpinBox(scanBuilder.extraPaddingX, 'Extra Padding X')
        # FIXME     dialogBuilder.addSpinBox(scanBuilder.extraPaddingY, 'Extra Padding Y')
        # FIXME     dialogBuilder.addDecimalSlider(scanBuilder.amplitudeMean, 'Amplitude Mean')
        # FIXME     dialogBuilder.addDecimalSlider(scanBuilder.amplitudeDeviation, 'Amplitude Deviation')
        # FIXME     dialogBuilder.addDecimalSlider(scanBuilder.phaseDeviation, 'Phase Deviation')
        # FIXME     return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
