from PyQt5.QtWidgets import QDialog, QMessageBox, QWidget

from ...model.scan import (CartesianScanBuilder, ConcentricScanBuilder, LissajousScanBuilder,
                           ScanPointTransform, ScanRepositoryItem, SpiralScanBuilder)
from ..parametric import ParameterDialogBuilder


class ScanEditorViewControllerFactory:

    def _appendTransformControls(self, dialogBuilder: ParameterDialogBuilder,
                                 transformBuilder: ScanPointTransform) -> None:
        transformationGroup = 'Transformation'
        # FIXME dialogBuilder.add(transformBuilder.parameter, 'Parameter Name', transformationGroup)

    def createEditorDialog(self, itemName: str, item: ScanRepositoryItem,
                           parent: QWidget) -> QDialog:
        scanBuilder = item.getBuilder()
        builderName = scanBuilder.getName()
        transformBuilder = item.getTransform()
        baseScanGroup = 'Base Scan'
        title = f'{builderName}: {itemName}'

        if isinstance(scanBuilder, CartesianScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfPointsX, 'Number of Points X',
                                     baseScanGroup)
            dialogBuilder.addSpinBox(scanBuilder.numberOfPointsY, 'Number of Points Y',
                                     baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.stepSizeXInMeters, 'Step Size X',
                                          baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.stepSizeYInMeters, 'Step Size Y',
                                          baseScanGroup)
            self._appendTransformControls(dialogBuilder, transformBuilder)
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, ConcentricScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfShells, 'Number of Shells', baseScanGroup)
            dialogBuilder.addSpinBox(scanBuilder.numberOfPointsInFirstShell,
                                     'Number of Points in First Shell', baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.radialStepSizeInMeters, 'Radial Step Size',
                                          baseScanGroup)
            self._appendTransformControls(dialogBuilder, transformBuilder)
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, SpiralScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfPoints, 'Number of Points', baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.radiusScalarInMeters, 'Radius Scalar',
                                          baseScanGroup)
            self._appendTransformControls(dialogBuilder, transformBuilder)
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, LissajousScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfPoints, 'Number of Points', baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.amplitudeXInMeters, 'Amplitude X',
                                          baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.amplitudeYInMeters, 'Amplitude Y',
                                          baseScanGroup)
            dialogBuilder.addAngleWidget(scanBuilder.angularStepXInTurns, 'Angular Step X',
                                         baseScanGroup)
            dialogBuilder.addAngleWidget(scanBuilder.angularStepYInTurns, 'Angular Step Y',
                                         baseScanGroup)
            dialogBuilder.addAngleWidget(scanBuilder.angularShiftInTurns, 'Angular Shift',
                                         baseScanGroup)
            self._appendTransformControls(dialogBuilder, transformBuilder)
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
