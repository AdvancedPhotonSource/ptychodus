from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog, QGridLayout, QLabel, QMessageBox, QWidget

from ...model.scan import (CartesianScanBuilder, ConcentricScanBuilder, LissajousScanBuilder,
                           ScanPointTransform, ScanRepositoryItem, SpiralScanBuilder)
from ..parametric import (DecimalLineEditParameterViewController,
                          LengthWidgetParameterViewController, ParameterDialogBuilder,
                          ParameterViewController)


class ScanTransformViewController(ParameterViewController):

    def __init__(self, transform: ScanPointTransform) -> None:
        super().__init__()
        self._widget = QWidget()

        self._labelXP = QLabel('x\u2032 =')
        self._labelXP.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._affineAXViewController = DecimalLineEditParameterViewController(transform.affineAX,
                                                                              isSigned=True)
        self._labelAX = QLabel('x +')
        self._affineAYViewController = DecimalLineEditParameterViewController(transform.affineAY,
                                                                              isSigned=True)
        self._labelAY = QLabel('y +')
        self._affineATViewController = LengthWidgetParameterViewController(transform.affineAT,
                                                                           isSigned=True)

        self._labelYP = QLabel('y\u2032 =')
        self._labelYP.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._affineBXViewController = DecimalLineEditParameterViewController(transform.affineBX,
                                                                              isSigned=True)
        self._labelBX = QLabel('x +')
        self._affineBYViewController = DecimalLineEditParameterViewController(transform.affineBY,
                                                                              isSigned=True)
        self._labelBY = QLabel('y +')
        self._affineBTViewController = LengthWidgetParameterViewController(transform.affineBT,
                                                                           isSigned=True)

        self._jitterRadiusLabel = QLabel('Jitter Radius:')
        self._jitterRadiusViewController = LengthWidgetParameterViewController(
            transform.jitterRadiusInMeters, isSigned=False)

        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._labelXP, 0, 0)
        layout.addWidget(self._affineAXViewController.getWidget(), 0, 1)
        layout.addWidget(self._labelAX, 0, 2)
        layout.addWidget(self._affineAYViewController.getWidget(), 0, 3)
        layout.addWidget(self._labelAY, 0, 4)
        layout.addWidget(self._affineATViewController.getWidget(), 0, 5)

        layout.addWidget(self._labelYP, 1, 0)
        layout.addWidget(self._affineBXViewController.getWidget(), 1, 1)
        layout.addWidget(self._labelBX, 1, 2)
        layout.addWidget(self._affineBYViewController.getWidget(), 1, 3)
        layout.addWidget(self._labelBY, 1, 4)
        layout.addWidget(self._affineBTViewController.getWidget(), 1, 5)

        layout.addWidget(self._jitterRadiusLabel, 2, 0)
        layout.addWidget(self._jitterRadiusViewController.getWidget(), 2, 1, 1, 5)
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class ScanEditorViewControllerFactory:

    def _appendTransformControls(self, dialogBuilder: ParameterDialogBuilder,
                                 transform: ScanPointTransform) -> None:
        # FIXME load transform presets (button menu?)
        transformationGroup = 'Transformation'
        dialogBuilder.addViewController(
            ScanTransformViewController(transform),
            '_Transform',
            transformationGroup,
        )

    def createEditorDialog(self, itemName: str, item: ScanRepositoryItem,
                           parent: QWidget) -> QDialog:
        scanBuilder = item.getBuilder()
        builderName = scanBuilder.getName()
        baseScanGroup = 'Base Scan'
        title = f'{itemName} [{builderName}]'

        if isinstance(scanBuilder, CartesianScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfPointsX, 'Number of Points X:',
                                     baseScanGroup)
            dialogBuilder.addSpinBox(scanBuilder.numberOfPointsY, 'Number of Points Y:',
                                     baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.stepSizeXInMeters, 'Step Size X:',
                                          baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.stepSizeYInMeters, 'Step Size Y:',
                                          baseScanGroup)
            self._appendTransformControls(dialogBuilder, item.getTransform())
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, ConcentricScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfShells, 'Number of Shells:',
                                     baseScanGroup)
            dialogBuilder.addSpinBox(scanBuilder.numberOfPointsInFirstShell,
                                     'Number of Points in First Shell:', baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.radialStepSizeInMeters, 'Radial Step Size:',
                                          baseScanGroup)
            self._appendTransformControls(dialogBuilder, item.getTransform())
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, SpiralScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfPoints, 'Number of Points:',
                                     baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.radiusScalarInMeters, 'Radius Scalar:',
                                          baseScanGroup)
            self._appendTransformControls(dialogBuilder, item.getTransform())
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, LissajousScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfPoints, 'Number of Points:',
                                     baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.amplitudeXInMeters, 'Amplitude X:',
                                          baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.amplitudeYInMeters, 'Amplitude Y:',
                                          baseScanGroup)
            dialogBuilder.addAngleWidget(scanBuilder.angularStepXInTurns, 'Angular Step X:',
                                         baseScanGroup)
            dialogBuilder.addAngleWidget(scanBuilder.angularStepYInTurns, 'Angular Step Y:',
                                         baseScanGroup)
            dialogBuilder.addAngleWidget(scanBuilder.angularShiftInTurns, 'Angular Shift:',
                                         baseScanGroup)
            self._appendTransformControls(dialogBuilder, item.getTransform())
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
