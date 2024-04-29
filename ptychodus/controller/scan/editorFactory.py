from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QFormLayout, QGridLayout, QGroupBox, QLabel, QMessageBox,
                             QWidget)

from ptychodus.api.observer import Observable, Observer

from ...model.product.scan import (CartesianScanBuilder, ConcentricScanBuilder,
                                   FromFileScanBuilder, FromMemoryScanBuilder,
                                   LissajousScanBuilder, ScanPointTransform, ScanRepositoryItem,
                                   SpiralScanBuilder)
from ..parametric import (DecimalLineEditParameterViewController,
                          LengthWidgetParameterViewController, ParameterDialogBuilder,
                          ParameterViewController)
from ...view.widgets import GroupBoxWithPresets

__all__ = [
    'ScanEditorViewControllerFactory',
]


class ScanTransformViewController(ParameterViewController):

    def __init__(self, transform: ScanPointTransform) -> None:
        super().__init__()
        self._widget = GroupBoxWithPresets('Transformation')

        for index, presetsLabel in enumerate(transform.labelsForPresets()):
            action = self._widget.presetsMenu.addAction(presetsLabel)
            action.triggered.connect(lambda _, index=index: transform.applyPresets(index))

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
        self._widget.contents.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class ScanBoundingBoxViewController(ParameterViewController, Observer):

    def __init__(self, item: ScanRepositoryItem) -> None:
        super().__init__()
        self._parameter = item.expandBoundingBox
        self._widget = QGroupBox('Expand Bounding Box')
        self._widget.setCheckable(True)

        self._minimumXController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMinimumXInMeters, isSigned=True)
        self._maximumXController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMaximumXInMeters, isSigned=True)
        self._minimumYController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMinimumYInMeters, isSigned=True)
        self._maximumYController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMaximumYInMeters, isSigned=True)

        layout = QFormLayout()
        layout.addRow('Minimum X:', self._minimumXController.getWidget())
        layout.addRow('Maximum X:', self._maximumXController.getWidget())
        layout.addRow('Minimum Y:', self._minimumYController.getWidget())
        layout.addRow('Maximum Y:', self._maximumYController.getWidget())
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._widget.toggled.connect(self._parameter.setValue)
        self._parameter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._widget.setChecked(self._parameter.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self._syncModelToView()


class ScanEditorViewControllerFactory:

    def _appendCommonControls(self, dialogBuilder: ParameterDialogBuilder,
                              item: ScanRepositoryItem) -> None:
        dialogBuilder.addViewControllerToBottom(ScanTransformViewController(item.getTransform()))
        dialogBuilder.addViewControllerToBottom(ScanBoundingBoxViewController(item))

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

            if not scanBuilder.isEquilateral:
                dialogBuilder.addLengthWidget(scanBuilder.stepSizeYInMeters, 'Step Size Y:',
                                              baseScanGroup)

            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, ConcentricScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfShells, 'Number of Shells:',
                                     baseScanGroup)
            dialogBuilder.addSpinBox(scanBuilder.numberOfPointsInFirstShell,
                                     'Number of Points in First Shell:', baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.radialStepSizeInMeters, 'Radial Step Size:',
                                          baseScanGroup)
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, FromFileScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, FromMemoryScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build(title, parent)
        elif isinstance(scanBuilder, SpiralScanBuilder):
            dialogBuilder = ParameterDialogBuilder()
            dialogBuilder.addSpinBox(scanBuilder.numberOfPoints, 'Number of Points:',
                                     baseScanGroup)
            dialogBuilder.addLengthWidget(scanBuilder.radiusScalarInMeters, 'Radius Scalar:',
                                          baseScanGroup)
            self._appendCommonControls(dialogBuilder, item)
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
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build(title, parent)

        return QMessageBox(QMessageBox.Icon.Information, title,
                           f'\"{builderName}\" has no editable parameters!', QMessageBox.Ok,
                           parent)
