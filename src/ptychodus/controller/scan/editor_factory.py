from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QGridLayout,
    QLabel,
    QMessageBox,
    QWidget,
)


from ...model.product.scan import (
    CartesianScanBuilder,
    ConcentricScanBuilder,
    FromFileScanBuilder,
    FromMemoryScanBuilder,
    LissajousScanBuilder,
    ScanPointTransform,
    ScanRepositoryItem,
    SpiralScanBuilder,
)
from ..parametric import (
    CheckableGroupBoxParameterViewController,
    DecimalLineEditParameterViewController,
    LengthWidgetParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
)
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
        self._affineAXViewController = DecimalLineEditParameterViewController(
            transform.affineAX, is_signed=True
        )
        self._labelAX = QLabel('x +')
        self._affineAYViewController = DecimalLineEditParameterViewController(
            transform.affineAY, is_signed=True
        )
        self._labelAY = QLabel('y +')
        self._affineATViewController = LengthWidgetParameterViewController(
            transform.affineATInMeters, is_signed=True
        )

        self._labelYP = QLabel('y\u2032 =')
        self._labelYP.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._affineBXViewController = DecimalLineEditParameterViewController(
            transform.affineBX, is_signed=True
        )
        self._labelBX = QLabel('x +')
        self._affineBYViewController = DecimalLineEditParameterViewController(
            transform.affineBY, is_signed=True
        )
        self._labelBY = QLabel('y +')
        self._affineBTViewController = LengthWidgetParameterViewController(
            transform.affineBTInMeters, is_signed=True
        )

        self._jitterRadiusLabel = QLabel('Jitter Radius:')
        self._jitterRadiusViewController = LengthWidgetParameterViewController(
            transform.jitterRadiusInMeters, is_signed=False
        )

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


class ScanBoundingBoxViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, item: ScanRepositoryItem) -> None:
        super().__init__(item.expandBoundingBox, 'Expand Bounding Box')
        self._minimumXController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMinimumXInMeters, is_signed=True
        )
        self._maximumXController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMaximumXInMeters, is_signed=True
        )
        self._minimumYController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMinimumYInMeters, is_signed=True
        )
        self._maximumYController = LengthWidgetParameterViewController(
            item.expandedBoundingBoxMaximumYInMeters, is_signed=True
        )

        layout = QFormLayout()
        layout.addRow('Minimum X:', self._minimumXController.getWidget())
        layout.addRow('Maximum X:', self._maximumXController.getWidget())
        layout.addRow('Minimum Y:', self._minimumYController.getWidget())
        layout.addRow('Maximum Y:', self._maximumYController.getWidget())
        self.getWidget().setLayout(layout)


class ScanEditorViewControllerFactory:
    def _appendCommonControls(
        self, dialogBuilder: ParameterViewBuilder, item: ScanRepositoryItem
    ) -> None:
        dialogBuilder.add_view_controller_to_bottom(
            ScanTransformViewController(item.getTransform())
        )
        dialogBuilder.add_view_controller_to_bottom(ScanBoundingBoxViewController(item))

    def createEditorDialog(
        self, itemName: str, item: ScanRepositoryItem, parent: QWidget
    ) -> QDialog:
        scanBuilder = item.getBuilder()
        builderName = scanBuilder.getName()
        baseScanGroup = 'Base Scan'
        title = f'{itemName} [{builderName}]'

        if isinstance(scanBuilder, CartesianScanBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.add_spin_box(
                scanBuilder.numberOfPointsX, 'Number of Points X:', group=baseScanGroup
            )
            dialogBuilder.add_spin_box(
                scanBuilder.numberOfPointsY, 'Number of Points Y:', group=baseScanGroup
            )
            dialogBuilder.addLengthWidget(
                scanBuilder.stepSizeXInMeters, 'Step Size X:', group=baseScanGroup
            )

            if not scanBuilder.isEquilateral:
                dialogBuilder.addLengthWidget(
                    scanBuilder.stepSizeYInMeters, 'Step Size Y:', group=baseScanGroup
                )

            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build_dialog(title, parent)
        elif isinstance(scanBuilder, ConcentricScanBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.add_spin_box(
                scanBuilder.numberOfShells, 'Number of Shells:', group=baseScanGroup
            )
            dialogBuilder.add_spin_box(
                scanBuilder.numberOfPointsInFirstShell,
                'Number of Points in First Shell:',
                group=baseScanGroup,
            )
            dialogBuilder.addLengthWidget(
                scanBuilder.radialStepSizeInMeters,
                'Radial Step Size:',
                group=baseScanGroup,
            )
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build_dialog(title, parent)
        elif isinstance(scanBuilder, FromFileScanBuilder):
            dialogBuilder = ParameterViewBuilder()
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build_dialog(title, parent)
        elif isinstance(scanBuilder, FromMemoryScanBuilder):
            dialogBuilder = ParameterViewBuilder()
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build_dialog(title, parent)
        elif isinstance(scanBuilder, SpiralScanBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.add_spin_box(
                scanBuilder.numberOfPoints, 'Number of Points:', group=baseScanGroup
            )
            dialogBuilder.addLengthWidget(
                scanBuilder.radiusScalarInMeters, 'Radius Scalar:', group=baseScanGroup
            )
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build_dialog(title, parent)
        elif isinstance(scanBuilder, LissajousScanBuilder):
            dialogBuilder = ParameterViewBuilder()
            dialogBuilder.add_spin_box(
                scanBuilder.numberOfPoints, 'Number of Points:', group=baseScanGroup
            )
            dialogBuilder.addLengthWidget(
                scanBuilder.amplitudeXInMeters, 'Amplitude X:', group=baseScanGroup
            )
            dialogBuilder.addLengthWidget(
                scanBuilder.amplitudeYInMeters, 'Amplitude Y:', group=baseScanGroup
            )
            dialogBuilder.addAngleWidget(
                scanBuilder.angularStepXInTurns, 'Angular Step X:', group=baseScanGroup
            )
            dialogBuilder.addAngleWidget(
                scanBuilder.angularStepYInTurns, 'Angular Step Y:', group=baseScanGroup
            )
            dialogBuilder.addAngleWidget(
                scanBuilder.angularShiftInTurns, 'Angular Shift:', group=baseScanGroup
            )
            self._appendCommonControls(dialogBuilder, item)
            return dialogBuilder.build_dialog(title, parent)

        return QMessageBox(
            QMessageBox.Icon.Information,
            title,
            f'"{builderName}" has no editable parameters!',
            QMessageBox.Ok,
            parent,
        )
