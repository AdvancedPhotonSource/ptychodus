from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoPackParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('PtychoPack', parent)
        self.device_combobox = QComboBox()
        self.plan_label = QLabel()

        self.device_combobox.setToolTip('Device to use for reconstruction.')
        self.plan_label.setToolTip('Correction plan information.')

        layout = QFormLayout()
        layout.addRow('Device:', self.device_combobox)
        layout.addRow(self.plan_label)
        self.setLayout(layout)


class PtychoPackCorrectionPlanWidget(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.start_lineedit = QLineEdit()
        self.stop_lineedit = QLineEdit()
        self.stride_lineedit = QLineEdit()

        self.start_lineedit.setValidator(QIntValidator())
        self.stop_lineedit.setValidator(QIntValidator())
        self.stride_lineedit.setValidator(QIntValidator())

        self.start_lineedit.setToolTip('Iteration to start correcting.')
        self.stop_lineedit.setToolTip('Iteration to stop correcting.')
        self.stride_lineedit.setToolTip('Iteration stride between corrections.')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.start_lineedit)
        layout.addWidget(self.stop_lineedit)
        layout.addWidget(self.stride_lineedit)
        self.setLayout(layout)


class PtychoPackObjectCorrectionView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Object Correction', parent)
        self.plan_widget = PtychoPackCorrectionPlanWidget()
        self.alpha_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self.plan_widget)
        layout.addRow('Alpha:', self.alpha_slider)
        layout.addRow('Relaxation:', self.relaxation_slider)
        self.setLayout(layout)


class PtychoPackProbeCorrectionView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Probe Correction', parent)
        self.plan_widget = PtychoPackCorrectionPlanWidget()
        self.power_plan_widget = PtychoPackCorrectionPlanWidget()
        self.beta_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self.plan_widget)
        layout.addRow('Power Correction Plan:', self.power_plan_widget)
        layout.addRow('Beta:', self.beta_slider)
        layout.addRow('Relaxation:', self.relaxation_slider)
        self.setLayout(layout)


class PtychoPackPositionCorrectionView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Position Correction', parent)
        self.plan_widget = PtychoPackCorrectionPlanWidget()
        self.probe_threshold_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.feedback_line_edit = DecimalLineEdit.createInstance()

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self.plan_widget)
        layout.addRow('Probe Threshold:', self.probe_threshold_slider)
        layout.addRow('Feedback:', self.feedback_line_edit)
        self.setLayout(layout)


class PtychoPackView(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameters_view = PtychoPackParametersView()
        self.object_view = PtychoPackObjectCorrectionView()
        self.probe_view = PtychoPackProbeCorrectionView()
        self.position_view = PtychoPackPositionCorrectionView()

        layout = QVBoxLayout()
        layout.addWidget(self.parameters_view)
        layout.addWidget(self.object_view)
        layout.addWidget(self.probe_view)
        layout.addWidget(self.position_view)
        layout.addStretch()
        self.setLayout(layout)
