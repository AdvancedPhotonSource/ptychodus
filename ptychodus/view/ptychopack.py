from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QSpinBox,
                             QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoPackParametersView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('PtychoPack', parent)
        self.device_combobox = QComboBox()
        self.iterations_label = QLabel('Planned Iterations: 999')  # FIXME to controller

        self.device_combobox.setToolTip("Device to use for reconstruction.")
        self.iterations_label.setToolTip("Number of iterations needed to execute correction plan.")

        layout = QFormLayout()
        layout.addRow('Device:', self.device_combobox)
        layout.addRow(self.iterations_label)
        self.setLayout(layout)


class PtychoPackCorrectionPlanWidget(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.start_spinbox = QSpinBox()
        self.stop_spinbox = QSpinBox()
        self.stride_spinbox = QSpinBox()

        self.start_spinbox.setToolTip("Iteration to start correcting.")
        self.stop_spinbox.setToolTip("Iteration to stop correcting.")
        self.stride_spinbox.setToolTip("Iteration stride between corrections.")

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.start_spinbox)
        layout.addWidget(self.stop_spinbox)
        layout.addWidget(self.stride_spinbox)
        self.setLayout(layout)


class PtychoPackExitWaveCorrectionView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Exit Wave Correction', parent)
        self.relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()
        layout.addRow('Relaxation:', self.relaxation_slider)
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
        self.exit_wave_view = PtychoPackExitWaveCorrectionView()
        self.object_view = PtychoPackObjectCorrectionView()
        self.probe_view = PtychoPackProbeCorrectionView()
        self.position_view = PtychoPackPositionCorrectionView()

        layout = QVBoxLayout()
        layout.addWidget(self.parameters_view)
        layout.addWidget(self.exit_wave_view)
        layout.addWidget(self.object_view)
        layout.addWidget(self.probe_view)
        layout.addWidget(self.position_view)
        layout.addStretch()
        self.setLayout(layout)
