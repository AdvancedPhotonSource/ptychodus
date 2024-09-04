from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QSpinBox, QVBoxLayout, QWidget

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoPackPositionCorrectionView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Position Correction', parent)
        self.start_spinbox = QSpinBox()
        self.stop_spinbox = QSpinBox()
        self.stride_spinbox = QSpinBox()
        self.probe_threshold_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.feedback_line_edit = DecimalLineEdit.createInstance()

        layout = QFormLayout()
        layout.addRow('Start Iteration:', self.start_spinbox)
        layout.addRow('Stop Iteration:', self.stop_spinbox)
        layout.addRow('Iteration Stride:', self.stride_spinbox)
        layout.addRow('Probe Threshold:', self.probe_threshold_slider)
        layout.addRow('Feedback:', self.feedback_line_edit)
        self.setLayout(layout)


class PtychoPackProbeCorrectionView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Probe Correction', parent)
        self.start_spinbox = QSpinBox()
        self.stop_spinbox = QSpinBox()
        self.stride_spinbox = QSpinBox()
        self.beta_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()
        layout.addRow('Start Iteration:', self.start_spinbox)
        layout.addRow('Stop Iteration:', self.stop_spinbox)
        layout.addRow('Iteration Stride:', self.stride_spinbox)
        layout.addRow('Beta:', self.beta_slider)
        layout.addRow('Relaxation:', self.relaxation_slider)
        self.setLayout(layout)


class PtychoPackObjectCorrectionView(QGroupBox):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__('Object Correction', parent)
        self.start_spinbox = QSpinBox()
        self.stop_spinbox = QSpinBox()
        self.stride_spinbox = QSpinBox()
        self.alpha_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()
        layout.addRow('Start Iteration:', self.start_spinbox)
        layout.addRow('Stop Iteration:', self.stop_spinbox)
        layout.addRow('Iteration Stride:', self.stride_spinbox)
        layout.addRow('Alpha:', self.alpha_slider)
        layout.addRow('Relaxation:', self.relaxation_slider)
        self.setLayout(layout)


class PtychoPackParametersView(QWidget):

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.position_correction_view = PtychoPackPositionCorrectionView()
        self.probe_correction_view = PtychoPackProbeCorrectionView()
        self.object_correction_view = PtychoPackObjectCorrectionView()

        layout = QVBoxLayout()
        layout.addWidget(self.position_correction_view)
        layout.addWidget(self.probe_correction_view)
        layout.addWidget(self.object_correction_view)
        layout.addStretch()
        self.setLayout(layout)
