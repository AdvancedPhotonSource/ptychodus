from __future__ import annotations
import enum

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QVBoxLayout, QWidget)

from .widgets import DecimalLineEdit, DecimalSlider


class PtychoPackAlgorithm(enum.Enum):
    PIE = enum.auto()
    DM = enum.auto()
    RAAR = enum.auto()


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


class PtychoPackExitWaveCorrectionView(QGroupBox):

    def __init__(self, algorithm: PtychoPackAlgorithm, parent: QWidget | None = None) -> None:
        super().__init__('Exit Wave Correction', parent)
        self.dm_relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.raar_relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()

        if algorithm == PtychoPackAlgorithm.DM:
            layout.addRow('Relaxation:', self.dm_relaxation_slider)

        if algorithm == PtychoPackAlgorithm.RAAR:
            layout.addRow('Relaxation:', self.raar_relaxation_slider)

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

        self.start_lineedit.setToolTip('Iteration to start correcting')
        self.stop_lineedit.setToolTip('Iteration to stop correcting')
        self.stride_lineedit.setToolTip('Number of iterations between corrections')

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.start_lineedit)
        layout.addWidget(self.stop_lineedit)
        layout.addWidget(self.stride_lineedit)
        self.setLayout(layout)


class PtychoPackObjectCorrectionView(QGroupBox):

    def __init__(self, algorithm: PtychoPackAlgorithm, parent: QWidget | None = None) -> None:
        super().__init__('Object Correction', parent)
        self.plan_widget = PtychoPackCorrectionPlanWidget()
        self.pie_alpha_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.pie_relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self.plan_widget)

        if algorithm == PtychoPackAlgorithm.PIE:
            layout.addRow('Alpha:', self.pie_alpha_slider)
            layout.addRow('Relaxation:', self.pie_relaxation_slider)

        self.setLayout(layout)


class PtychoPackProbeCorrectionView(QGroupBox):

    def __init__(self, algorithm: PtychoPackAlgorithm, parent: QWidget | None = None) -> None:
        super().__init__('Probe Correction', parent)
        self.plan_widget = PtychoPackCorrectionPlanWidget()
        self.pie_beta_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)
        self.pie_relaxation_slider = DecimalSlider.createInstance(Qt.Orientation.Horizontal)

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self.plan_widget)

        if algorithm == PtychoPackAlgorithm.PIE:
            layout.addRow('Beta:', self.pie_beta_slider)
            layout.addRow('Relaxation:', self.pie_relaxation_slider)

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

    def __init__(self, algorithm: PtychoPackAlgorithm, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.parameters_view = PtychoPackParametersView()
        self.exit_wave_view = PtychoPackExitWaveCorrectionView(algorithm)
        self.object_view = PtychoPackObjectCorrectionView(algorithm)
        self.probe_view = PtychoPackProbeCorrectionView(algorithm)
        self.position_view = PtychoPackPositionCorrectionView()

        layout = QVBoxLayout()
        layout.addWidget(self.parameters_view)

        if algorithm != PtychoPackAlgorithm.PIE:
            layout.addWidget(self.exit_wave_view)

        layout.addWidget(self.object_view)
        layout.addWidget(self.probe_view)
        layout.addWidget(self.position_view)
        layout.addStretch()
        self.setLayout(layout)
