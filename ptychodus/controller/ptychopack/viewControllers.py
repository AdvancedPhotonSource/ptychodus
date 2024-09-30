from __future__ import annotations
import enum
import logging

from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import IntegerParameter

from ...model.ptychopack import PtychoPackPresenter, PtychoPackSettings
from ..parametric import (
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    IntegerLineEditParameterViewController,
    ParameterViewController,
)

logger = logging.getLogger(__name__)


class PtychoPackAlgorithm(enum.Enum):
    PIE = enum.auto()
    DM = enum.auto()
    RAAR = enum.auto()


class PtychoPackParametersViewController(ParameterViewController, Observer):
    def __init__(self, presenter: PtychoPackPresenter) -> None:
        super().__init__()
        self._presenter = presenter
        self._deviceComboBox = QComboBox()
        self._planLabel = QLabel()
        self._widget = QGroupBox('PtychoPack Parameters')

        for device in presenter.get_available_devices():
            self._deviceComboBox.addItem(device)

        self._deviceComboBox.textActivated.connect(presenter.set_device)

        self._deviceComboBox.setToolTip('Device to use for reconstruction.')
        self._planLabel.setToolTip('Correction plan information.')

        layout = QFormLayout()
        layout.addRow('Device:', self._deviceComboBox)
        layout.addRow(self._planLabel)
        self._widget.setLayout(layout)

        self._syncModelToView()
        self._presenter.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        self._deviceComboBox.setCurrentText(self._presenter.get_device())
        self._planLabel.setText(self._presenter.get_plan())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PtychoPackExitWaveCorrectionViewController(ParameterViewController):
    def __init__(self, settings: PtychoPackSettings, algorithm: PtychoPackAlgorithm) -> None:
        super().__init__()
        self._widget = QGroupBox('Exit Wave Correction')

        if algorithm == PtychoPackAlgorithm.DM:
            self._relaxationViewController = DecimalSliderParameterViewController(
                settings.dm_exit_wave_relaxation
            )  # TODO tool_tip

        if algorithm == PtychoPackAlgorithm.RAAR:
            self._relaxationViewController = DecimalSliderParameterViewController(
                settings.raar_exit_wave_relaxation
            )  # TODO tool_tip

        layout = QFormLayout()
        layout.addRow('Relaxation:', self._relaxationViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtychoPackCorrectionPlanViewController(ParameterViewController):
    def __init__(
        self, start: IntegerParameter, stop: IntegerParameter, stride: IntegerParameter
    ) -> None:
        super().__init__()
        self._startViewController = IntegerLineEditParameterViewController(
            start, tool_tip='Iteration to start correcting'
        )
        self._stopViewController = IntegerLineEditParameterViewController(
            stop, tool_tip='Iteration to stop correcting'
        )
        self._strideViewController = IntegerLineEditParameterViewController(
            stride, tool_tip='Number of iterations between corrections'
        )
        self._widget = QWidget()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._startViewController.getWidget())
        layout.addWidget(self._stopViewController.getWidget())
        layout.addWidget(self._strideViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtychoPackObjectCorrectionViewController(ParameterViewController):
    def __init__(self, settings: PtychoPackSettings, algorithm: PtychoPackAlgorithm) -> None:
        super().__init__()
        self._planViewController = PtychoPackCorrectionPlanViewController(
            settings.object_correction_plan_start,
            settings.object_correction_plan_stop,
            settings.object_correction_plan_stride,
        )
        self._widget = QGroupBox('Object Correction')

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self._planViewController.getWidget())

        if algorithm == PtychoPackAlgorithm.PIE:
            self._pieAlphaViewController = DecimalSliderParameterViewController(
                settings.pie_alpha
            )  # TODO tool_tip
            self._pieRelaxationViewController = DecimalSliderParameterViewController(
                settings.pie_object_relaxation
            )  # TODO tool_tip

            layout.addRow('Alpha:', self._pieAlphaViewController.getWidget())
            layout.addRow('Relaxation:', self._pieRelaxationViewController.getWidget())

        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtychoPackProbeCorrectionViewController(ParameterViewController):
    def __init__(self, settings: PtychoPackSettings, algorithm: PtychoPackAlgorithm) -> None:
        super().__init__()
        self._planViewController = PtychoPackCorrectionPlanViewController(
            settings.probe_correction_plan_start,
            settings.probe_correction_plan_stop,
            settings.probe_correction_plan_stride,
        )
        self._widget = QGroupBox('Probe Correction')

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self._planViewController.getWidget())

        if algorithm == PtychoPackAlgorithm.PIE:
            self._pieBetaViewController = DecimalSliderParameterViewController(
                settings.pie_beta
            )  # TODO tool_tip
            self._pieRelaxationViewController = DecimalSliderParameterViewController(
                settings.pie_probe_relaxation
            )  # TODO tool_tip

            layout.addRow('Beta:', self._pieBetaViewController.getWidget())
            layout.addRow('Relaxation:', self._pieRelaxationViewController.getWidget())

        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PtychoPackPositionCorrectionViewController(ParameterViewController):
    def __init__(self, settings: PtychoPackSettings) -> None:
        super().__init__()
        self._planViewController = PtychoPackCorrectionPlanViewController(
            settings.position_correction_plan_start,
            settings.position_correction_plan_stop,
            settings.position_correction_plan_stride,
        )
        self._probeThresholdViewController = DecimalSliderParameterViewController(
            settings.position_correction_probe_threshold
        )  # TODO tool_tip
        self._feedbackViewController = DecimalLineEditParameterViewController(
            settings.position_correction_feedback
        )  # TODO tool_tip
        self._widget = QGroupBox('Position Correction')

        layout = QFormLayout()
        layout.addRow('Correction Plan:', self._planViewController.getWidget())
        layout.addRow('Probe Threshold:', self._probeThresholdViewController.getWidget())
        layout.addRow('Feedback:', self._feedbackViewController.getWidget())

        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget
