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

from ..model.ptychopack import (
    PtychoPackReconstructorLibrary,
    PtychoPackPresenter,
    PtychoPackSettings,
)
from .parametric import (
    IntegerLineEditParameterViewController,
    ParameterViewBuilder,
    ParameterViewController,
)
from .reconstructor import ReconstructorViewControllerFactory

logger = logging.getLogger(__name__)

__all__ = [
    'PtychoPackViewControllerFactory',
]


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


def _createViewController(settings: PtychoPackSettings, algorithm: PtychoPackAlgorithm) -> QWidget:
    builder = ParameterViewBuilder()

    exit_wave_correction = 'Exit Wave Correction'

    if algorithm == PtychoPackAlgorithm.DM:
        builder.addDecimalSlider(
            settings.dm_exit_wave_relaxation, 'Relaxation:', group=exit_wave_correction
        )

    if algorithm == PtychoPackAlgorithm.RAAR:
        builder.addDecimalSlider(
            settings.raar_exit_wave_relaxation, 'Relaxation:', group=exit_wave_correction
        )

    object_correction = 'Object Correction'
    object_plan_view_controller = PtychoPackCorrectionPlanViewController(
        settings.object_correction_plan_start,
        settings.object_correction_plan_stop,
        settings.object_correction_plan_stride,
    )
    builder.addViewController(
        object_plan_view_controller, 'Correction Plan:', group=object_correction
    )

    if algorithm == PtychoPackAlgorithm.PIE:
        builder.addDecimalSlider(settings.pie_alpha, 'Alpha:', group=object_correction)
        builder.addDecimalSlider(
            settings.pie_object_relaxation, 'Relaxation:', group=object_correction
        )

    probe_correction = 'Probe Correction'
    probe_plan_view_controller = PtychoPackCorrectionPlanViewController(
        settings.probe_correction_plan_start,
        settings.probe_correction_plan_stop,
        settings.probe_correction_plan_stride,
    )
    builder.addViewController(
        probe_plan_view_controller, 'Correction Plan:', group=probe_correction
    )

    if algorithm == PtychoPackAlgorithm.PIE:
        builder.addDecimalSlider(settings.pie_beta, 'Beta:', group=probe_correction)
        builder.addDecimalSlider(
            settings.pie_probe_relaxation, 'Relaxation:', group=probe_correction
        )

    position_correction = 'Position Correction'
    position_plan_view_controller = PtychoPackCorrectionPlanViewController(
        settings.position_correction_plan_start,
        settings.position_correction_plan_stop,
        settings.position_correction_plan_stride,
    )
    builder.addViewController(
        position_plan_view_controller, 'Correction Plan:', group=position_correction
    )
    builder.addDecimalSlider(settings.position_correction_probe_threshold, 'Probe Threshold:')
    builder.addDecimalLineEdit(settings.position_correction_feedback, 'Feedback:')

    return builder.buildWidget()


class PtychoPackViewControllerFactory(ReconstructorViewControllerFactory):
    def __init__(self, model: PtychoPackReconstructorLibrary) -> None:
        super().__init__()
        self._model = model

    @property
    def backendName(self) -> str:
        return 'PtychoPack'

    def createViewController(self, reconstructorName: str) -> QWidget:
        for algorithm in PtychoPackAlgorithm:
            if reconstructorName.upper() == algorithm.name:
                return _createViewController(self._model.settings, algorithm)

        raise ValueError(f'Unknown {reconstructorName=}!')
