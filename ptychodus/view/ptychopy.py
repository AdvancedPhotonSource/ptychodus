from __future__ import annotations

from PyQt5.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QSpinBox, QVBoxLayout, QWidget


class PtychoPyBasicView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__('Basic Parameters', parent)
        self.probeModesSpinBox = QSpinBox()
        self.thresholdSpinBox = QSpinBox()
        self.iterationLimitSpinBox = QSpinBox()
        self.timeLimitSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> PtychoPyBasicView:
        view = cls(parent)

        view.probeModesSpinBox.setToolTip(
            'Number of orthogonal probe modes to simulate partial incoherence of the beam')
        view.thresholdSpinBox.setToolTip(
            'To remove noise from the diffraction patterns. Any count below this number will be set to zero in the diffraction data.'
        )
        view.iterationLimitSpinBox.setToolTip('Number of reconstruction iterations')
        view.timeLimitSpinBox.setToolTip(
            'Maximum allowed reconstruction time (in sec). Overrides iterations.')

        layout = QFormLayout()
        layout.addRow('Probe Modes:', view.probeModesSpinBox)
        layout.addRow('Threshold:', view.thresholdSpinBox)
        layout.addRow('Iteration Limit:', view.iterationLimitSpinBox)
        layout.addRow('Time Limit:', view.timeLimitSpinBox)
        view.setLayout(layout)

        return view


class PtychoPyAdvancedView(QGroupBox):
    def __init__(self, parent: QWidget) -> None:
        super().__init__('Advanced Parameters', parent)
        self.calculateRMSCheckBox = QCheckBox('Calculate RMS')
        self.updateProbeSpinBox = QSpinBox()
        self.updateModesSpinBox = QSpinBox()
        self.phaseConstraintSpinBox = QSpinBox()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> PtychoPyAdvancedView:
        view = cls(parent)

        view.updateProbeSpinBox.setToolTip('The number of iterations after which to start updating'
                                           ' the primary probe mode')
        view.updateModesSpinBox.setToolTip('The number of iterations after which to start updating'
                                           ' all probe modes')
        view.phaseConstraintSpinBox.setToolTip(
            'The number of iterations to keep applying'
            ' a phase constraint (forcing the reconstructed phase in the range [-2pi, 0])')

        layout = QFormLayout()
        layout.addRow(view.calculateRMSCheckBox)
        layout.addRow('updateProbe:', view.updateProbeSpinBox)
        layout.addRow('updateModes:', view.updateModesSpinBox)
        layout.addRow('phaseConstraint:', view.phaseConstraintSpinBox)
        view.setLayout(layout)

        return view


class PtychoPyParametersView(QWidget):
    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.basicView = PtychoPyBasicView.createInstance()
        self.advancedView = PtychoPyAdvancedView.createInstance()

    @classmethod
    def createInstance(cls, parent: QWidget = None) -> PtychoPyParametersView:
        view = cls(parent)

        layout = QVBoxLayout()
        layout.addWidget(view.basicView)
        layout.addWidget(view.advancedView)
        layout.addStretch()
        view.setLayout(layout)

        return view
