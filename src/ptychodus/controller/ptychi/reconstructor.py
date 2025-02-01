from PyQt5.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, RealParameter

from ...model.ptychi import (
    PtyChiAutodiffSettings,
    PtyChiDMSettings,
    PtyChiDeviceRepository,
    PtyChiEnumerators,
    PtyChiLSQMLSettings,
    PtyChiReconstructorSettings,
)
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
    DecimalLineEditParameterViewController,
    DecimalSliderParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)

__all__ = ['PtyChiReconstructorViewController']


class PtyChiDeviceViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        useDevices: BooleanParameter,
        repository: PtyChiDeviceRepository,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__(useDevices, 'Use Devices', tool_tip=tool_tip)
        layout = QVBoxLayout()

        for device in repository:
            deviceLabel = QLabel(device)
            layout.addWidget(deviceLabel)

        self.getWidget().setLayout(layout)


class PtyChiPrecisionParameterViewController(ParameterViewController, Observer):
    def __init__(self, useDoublePrecision: BooleanParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._useDoublePrecision = useDoublePrecision
        self._singlePrecisionButton = QRadioButton('Single')
        self._doublePrecisionButton = QRadioButton('Double')
        self._buttonGroup = QButtonGroup()
        self._widget = QWidget()

        self._singlePrecisionButton.setToolTip('Compute using single precision.')
        self._doublePrecisionButton.setToolTip('Compute using double precision.')

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._buttonGroup.addButton(self._singlePrecisionButton, 1)
        self._buttonGroup.addButton(self._doublePrecisionButton, 2)
        self._buttonGroup.setExclusive(True)
        self._buttonGroup.idToggled.connect(self._syncViewToModel)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._singlePrecisionButton)
        layout.addWidget(self._doublePrecisionButton)
        layout.addStretch()
        self._widget.setLayout(layout)

        self._syncModelToView()
        useDoublePrecision.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncViewToModel(self, toolId: int, checked: bool) -> None:
        if toolId == 2:
            self._useDoublePrecision.setValue(checked)

    def _syncModelToView(self) -> None:
        button = self._buttonGroup.button(2 if self._useDoublePrecision.getValue() else 1)
        button.setChecked(True)

    def update(self, observable: Observable) -> None:
        if observable is self._useDoublePrecision:
            self._syncModelToView()


class PtyChiMomentumAccelerationGradientMixingFactorViewController(
    CheckableGroupBoxParameterViewController
):
    def __init__(
        self,
        useGradientMixingFactor: BooleanParameter,
        gradientMixingFactor: RealParameter,
    ) -> None:
        super().__init__(
            useGradientMixingFactor,
            'Use Gradient Mixing Factor',
            tool_tip='Controls how the current gradient is mixed with the accumulated velocity in LSQML momentum acceleration.',
        )
        self._gradientMixingFactorViewController = DecimalLineEditParameterViewController(
            gradientMixingFactor
        )

        layout = QVBoxLayout()
        layout.addWidget(self._gradientMixingFactorViewController.getWidget())
        self.getWidget().setLayout(layout)


class PtyChiReconstructorViewController(ParameterViewController):
    def __init__(
        self,
        settings: PtyChiReconstructorSettings,
        autodiffSettings: PtyChiAutodiffSettings | None,
        dmSettings: PtyChiDMSettings | None,
        lsqmlSettings: PtyChiLSQMLSettings | None,
        enumerators: PtyChiEnumerators,
        repository: PtyChiDeviceRepository,
    ) -> None:
        super().__init__()
        self._numEpochsViewController = SpinBoxParameterViewController(
            settings.numEpochs, tool_tip='Number of epochs to run.'
        )
        self._batchSizeViewController = SpinBoxParameterViewController(
            settings.batchSize, tool_tip='Number of data to process in each minibatch.'
        )
        self._batchingModeViewController = ComboBoxParameterViewController(
            settings.batchingMode, enumerators.batchingModes(), tool_tip='Batching mode to use.'
        )
        self._batchStride = SpinBoxParameterViewController(
            settings.batchStride, tool_tip='Number of epochs between updating clusters.'
        )
        self._precisionViewController = PtyChiPrecisionParameterViewController(
            settings.useDoublePrecision,
            tool_tip='Floating point precision to use for computation.',
        )
        self._deviceViewController = PtyChiDeviceViewController(
            settings.useDevices, repository, tool_tip='Default device to use for computation.'
        )
        self._useLowMemoryViewController = CheckBoxParameterViewController(
            settings.useLowMemoryForwardModel,
            'Use Low Memory Forward Model',
            tool_tip='When checked, forward propagation of ptychography will be done using less vectorized code. This reduces the speed, but also lowers memory usage.',
        )
        self._saveDataOnDeviceViewController = CheckBoxParameterViewController(
            settings.saveDataOnDevice,
            'Save Data on Device',
            tool_tip='When checked, diffraction data will be saved on the device.',
        )
        self._widget = QGroupBox('Reconstructor')

        layout = QFormLayout()
        layout.addRow('Number of Epochs:', self._numEpochsViewController.getWidget())
        layout.addRow('Batch Size:', self._batchSizeViewController.getWidget())
        layout.addRow('Batch Mode:', self._batchingModeViewController.getWidget())
        layout.addRow('Batch Stride:', self._batchStride.getWidget())

        if repository:
            layout.addRow(self._deviceViewController.getWidget())

        layout.addRow('Precision:', self._precisionViewController.getWidget())
        layout.addRow(self._useLowMemoryViewController.getWidget())

        if autodiffSettings is not None:
            self._lossFunctionViewController = ComboBoxParameterViewController(
                autodiffSettings.lossFunction, enumerators.lossFunctions()
            )
            layout.addRow('Loss Function:', self._lossFunctionViewController.getWidget())

            self._forwardModelClassViewController = ComboBoxParameterViewController(
                autodiffSettings.forwardModelClass, enumerators.forwardModels()
            )
            layout.addRow('Forward Model:', self._forwardModelClassViewController.getWidget())

        if dmSettings is not None:
            self._exitWaveUpdateRelaxationViewController = DecimalSliderParameterViewController(
                dmSettings.exitWaveUpdateRelaxation
            )
            layout.addRow(
                'Exit Wave Update Relaxation:',
                self._exitWaveUpdateRelaxationViewController.getWidget(),
            )

            self._chunkLengthViewController = SpinBoxParameterViewController(dmSettings.chunkLength)
            layout.addRow('Chunk Length:', self._chunkLengthViewController.getWidget())

        if lsqmlSettings is not None:
            self._noiseModelViewController = ComboBoxParameterViewController(
                lsqmlSettings.noiseModel, enumerators.noiseModels()
            )
            layout.addRow('Noise Model:', self._noiseModelViewController.getWidget())

            self._gaussianNoiseDeviationViewController = DecimalLineEditParameterViewController(
                lsqmlSettings.gaussianNoiseDeviation
            )
            layout.addRow(
                'Gaussian Noise Deviation:', self._gaussianNoiseDeviationViewController.getWidget()
            )

            self._solveObjectProbeStepSizeJointlyForFirstSliceInMultisliceViewController = (
                CheckBoxParameterViewController(
                    lsqmlSettings.solveObjectProbeStepSizeJointlyForFirstSliceInMultislice,
                    'SolveObjectProbeStepSizeJointlyForFirstSliceInMultislice',
                )
            )
            layout.addRow(
                self._solveObjectProbeStepSizeJointlyForFirstSliceInMultisliceViewController.getWidget()
            )

            self._solveStepSizesOnlyUsingFirstProbeModeViewController = (
                CheckBoxParameterViewController(
                    lsqmlSettings.solveStepSizesOnlyUsingFirstProbeMode,
                    'SolveStepSizesOnlyUsingFirstProbeMode',
                )
            )
            layout.addRow(self._solveStepSizesOnlyUsingFirstProbeModeViewController.getWidget())

            self._momentumAccelerationGainViewController = DecimalLineEditParameterViewController(
                lsqmlSettings.momentumAccelerationGain
            )
            layout.addRow(
                'Momentum Acceleration Gain:',
                self._momentumAccelerationGainViewController.getWidget(),
            )

            self._momentumAccelerationGradientMixingFactorViewController = (
                PtyChiMomentumAccelerationGradientMixingFactorViewController(
                    lsqmlSettings.useMomentumAccelerationGradientMixingFactor,
                    lsqmlSettings.momentumAccelerationGradientMixingFactor,
                )
            )
            layout.addRow(self._momentumAccelerationGradientMixingFactorViewController.getWidget())

        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget
