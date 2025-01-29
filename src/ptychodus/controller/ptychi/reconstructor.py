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
from ptychodus.api.parametric import BooleanParameter

from ...model.ptychi import PtyChiDeviceRepository, PtyChiEnumerators, PtyChiReconstructorSettings
from ..parametric import (
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ComboBoxParameterViewController,
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


class PtyChiReconstructorViewController(ParameterViewController):
    def __init__(
        self,
        settings: PtyChiReconstructorSettings,
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

        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget
