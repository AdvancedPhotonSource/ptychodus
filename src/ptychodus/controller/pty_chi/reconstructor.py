from typing import Any

from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex, QObject
from PyQt5.QtWidgets import (
    QButtonGroup,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListView,
    QRadioButton,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import BooleanParameter, IntegerSequenceParameter

from ...model.pty_chi import PtyChiDeviceRepository, PtyChiReconstructorSettings
from ..parametric import ParameterViewController, SpinBoxParameterViewController


class PtyChiDeviceListModel(QAbstractListModel):
    def __init__(self, repository: PtyChiDeviceRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._repository = repository
        self._checkedRows: set[int] = set()

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            if role == Qt.ItemDataRole.DisplayRole:
                return self._repository[index.row()]
            elif role == Qt.ItemDataRole.CheckStateRole:
                return (
                    Qt.CheckState.Checked
                    if index.row() in self._checkedRows
                    else Qt.CheckState.Unchecked
                )

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        value = super().flags(index)

        if index.isValid():
            value |= Qt.ItemFlag.ItemIsUserCheckable

        return value

    def setData(self, index: QModelIndex, value: Any, role: int = Qt.ItemDataRole.EditRole) -> bool:
        if index.isValid() and role == Qt.ItemDataRole.CheckStateRole:
            if value == Qt.CheckState.Checked:
                self._checkedRows.add(index.row())
            else:
                self._checkedRows.discard(index.row())

            self.dataChanged.emit(index, index)

            return True

        return False

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._repository)


class PtyChiDeviceViewController(ParameterViewController, Observer):
    def __init__(
        self,
        useDevices: BooleanParameter,
        devices: IntegerSequenceParameter,
        repository: PtyChiDeviceRepository,
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__()
        self._useDevices = useDevices
        self._devices = devices
        self._repository = repository
        self._listModel = PtyChiDeviceListModel(repository)
        self._widget = QListView()
        self._widget.setModel(self._listModel)

        if tool_tip:
            self._widget.setToolTip(tool_tip)

        self._syncModelToView()
        useDevices.addObserver(self)
        devices.addObserver(self)

    def getWidget(self) -> QWidget:
        return self._widget

    def _syncModelToView(self) -> None:
        pass  # FIXME update checked indexes, etc.

    def update(self, observable: Observable) -> None:
        if observable in (self._useDevices, self._devices):
            self._syncModelToView()


class PtyChiPrecisionParameterViewController(ParameterViewController, Observer):
    def __init__(self, useDoublePrecision: BooleanParameter, *, tool_tip: str = '') -> None:
        super().__init__()
        self._useDoublePrecision = useDoublePrecision
        self._singlePrecisionButton = QRadioButton('Single')
        self._doublePrecisionButton = QRadioButton('Double')
        self._buttonGroup = QButtonGroup()
        self._widget = QWidget()

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
        self, settings: PtyChiReconstructorSettings, repository: PtyChiDeviceRepository
    ) -> None:
        super().__init__()
        self._numEpochsViewController = SpinBoxParameterViewController(
            settings.numEpochs, tool_tip='The number of epochs to run.'
        )
        self._batchSizeViewController = SpinBoxParameterViewController(
            settings.batchSize,
            tool_tip='The number of probe positions to process in each minibatch.',
        )
        self._deviceViewController = PtyChiDeviceViewController(
            settings.useDevices,
            settings.devices,
            repository,
            tool_tip='The devices to use for computation.',
        )
        self._precisionViewController = PtyChiPrecisionParameterViewController(
            settings.useDoublePrecision,
            tool_tip='The floating point precision to use for computation.',
        )
        # TODO random_seed
        # TODO displayed_loss_function
        # TODO log_level
        self._widget = QGroupBox('Reconstructor')

        layout = QFormLayout()
        layout.addRow('Number of Epochs:', self._numEpochsViewController.getWidget())
        layout.addRow('Batch Size:', self._batchSizeViewController.getWidget())
        layout.addRow('Devices:', self._deviceViewController.getWidget())
        layout.addRow('Precision:', self._precisionViewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget
