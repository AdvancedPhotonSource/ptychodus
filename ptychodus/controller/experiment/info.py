from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QWidget

from ...api.experiment import Experiment
from ...api.observer import Observable, Observer
from ...view.experiment import ExperimentInfoDialog


class ExperimentPropertyTableModel(QAbstractTableModel):

    def __init__(self, experiment: Experiment, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._experiment = experiment
        self._header = ['Property', 'Value']
        self._properties = [
            'Probe Wavelength [nm]',
            'Fresnel Number',
            'Object Plane Pixel Width [nm]',
            'Object Plane Pixel Height [nm]',
            'Resolution Gain',  # FIXME
        ]

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            if index.column() == 0:
                value = QVariant(self._properties[index.row()])
            elif index.column() == 1:
                if index.row() == 0:
                    probeEnergy = self._experiment.getProbeEnergyInElectronVolts()
                    value = QVariant(f'{probeEnergy / 1000:.1f}')
                else:
                    value = QVariant(index.row())  # FIXME

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ExperimentInfoViewController(Observer):

    def __init__(self, experiment: Experiment, tableModel: ExperimentPropertyTableModel) -> None:
        super().__init__()
        self._experiment = experiment
        self._tableModel = tableModel

    @classmethod
    def showInfo(cls, experiment: Experiment, parent: QWidget) -> None:
        tableModel = ExperimentPropertyTableModel(experiment)
        controller = cls(experiment, tableModel)
        experiment.addObserver(controller)

        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        dialog = ExperimentInfoDialog.createInstance(parent)
        dialog.setWindowTitle(f'Edit Experiment: {experiment.getName()}')
        dialog.tableView.setModel(tableProxyModel)
        dialog.tableView.setSortingEnabled(True)
        dialog.tableView.verticalHeader().hide()
        dialog.finished.connect(controller._finish)

        controller._syncModelToView()
        dialog.open()

    def _finish(self, result: int) -> None:
        self._experiment.removeObserver(self)

    def _syncModelToView(self) -> None:
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def update(self, observable: Observable) -> None:
        if observable is self._experiment:
            self._syncModelToView()
