from __future__ import annotations
import logging

from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QWidget

from ...api.experiment import Experiment
from ...api.observer import Observable, Observer
from ...view.experiment import ExperimentEditorDialog

logger = logging.getLogger(__name__)


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
            'ptycho gain x/y',
        ]

    def headerData(self,
                   section: int,
                   orientation: Qt.Orientation,
                   role: int = Qt.DisplayRole) -> QVariant:
        result = QVariant()

        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            result = QVariant(self._header[section])

        return result

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> QVariant:
        value = QVariant()

        if index.isValid() and role == Qt.DisplayRole:
            if index.column() == 0:
                value = QVariant(self._properties[index.row()])
            elif index.column() == 1:
                value = QVariant(index.row())  # FIXME

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ExperimentEditorViewController(Observer):

    def __init__(self, experiment: Experiment, dialog: ExperimentEditorDialog,
                 tableModel: ExperimentPropertyTableModel) -> None:
        super().__init__()
        self._experiment = experiment
        self._dialog = dialog
        self._tableModel = tableModel

    @classmethod
    def editParameters(cls, experiment: Experiment, parent: QWidget) -> None:
        tableModel = ExperimentPropertyTableModel(experiment)
        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        name = experiment.getName()
        dialog = ExperimentEditorDialog.createInstance(parent)
        dialog.setWindowTitle(f'Edit Experiment: {name}')
        dialog.tableView.setModel(tableProxyModel)
        dialog.tableView.setSortingEnabled(True)
        dialog.tableView.verticalHeader().hide()

        controller = cls(experiment, dialog, tableModel)
        experiment.addObserver(controller)

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
