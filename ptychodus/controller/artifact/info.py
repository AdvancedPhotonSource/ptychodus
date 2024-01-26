from PyQt5.QtCore import (Qt, QAbstractTableModel, QModelIndex, QObject, QSortFilterProxyModel,
                          QVariant)
from PyQt5.QtWidgets import QWidget

from ...api.artifact import Artifact
from ...api.observer import Observable, Observer
from ...view.artifact import ArtifactInfoDialog


class ArtifactPropertyTableModel(QAbstractTableModel):

    def __init__(self, artifact: Artifact, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._artifact = artifact
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
                    probeEnergy = self._artifact.getProbeEnergyInElectronVolts()
                    value = QVariant(f'{probeEnergy / 1000:.1f}')
                else:
                    value = QVariant(index.row())  # FIXME

        return value

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._properties)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._header)


class ArtifactInfoViewController(Observer):

    def __init__(self, artifact: Artifact, tableModel: ArtifactPropertyTableModel) -> None:
        super().__init__()
        self._artifact = artifact
        self._tableModel = tableModel

    @classmethod
    def showInfo(cls, artifact: Artifact, parent: QWidget) -> None:
        tableModel = ArtifactPropertyTableModel(artifact)
        controller = cls(artifact, tableModel)
        artifact.addObserver(controller)

        tableProxyModel = QSortFilterProxyModel()
        tableProxyModel.setSourceModel(tableModel)

        dialog = ArtifactInfoDialog.createInstance(parent)
        dialog.setWindowTitle(f'Edit Artifact: {artifact.getName()}')
        dialog.tableView.setModel(tableProxyModel)
        dialog.tableView.setSortingEnabled(True)
        dialog.tableView.verticalHeader().hide()
        dialog.finished.connect(controller._finish)

        controller._syncModelToView()
        dialog.open()

    def _finish(self, result: int) -> None:
        self._artifact.removeObserver(self)

    def _syncModelToView(self) -> None:
        self._tableModel.beginResetModel()
        self._tableModel.endResetModel()

    def update(self, observable: Observable) -> None:
        if observable is self._artifact:
            self._syncModelToView()
