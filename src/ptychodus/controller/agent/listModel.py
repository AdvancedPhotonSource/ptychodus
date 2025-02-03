from typing import Any

from PyQt5.QtCore import QAbstractListModel, QModelIndex, QObject, Qt

from ...model.agent import (
    ChatRepository,
)


class AgentMessageListModel(QAbstractListModel):
    def __init__(self, model: ChatRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = model

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            message = self._model[index.row()]

            match role:
                case Qt.ItemDataRole.DisplayRole:
                    return message.contents
                case Qt.ItemDataRole.UserRole:
                    return message.sender

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._model)
