from typing import Any, Final

from PyQt5.QtCore import QAbstractListModel, QModelIndex, QObject, Qt
from PyQt5.QtGui import QColor

from ...model.agent import ChatMessageSender, ChatRepository


class AgentMessageListModel(QAbstractListModel):
    DARK_BLUE: Final[QColor] = QColor('#243689')
    LIGHT_BLUE: Final[QColor] = QColor('#0492d2')
    DARK_GREEN: Final[QColor] = QColor('#00894d')
    LIGHT_GREEN: Final[QColor] = QColor('#78ca2a')

    def __init__(self, model: ChatRepository, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._model = model

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid():
            message = self._model[index.row()]

            match role:
                case Qt.ItemDataRole.DisplayRole:
                    return message.contents
                case Qt.ItemDataRole.TextAlignmentRole:
                    return (
                        Qt.AlignRight if message.sender == ChatMessageSender.HUMAN else Qt.AlignLeft
                    )
                case Qt.ItemDataRole.BackgroundRole:
                    return (
                        self.LIGHT_BLUE
                        if message.sender == ChatMessageSender.HUMAN
                        else self.LIGHT_GREEN
                    )
                case Qt.ItemDataRole.ForegroundRole:
                    return (
                        self.DARK_BLUE
                        if message.sender == ChatMessageSender.HUMAN
                        else self.DARK_GREEN
                    )

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._model)
