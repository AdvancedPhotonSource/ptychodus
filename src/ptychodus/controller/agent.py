from typing import Any

from PyQt5.QtCore import QAbstractListModel, QEvent, QModelIndex, QObject, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QFormLayout, QGroupBox, QVBoxLayout

from ..model.agent import AgentPresenter, ArgoSettings
from ..view.agent import AgentChatView, AgentInputView, AgentView
from .parametric import (
    ComboBoxParameterViewController,
    DecimalSliderParameterViewController,
    LineEditParameterViewController,
    SpinBoxParameterViewController,
)

__all__ = ['AgentChatController', 'AgentController']


class AgentMessageListModel(QAbstractListModel):
    def __init__(self, presenter: AgentPresenter, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._presenter = presenter

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole) -> Any:
        if index.isValid() and role == Qt.ItemDataRole.DisplayRole:
            # FIXME indicate sent vs received
            # FIXME make clearable
            return self._presenter.get_message(index.row())

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return self._presenter.get_number_of_messages()


class AgentInputController(QObject):
    def __init__(self, presenter: AgentPresenter, view: AgentInputView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

        view.textEdit.installEventFilter(self)
        view.sendButton.clicked.connect(self._send_message)

    def _send_message(self) -> None:
        text = self._view.textEdit.toPlainText()
        self._presenter.send_message(text)
        self._view.textEdit.clear()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj == self._view.textEdit and isinstance(event, QKeyEvent):
            isShiftPressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier

            # require shift+enter for new line, otherwise send on enter
            if event.key() in (Qt.Key_Enter, Qt.Key_Return) and not isShiftPressed:
                self._send_message()
                return True

        return super().eventFilter(obj, event)


class AgentChatController:
    def __init__(self, presenter: AgentPresenter, view: AgentChatView) -> None:
        self._presenter = presenter
        self._view = view
        self._messageListModel = AgentMessageListModel(presenter)
        self._inputController = AgentInputController(presenter, view.inputView)

        view.messageListView.setModel(self._messageListModel)


#     def handleNewMessage(self, index: int, item: ProductRepositoryItem) -> None: # FIXME
#         parent = QModelIndex()
#         self._messageListModel.beginInsertRows(parent, index, index)
#         self._messageListModel.endInsertRows()


class AgentController:
    def __init__(self, settings: ArgoSettings, presenter: AgentPresenter, view: AgentView) -> None:
        # FIXME tool_tip
        self._chatEndpointURLViewController = LineEditParameterViewController(
            settings.chatEndpointURL
        )
        self._embedEndpointURLViewController = LineEditParameterViewController(
            settings.embedEndpointURL
        )
        self._userViewController = LineEditParameterViewController(settings.user)
        self._modelViewController = ComboBoxParameterViewController(
            settings.model, presenter.get_available_models()
        )
        self._temperatureViewController = DecimalSliderParameterViewController(settings.temperature)
        self._topPViewController = DecimalSliderParameterViewController(settings.top_p)
        self._maxTokensViewController = SpinBoxParameterViewController(settings.max_tokens)
        self._maxCompletionTokensViewController = SpinBoxParameterViewController(
            settings.max_completion_tokens
        )

        groupBoxLayout = QFormLayout()
        groupBoxLayout.addRow('Chat Endpoint URL:', self._chatEndpointURLViewController.getWidget())
        groupBoxLayout.addRow(
            'Embed Endpoint URL:', self._embedEndpointURLViewController.getWidget()
        )
        groupBoxLayout.addRow('User:', self._userViewController.getWidget())
        groupBoxLayout.addRow('Model:', self._modelViewController.getWidget())
        groupBoxLayout.addRow('Temperature:', self._temperatureViewController.getWidget())
        groupBoxLayout.addRow('Top P:', self._topPViewController.getWidget())
        groupBoxLayout.addRow('Max Tokens:', self._maxTokensViewController.getWidget())
        groupBoxLayout.addRow(
            'Max Completion Tokens:', self._maxCompletionTokensViewController.getWidget()
        )

        groupBox = QGroupBox('Argo')
        groupBox.setLayout(groupBoxLayout)

        layout = QVBoxLayout()
        layout.addWidget(groupBox)
        layout.addStretch()
        view.setLayout(layout)
