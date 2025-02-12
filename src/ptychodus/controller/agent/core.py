from PyQt5.QtCore import QEvent, QModelIndex, QObject, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QFormLayout,
    QGroupBox,
    QInputDialog,
    QListView,
    QPushButton,
    QVBoxLayout,
)

from ...model.agent import (
    AgentPresenter,
    ArgoSettings,
    ChatHistory,
    ChatMessage,
    ChatObserver,
)
from ...view.agent import AgentChatView, AgentInputView, AgentView
from ..parametric import (
    ComboBoxParameterViewController,
    DecimalSliderParameterViewController,
    LineEditParameterViewController,
    SpinBoxParameterViewController,
)
from .itemDelegate import ChatBubbleItemDelegate
from .listModel import AgentMessageListModel

__all__ = ['AgentChatController', 'AgentController']


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
            is_shift_pressed = event.modifiers() & Qt.KeyboardModifier.ShiftModifier

            # require shift+enter for new line, otherwise send on enter
            if event.key() in (Qt.Key_Enter, Qt.Key_Return) and not is_shift_pressed:
                self._send_message()
                return True

        return super().eventFilter(obj, event)


class AgentChatController(ChatObserver):
    def __init__(
        self, history: ChatHistory, presenter: AgentPresenter, view: AgentChatView
    ) -> None:
        super().__init__()
        self._history = history
        self._presenter = presenter
        self._view = view
        self._message_list_model = AgentMessageListModel(history)
        self._input_controller = AgentInputController(presenter, view.inputView)

        view.messageListView.setModel(self._message_list_model)
        view.messageListView.setItemDelegate(ChatBubbleItemDelegate())
        view.messageListView.setResizeMode(QListView.ResizeMode.Adjust)
        view.messageListView.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)

        history.add_observer(self)

    def handle_new_message(self, message: ChatMessage, index: int) -> None:
        parent = QModelIndex()
        self._message_list_model.beginInsertRows(parent, index, index)
        self._message_list_model.endInsertRows()

    def handle_chat_cleared(self) -> None:
        self._message_list_model.beginResetModel()
        self._message_list_model.endResetModel()


class AgentController:
    def __init__(self, settings: ArgoSettings, presenter: AgentPresenter, view: AgentView) -> None:
        self._settings = settings
        self._presenter = presenter
        self._view = view

        self._userViewController = LineEditParameterViewController(settings.user)
        self._chatEndpointURLViewController = LineEditParameterViewController(
            settings.chatEndpointURL, tool_tip='The chat endpoint URL.'
        )
        self._chatModelViewController = ComboBoxParameterViewController(
            settings.chatModel,
            presenter.get_available_chat_models(),
            tool_tip='The chat model to use.',
        )
        self._temperatureViewController = DecimalSliderParameterViewController(
            settings.temperature,
            tool_tip='What sampling temperature to use, between 0 and 2. Higher values mean the model takes more risks.',
        )
        self._topPViewController = DecimalSliderParameterViewController(
            settings.top_p,
            tool_tip='An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.',
        )
        self._maxTokensViewController = SpinBoxParameterViewController(
            settings.max_tokens,
            tool_tip='The maximum number of tokens that can be generated in the chat completion.',
        )
        self._maxCompletionTokensViewController = SpinBoxParameterViewController(
            settings.max_completion_tokens,
            tool_tip='An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.',
        )
        self._embeddingsEndpointURLViewController = LineEditParameterViewController(
            settings.embeddingsEndpointURL, tool_tip='The embeddings endpoint URL.'
        )
        self._embeddingsModelViewController = ComboBoxParameterViewController(
            settings.embeddingsModel,
            presenter.get_available_embeddings_models(),
            tool_tip='The embeddings model to use.',
        )
        self._embedButton = QPushButton('Embed Text')
        self._embedButton.clicked.connect(self._embed_text)

        groupBoxLayout = QFormLayout()
        groupBoxLayout.addRow('User:', self._userViewController.getWidget())
        groupBoxLayout.addRow('Chat Endpoint URL:', self._chatEndpointURLViewController.getWidget())
        groupBoxLayout.addRow('Chat Model:', self._chatModelViewController.getWidget())
        groupBoxLayout.addRow('Temperature:', self._temperatureViewController.getWidget())
        groupBoxLayout.addRow('Top P:', self._topPViewController.getWidget())
        groupBoxLayout.addRow('Max Tokens:', self._maxTokensViewController.getWidget())
        groupBoxLayout.addRow(
            'Max Completion Tokens:', self._maxCompletionTokensViewController.getWidget()
        )
        groupBoxLayout.addRow(
            'Embeddings Endpoint URL:', self._embeddingsEndpointURLViewController.getWidget()
        )
        groupBoxLayout.addRow('Embeddings Model:', self._embeddingsModelViewController.getWidget())
        groupBoxLayout.addRow(self._embedButton)

        groupBox = QGroupBox('Argo')
        groupBox.setLayout(groupBoxLayout)

        layout = QVBoxLayout()
        layout.addWidget(groupBox)
        layout.addStretch()
        view.setLayout(layout)

    def _embed_text(self) -> None:
        title = 'Embed Text'
        label = 'Enter text to embed:'
        text, okPressed = QInputDialog.getMultiLineText(self._view, title, label, text='')

        if okPressed:
            self._presenter.embed_text(text.splitlines())
