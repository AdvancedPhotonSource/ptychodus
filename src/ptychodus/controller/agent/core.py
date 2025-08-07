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
from .item_delegate import ChatBubbleItemDelegate
from .list_model import AgentMessageListModel

__all__ = ['AgentChatController', 'AgentController']


class AgentInputController(QObject):
    def __init__(self, presenter: AgentPresenter, view: AgentInputView) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

        view.text_edit.installEventFilter(self)
        view.send_button.clicked.connect(self._send_message)

    def _send_message(self) -> None:
        text = self._view.text_edit.toPlainText()
        self._presenter.send_message(text)
        self._view.text_edit.clear()

    def eventFilter(self, a0: QObject | None, a1: QEvent | None) -> bool:  # noqa: N802
        if a0 == self._view.text_edit and isinstance(a1, QKeyEvent):
            is_shift_pressed = bool(a1.modifiers() & Qt.KeyboardModifier.ShiftModifier)

            # require shift+enter for new line, otherwise send on enter
            if a1.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return) and not is_shift_pressed:
                self._send_message()
                return True

        return super().eventFilter(a0, a1)


class AgentChatController(ChatObserver):
    def __init__(
        self, history: ChatHistory, presenter: AgentPresenter, view: AgentChatView
    ) -> None:
        super().__init__()
        self._history = history
        self._presenter = presenter
        self._view = view
        self._message_list_model = AgentMessageListModel(history)
        self._input_controller = AgentInputController(presenter, view.input_view)

        view.message_list_view.setModel(self._message_list_model)
        view.message_list_view.setItemDelegate(ChatBubbleItemDelegate())
        view.message_list_view.setResizeMode(QListView.ResizeMode.Adjust)
        view.message_list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

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

        self._user_view_controller = LineEditParameterViewController(settings.user)
        self._chat_endpoint_url_view_controller = LineEditParameterViewController(
            settings.chat_endpoint_url, tool_tip='The chat endpoint URL.'
        )
        self._chat_model_view_controller = ComboBoxParameterViewController(
            settings.chat_model,
            presenter.get_available_chat_models(),
            tool_tip='The chat model to use.',
        )
        self._temperature_view_controller = DecimalSliderParameterViewController(
            settings.temperature,
            tool_tip='What sampling temperature to use, between 0 and 2. Higher values mean the model takes more risks.',
        )
        self._top_p_view_controller = DecimalSliderParameterViewController(
            settings.top_p,
            tool_tip='An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass.',
        )
        self._max_tokens_view_controller = SpinBoxParameterViewController(
            settings.max_tokens,
            tool_tip='The maximum number of tokens that can be generated in the chat completion.',
        )
        self._max_completion_tokens_view_controller = SpinBoxParameterViewController(
            settings.max_completion_tokens,
            tool_tip='An upper bound for the number of tokens that can be generated for a completion, including visible output tokens and reasoning tokens.',
        )
        self._embeddings_endpoint_url_view_controller = LineEditParameterViewController(
            settings.embeddings_endpoint_url, tool_tip='The embeddings endpoint URL.'
        )
        self._embeddings_model_view_controller = ComboBoxParameterViewController(
            settings.embeddings_model,
            presenter.get_available_embeddings_models(),
            tool_tip='The embeddings model to use.',
        )
        self._embed_button = QPushButton('Embed Text')
        self._embed_button.clicked.connect(self._embed_text)

        group_box_layout = QFormLayout()
        group_box_layout.addRow('User:', self._user_view_controller.get_widget())
        group_box_layout.addRow(
            'Chat Endpoint URL:', self._chat_endpoint_url_view_controller.get_widget()
        )
        group_box_layout.addRow('Chat Model:', self._chat_model_view_controller.get_widget())
        group_box_layout.addRow('Temperature:', self._temperature_view_controller.get_widget())
        group_box_layout.addRow('Top P:', self._top_p_view_controller.get_widget())
        group_box_layout.addRow('Max Tokens:', self._max_tokens_view_controller.get_widget())
        group_box_layout.addRow(
            'Max Completion Tokens:', self._max_completion_tokens_view_controller.get_widget()
        )
        group_box_layout.addRow(
            'Embeddings Endpoint URL:', self._embeddings_endpoint_url_view_controller.get_widget()
        )
        group_box_layout.addRow(
            'Embeddings Model:', self._embeddings_model_view_controller.get_widget()
        )
        group_box_layout.addRow(self._embed_button)

        group_box = QGroupBox('Argo')
        group_box.setLayout(group_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(group_box)
        layout.addStretch()
        view.setLayout(layout)

    def _embed_text(self) -> None:
        title = 'Embed Text'
        label = 'Enter text to embed:'
        text, ok_pressed = QInputDialog.getMultiLineText(self._view, title, label, text='')

        if ok_pressed:
            self._presenter.embed_text(text.splitlines())
