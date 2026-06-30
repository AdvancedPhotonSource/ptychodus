from collections.abc import Callable, Iterable

from PyQt5.QtCore import QEvent, QModelIndex, QObject, Qt
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QListView,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ptychodus.api.observer import Observable, Observer
from ptychodus.api.parametric import StringParameter

from ...model.agent import (
    AgentSettings,
    ChatMessage,
    ChatTerminal,
    ConversationObserver,
    ConversationRepository,
    ModelCatalog,
)
from ...view.agent import AgentChatView, AgentInputView, AgentView
from ..parametric import (
    DecimalSliderParameterViewController,
    LineEditParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
)
from .item_delegate import ChatBubbleItemDelegate
from .list_model import AgentMessageListModel

__all__ = ['AgentChatController', 'AgentController']


class AgentInputController(QObject):
    def __init__(self, terminal: ChatTerminal, view: AgentInputView) -> None:
        super().__init__()
        self._terminal = terminal
        self._view = view

        view.text_edit.installEventFilter(self)
        view.send_button.clicked.connect(self._send_message)
        view.clear_button.clicked.connect(terminal.clear_conversation)

    def _send_message(self) -> None:
        text = self._view.text_edit.toPlainText()
        self._terminal.send_message(text)
        self._view.text_edit.clear()

    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:  # noqa: N802
        if a0 == self._view.text_edit and isinstance(a1, QKeyEvent):
            is_shift_pressed = bool(a1.modifiers() & Qt.KeyboardModifier.ShiftModifier)

            # require shift+enter for new line, otherwise send on enter
            if a1.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return) and not is_shift_pressed:
                self._send_message()
                return True

        return super().eventFilter(a0, a1)


class AgentChatController(ConversationObserver):
    def __init__(
        self,
        repository: ConversationRepository,
        terminal: ChatTerminal,
        view: AgentChatView,
    ) -> None:
        super().__init__()
        self._repository = repository
        self._terminal = terminal
        self._view = view
        self._message_list_model = AgentMessageListModel(repository)
        self._input_controller = AgentInputController(terminal, view.input_view)

        view.message_list_view.setModel(self._message_list_model)
        view.message_list_view.setItemDelegate(ChatBubbleItemDelegate())
        view.message_list_view.setResizeMode(QListView.ResizeMode.Adjust)
        view.message_list_view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        repository.add_observer(self)

    def handle_message_appended(self, message: ChatMessage, index: int) -> None:
        parent = QModelIndex()
        self._message_list_model.beginInsertRows(parent, index, index)
        self._message_list_model.endInsertRows()

    def handle_conversation_cleared(self) -> None:
        self._message_list_model.beginResetModel()
        self._message_list_model.endResetModel()


class _RefreshableComboBoxParameterViewController(ParameterViewController, Observer):
    """ComboBox bound to a StringParameter with a Refresh button that repopulates the items."""

    def __init__(
        self,
        parameter: StringParameter,
        on_refresh: Callable[[], Iterable[str]],
        *,
        tool_tip: str = '',
    ) -> None:
        super().__init__()
        self._parameter = parameter
        self._on_refresh = on_refresh

        self._combo = QComboBox()
        self._refresh_button = QPushButton('Refresh')

        if tool_tip:
            self._combo.setToolTip(tool_tip)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._combo, 1)
        layout.addWidget(self._refresh_button)

        self._widget = QWidget()
        self._widget.setLayout(layout)

        self._combo.textActivated.connect(parameter.set_value)
        self._refresh_button.clicked.connect(self._handle_refresh_clicked)
        parameter.add_observer(self)
        self.__sync_model_to_view()

    def get_widget(self) -> QWidget:
        return self._widget

    def populate(self, items: Iterable[str]) -> None:
        self._combo.blockSignals(True)
        self._combo.clear()
        for item in items:
            self._combo.addItem(item)
        self._combo.blockSignals(False)
        self.__sync_model_to_view()

    def _handle_refresh_clicked(self) -> None:
        self.populate(self._on_refresh())

    def __sync_model_to_view(self) -> None:
        self._combo.setCurrentText(self._parameter.get_value())

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class AgentController(QObject):
    def __init__(self, settings: AgentSettings, catalog: ModelCatalog, view: AgentView) -> None:
        super().__init__()
        self._settings = settings
        self._catalog = catalog
        self._view = view
        self._models_loaded = False

        self._base_url_view_controller = LineEditParameterViewController(
            settings.base_url,
            tool_tip=(
                'OpenAI-compatible base URL. pydantic-ai appends /chat/completions and /models. '
                'OPENAI_API_KEY env var is sent as a Bearer token; for Argo set this to your '
                'Argonne username.'
            ),
        )
        self._model_view_controller = _RefreshableComboBoxParameterViewController(
            settings.model,
            catalog.refresh,
            tool_tip='The chat model to use. Click Refresh to re-fetch the list from /models.',
        )
        self._system_prompt_view_controller = LineEditParameterViewController(
            settings.system_prompt,
            tool_tip='System prompt sent at the start of every conversation.',
        )
        self._temperature_view_controller = DecimalSliderParameterViewController(
            settings.temperature,
            tool_tip=(
                'Sampling temperature between 0 and 2. Higher values mean the model takes '
                'more risks.'
            ),
        )
        self._top_p_view_controller = DecimalSliderParameterViewController(
            settings.top_p,
            tool_tip=(
                'Nucleus sampling: the model considers tokens with top_p probability mass. '
                'Alternative to temperature.'
            ),
        )
        self._max_tokens_view_controller = SpinBoxParameterViewController(
            settings.max_tokens,
            tool_tip='Maximum number of tokens generated in the chat completion.',
        )
        self._mcp_server_url_view_controller = LineEditParameterViewController(
            settings.mcp_server_url,
            tool_tip='Optional MCP server URL (e.g. ptychodus_store at http://localhost:8000/mcp).',
        )

        group_box_layout = QFormLayout()
        group_box_layout.addRow('Base URL:', self._base_url_view_controller.get_widget())
        group_box_layout.addRow('Model:', self._model_view_controller.get_widget())
        group_box_layout.addRow('System Prompt:', self._system_prompt_view_controller.get_widget())
        group_box_layout.addRow('Temperature:', self._temperature_view_controller.get_widget())
        group_box_layout.addRow('Top P:', self._top_p_view_controller.get_widget())
        group_box_layout.addRow('Max Tokens:', self._max_tokens_view_controller.get_widget())
        group_box_layout.addRow(
            'MCP Server URL:', self._mcp_server_url_view_controller.get_widget()
        )

        group_box = QGroupBox('Agent')
        group_box.setLayout(group_box_layout)

        layout = QVBoxLayout()
        layout.addWidget(group_box)
        layout.addStretch()
        view.setLayout(layout)

        view.installEventFilter(self)

    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:  # noqa: N802
        if a0 is self._view and a1.type() == QEvent.Type.Show and not self._models_loaded:
            self._models_loaded = True
            self._model_view_controller.populate(self._catalog.get_available_models())
        return super().eventFilter(a0, a1)
