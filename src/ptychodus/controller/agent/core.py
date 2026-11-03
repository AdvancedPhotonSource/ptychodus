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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ptychodus.model.observer import Observable, Observer
from ptychodus.model.parameters import StringParameter

from ...model.agent import (
    AgentSettings,
    ChatMessage,
    ChatTerminal,
    ConversationObserver,
    ConversationRepository,
    LanguageModelProviderOptions,
    LanguageModelProviderSelection,
    ModelCatalog,
    ThinkingLevel,
)
from ...view.agent import AgentChatView, AgentInputView, AgentView
from ..parameters import (
    ComboBoxParameterViewController,
    LineEditParameterViewController,
    ParameterViewController,
    SpinBoxParameterViewController,
    create_parameter_view_controller,
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
    """Editable combo box bound to a StringParameter, with a Refresh button that repopulates
    the items. Editable so a value can be typed when the items cannot be fetched."""

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
        self._combo.setEditable(True)
        self._combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
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
        line_edit = self._combo.lineEdit()

        if line_edit is not None:
            line_edit.editingFinished.connect(self._handle_editing_finished)

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

    def _handle_editing_finished(self) -> None:
        self._parameter.set_value(self._combo.currentText().strip())

    def _handle_refresh_clicked(self) -> None:
        self.populate(self._on_refresh())

    def __sync_model_to_view(self) -> None:
        self._combo.setCurrentText(self._parameter.get_value())

    def _update(self, observable: Observable) -> None:
        if observable is self._parameter:
            self.__sync_model_to_view()


class _ProviderOptionsViewController:
    """The options a provider declared, and its model, as one form."""

    def __init__(self, options: LanguageModelProviderOptions, catalog: ModelCatalog) -> None:
        self._view_controllers: list[ParameterViewController] = []
        layout = QFormLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        for description in options.describe_parameters():
            view_controller = create_parameter_view_controller(
                options.parameters()[description.key],
                name=description.name,
                label=description.label,
                tool_tip=description.tool_tip,
                choices=description.choices,
                unit=description.unit,
                display_unit=description.display_unit,
            )

            if view_controller is not None:
                self._view_controllers.append(view_controller)
                layout.addRow(description.label, view_controller.get_widget())

        self.model_view_controller = _RefreshableComboBoxParameterViewController(
            options.model,
            catalog.refresh,
            tool_tip=(
                f'The {options.plugin.display_name} model ID. Click Refresh to re-fetch the '
                'list, or type an ID when it cannot be fetched.'
            ),
        )
        layout.addRow('Model:', self.model_view_controller.get_widget())

        self.widget = QWidget()
        self.widget.setLayout(layout)


class _ShowEventFilter(QObject):
    def __init__(self, on_show: Callable[[], None]) -> None:
        super().__init__()
        self._on_show = on_show

    def eventFilter(self, a0: QObject, a1: QEvent) -> bool:  # noqa: N802
        if a1.type() == QEvent.Type.Show:
            self._on_show()
        return super().eventFilter(a0, a1)


class AgentController(Observer):
    def __init__(
        self,
        settings: AgentSettings,
        selection: LanguageModelProviderSelection,
        catalog: ModelCatalog,
        view: AgentView,
    ) -> None:
        super().__init__()
        self._settings = settings
        self._selection = selection
        self._catalog = catalog
        self._view = view
        self._models_loaded: set[str] = set()

        self._provider_view_controller = ComboBoxParameterViewController(
            selection.provider,
            selection.provider.choices(),
            tool_tip='The service hosting the chat models.',
        )
        self._provider_stack = QStackedWidget()
        self._provider_option_view_controllers: dict[str, _ProviderOptionsViewController] = {}

        for options in selection:
            view_controller = _ProviderOptionsViewController(options, catalog)
            self._provider_option_view_controllers[options.plugin.simple_name] = view_controller
            self._provider_stack.addWidget(view_controller.widget)

        self._instructions_view_controller = LineEditParameterViewController(
            settings.instructions,
            tool_tip='Instructions sent to the model with every message.',
        )
        self._thinking_view_controller = ComboBoxParameterViewController(
            settings.thinking,
            [level.value for level in ThinkingLevel],
            tool_tip=(
                'Reasoning effort. Default leaves it to the model; models without a matching '
                'level use the closest one.'
            ),
        )
        self._max_tokens_view_controller = SpinBoxParameterViewController(
            settings.max_tokens,
            tool_tip='Maximum number of tokens in each response; 0 leaves it to the model.',
        )
        self._mcp_server_url_view_controller = LineEditParameterViewController(
            settings.mcp_server_url,
            tool_tip=(
                'Optional MCP server whose tools the agent may call (e.g. ptychodus-webapp at '
                'http://localhost:8000/mcp). Must be https:// or loopback http://.'
            ),
        )

        provider_layout = QFormLayout()
        provider_layout.addRow('Provider:', self._provider_view_controller.get_widget())
        provider_layout.addRow(self._provider_stack)

        provider_group_box = QGroupBox('Provider')
        provider_group_box.setLayout(provider_layout)

        agent_layout = QFormLayout()
        agent_layout.addRow('Instructions:', self._instructions_view_controller.get_widget())
        agent_layout.addRow('Thinking:', self._thinking_view_controller.get_widget())
        agent_layout.addRow('Max Tokens:', self._max_tokens_view_controller.get_widget())
        agent_layout.addRow('MCP Server URL:', self._mcp_server_url_view_controller.get_widget())

        agent_group_box = QGroupBox('Agent')
        agent_group_box.setLayout(agent_layout)

        layout = QVBoxLayout()
        layout.addWidget(provider_group_box)
        layout.addWidget(agent_group_box)
        layout.addStretch()
        view.setLayout(layout)

        self._show_event_filter = _ShowEventFilter(
            lambda: self._sync_provider_to_view(load_models=True)
        )
        view.installEventFilter(self._show_event_filter)
        selection.add_observer(self)
        self._sync_provider_to_view(load_models=False)

    def _sync_provider_to_view(self, *, load_models: bool) -> None:
        """Show the selected provider's form, fetching its models the first time it is
        shown while the panel is visible, so a hidden panel never touches the network."""
        try:
            name = self._selection.get_current().plugin.simple_name
        except LookupError:
            return

        view_controller = self._provider_option_view_controllers[name]
        self._provider_stack.setCurrentWidget(view_controller.widget)

        if load_models and name not in self._models_loaded:
            self._models_loaded.add(name)
            view_controller.model_view_controller.populate(self._catalog.get_available_models())

    def _update(self, observable: Observable) -> None:
        if observable is self._selection:
            self._sync_provider_to_view(load_models=self._view.isVisible())
