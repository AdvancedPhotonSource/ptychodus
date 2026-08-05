from __future__ import annotations
from dataclasses import dataclass
import logging

from PyQt5.QtCore import (
    PYQT_VERSION_STR,
    QEasingCurve,
    QPropertyAnimation,
    QSize,
    QT_VERSION_STR,
    Qt,
)
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QApplication,
    QLCDNumber,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableView,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from . import resources  # noqa
from .agent import AgentView, AgentChatView
from .diffraction import DatasetsView, DiffractionImageView
from .fluorescence import FluorescenceImageView, FluorescenceView
from .image import ImageView
from .product import ProductView, ProductVisualizationView
from .processing import ProcessingStatusView
from .repository import RepositoryTableView, RepositoryTreeView
from .probe_positions import ProbePositionsPlotView
from .settings import SettingsView

logger = logging.getLogger(__name__)


_SUBVIEW_GROUP_STYLE = (
    '_SubviewGroupContainer { background-color: palette(mid); }'
    '_SubviewGroupContainer QToolButton { background: transparent; border: none; }'
    '_SubviewGroupContainer QToolButton:hover { background-color: palette(midlight); }'
    '_SubviewGroupContainer QToolButton:checked {'
    '    background-color: palette(highlight);'
    '    color: palette(highlighted-text);'
    '}'
)

_EXPAND_COLLAPSE_DURATION_MS = 180


class _SubviewGroupContainer(QWidget):
    def __init__(
        self,
        child_actions: tuple[QAction, ...],
        *,
        icon_size: QSize,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setStyleSheet(_SUBVIEW_GROUP_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._buttons: dict[QAction, QToolButton] = {}
        for action in child_actions:
            btn = QToolButton(self)
            btn.setDefaultAction(action)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setIconSize(icon_size)
            layout.addWidget(btn)
            self._buttons[action] = btn

        layout.activate()
        self._cached_natural_height = layout.sizeHint().height()

        self.setMaximumHeight(0)
        self._expanded = False

        self._animation = QPropertyAnimation(self, b'maximumHeight', self)
        self._animation.setDuration(_EXPAND_COLLAPSE_DURATION_MS)
        self._animation.setEasingCurve(QEasingCurve.Type.InOutQuad)

    def child_button(self, action: QAction) -> QToolButton | None:
        return self._buttons.get(action)

    def set_child_button_visible(self, action: QAction, visible: bool) -> None:
        btn = self._buttons.get(action)
        if btn is not None:
            btn.setVisible(visible)

    def set_expanded(self, expanded: bool, *, animated: bool = True) -> None:
        running = self._animation.state() == QPropertyAnimation.State.Running
        if expanded == self._expanded and not running:
            return
        self._expanded = expanded
        self._animation.stop()
        target = self._cached_natural_height if expanded else 0
        if not animated:
            self.setMaximumHeight(target)
            return
        self._animation.setStartValue(self.maximumHeight())
        self._animation.setEndValue(target)
        self._animation.start()


@dataclass(frozen=True)
class NavigationSubviewGroup:
    parent_action: QAction
    child_actions: tuple[QAction, ...]
    container: _SubviewGroupContainer
    top_separator: QAction
    bottom_separator: QAction


class NavigationPanel:
    def __init__(self) -> None:
        self.tool_bar = QToolBar()
        self.action_group = QActionGroup(self.tool_bar)
        self.left_stack = QStackedWidget()
        self.right_stack = QStackedWidget()
        self.top_level_actions: list[QAction] = []
        self.subview_groups: list[NavigationSubviewGroup] = []

    def add_panel(self, icon: QIcon, label: str, *, left: QWidget, right: QWidget) -> QAction:
        index = self.left_stack.count()
        assert index == self.right_stack.count()
        action = self.tool_bar.addAction(icon, label)
        action.setCheckable(True)
        action.setData(index)
        self.action_group.addAction(action)
        self.left_stack.addWidget(left)
        self.right_stack.addWidget(right)
        self.top_level_actions.append(action)
        return action

    def add_subview_group(
        self,
        parent_action: QAction,
        child_actions: tuple[QAction, ...],
        *,
        insert_before: QAction,
        child_icon_size: QSize,
    ) -> NavigationSubviewGroup:
        for child in child_actions:
            self.tool_bar.removeAction(child)
        top_separator = self.tool_bar.insertSeparator(insert_before)
        container = _SubviewGroupContainer(child_actions, icon_size=child_icon_size)
        self.tool_bar.insertWidget(insert_before, container)
        bottom_separator = self.tool_bar.insertSeparator(insert_before)
        top_separator.setVisible(False)
        bottom_separator.setVisible(False)
        group = NavigationSubviewGroup(
            parent_action=parent_action,
            child_actions=child_actions,
            container=container,
            top_separator=top_separator,
            bottom_separator=bottom_separator,
        )
        self.subview_groups.append(group)
        return group

    def set_current_index(self, index: int) -> None:
        self.left_stack.setCurrentIndex(index)
        self.right_stack.setCurrentIndex(index)

    def normalize_button_widths(self) -> None:
        top_level_buttons: list[QToolButton] = []
        for action in self.top_level_actions:
            widget = self.tool_bar.widgetForAction(action)
            if isinstance(widget, QToolButton):
                top_level_buttons.append(widget)

        child_buttons: list[QToolButton] = []
        for group in self.subview_groups:
            for action in group.child_actions:
                btn = group.container.child_button(action)
                if btn is not None:
                    child_buttons.append(btn)

        all_buttons = top_level_buttons + child_buttons
        if not all_buttons:
            return

        target_width = max(btn.sizeHint().width() for btn in all_buttons)
        for btn in all_buttons:
            btn.setFixedWidth(target_width)
        for group in self.subview_groups:
            group.container.setFixedWidth(target_width)


class ViewCore(QMainWindow):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        is_developer_mode_enabled: bool = False,
    ) -> None:
        super().__init__(parent)

        logger.info(f'PyQt {PYQT_VERSION_STR}')
        logger.info(f'Qt {QT_VERSION_STR}')

        self.navigation = NavigationPanel()
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.memory_widget = QLCDNumber()

        self.settings_view = SettingsView()
        self.settings_table_view = QTableView()
        self.settings_action = self.navigation.add_panel(
            QIcon(':/icons/settings'),
            'Settings',
            left=self.settings_view,
            right=self.settings_table_view,
        )

        self.datasets_view = DatasetsView()
        self.diffraction_image_view = DiffractionImageView()
        self.datasets_action = self.navigation.add_panel(
            QIcon(':/icons/patterns'),
            'Diffraction',
            left=self.datasets_view,
            right=self.diffraction_image_view,
        )

        self.product_view = ProductView()
        if is_developer_mode_enabled:
            self.product_visualization_view = ProductVisualizationView()
            product_right: QWidget = self.product_visualization_view
        else:
            product_right = QWidget()
        self.product_action = self.navigation.add_panel(
            QIcon(':/icons/products'),
            'Products',
            left=self.product_view,
            right=product_right,
        )

        self.probe_positions_view = RepositoryTableView()
        self.probe_positions_plot_view = ProbePositionsPlotView()
        self.positions_action = self.navigation.add_panel(
            QIcon(':/icons/positions'),
            'Positions',
            left=self.probe_positions_view,
            right=self.probe_positions_plot_view,
        )

        self.probe_view = RepositoryTreeView()
        self.probe_image_view = ImageView()
        self.probe_action = self.navigation.add_panel(
            QIcon(':/icons/probe'),
            'Probe',
            left=self.probe_view,
            right=self.probe_image_view,
        )

        self.object_view = RepositoryTreeView()
        self.object_image_view = ImageView()
        self.object_action = self.navigation.add_panel(
            QIcon(':/icons/object'),
            'Object',
            left=self.object_view,
            right=self.object_image_view,
        )

        self.fluorescence_view = FluorescenceView()
        self.fluorescence_image_view = FluorescenceImageView()
        self.fluorescence_action = self.navigation.add_panel(
            QIcon(':/icons/fluorescence'),
            'Fluorescence',
            left=self.fluorescence_view,
            right=self.fluorescence_image_view,
        )

        self.processing_view = QWidget()
        self.processing_status_view = ProcessingStatusView()
        self.processing_action = self.navigation.add_panel(
            QIcon(':/icons/processing'),
            'Processing',
            left=self.processing_view,
            right=self.processing_status_view,
        )

        self.globus_view = QWidget()
        self.globus_status_view = QTableView()
        self.globus_action = self.navigation.add_panel(
            QIcon(':/icons/globus'),
            'Globus',
            left=self.globus_view,
            right=self.globus_status_view,
        )

        self.genesis_view = QWidget()
        self.genesis_status_view = QTableView()
        self.genesis_action = self.navigation.add_panel(
            QIcon(':/icons/genesis'),
            'Genesis',
            left=self.genesis_view,
            right=self.genesis_status_view,
        )

        self.automation_view = QWidget()
        self.automation_widget = QWidget()
        self.automation_action = self.navigation.add_panel(
            QIcon(':/icons/automate'),
            'Automation',
            left=self.automation_view,
            right=self.automation_widget,
        )

        self.agent_view = AgentView()
        self.agent_chat_view = AgentChatView()
        self.agent_action = self.navigation.add_panel(
            QIcon(':/icons/sparkles'),
            'Agent',
            left=self.agent_view,
            right=self.agent_chat_view,
        )

        #####

        self.setWindowIcon(QIcon(':/icons/ptychodus'))

        self.navigation.tool_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.navigation.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.navigation.tool_bar.setIconSize(QSize(32, 32))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.navigation.tool_bar)

        self.navigation.add_subview_group(
            parent_action=self.product_action,
            child_actions=(
                self.positions_action,
                self.probe_action,
                self.object_action,
                self.fluorescence_action,
            ),
            insert_before=self.processing_action,
            child_icon_size=QSize(24, 24),
        )
        self.navigation.add_subview_group(
            parent_action=self.processing_action,
            child_actions=(self.globus_action, self.genesis_action, self.automation_action),
            insert_before=self.agent_action,
            child_icon_size=QSize(24, 24),
        )

        self.navigation.normalize_button_widths()

        self.navigation.left_stack.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.splitter.addWidget(self.navigation.left_stack)

        self.navigation.right_stack.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )
        self.splitter.addWidget(self.navigation.right_stack)

        self.setCentralWidget(self.splitter)

        application_desktop = QApplication.desktop()

        if application_desktop is not None:
            desktop_size = application_desktop.availableGeometry().size()
            preferred_height = desktop_size.height() * 2 // 3
            preferred_width = min(desktop_size.width() * 2 // 3, 2 * preferred_height)
            self.resize(preferred_width, preferred_height)

        status_bar = self.statusBar()

        if status_bar is not None:
            status_bar.addPermanentWidget(self.memory_widget)
