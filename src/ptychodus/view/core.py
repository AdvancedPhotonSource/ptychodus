from __future__ import annotations
from dataclasses import dataclass
import logging

from PyQt5.QtCore import (
    PYQT_VERSION_STR,
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
from .diffraction import DiffractionView, DiffractionImageView
from .fluorescence import FluorescenceView
from .image import ImageView
from .product import ProductRightPanelView, ProductView, ProductVisualizationView
from .processing import ProcessingStatusView
from .repository import RepositoryTableView, RepositoryTreeView
from .probe_positions import ProbePositionsPlotView
from .settings import SettingsView

logger = logging.getLogger(__name__)


# A child button matches the panels above it in shape, so the rail down the left of
# its group, the indent, and the smaller icon are all that mark the two levels apart.
_CHILD_RAIL_PX = 3
_CHILD_INDENT_PX = 14

_SUBVIEW_GROUP_STYLE = (
    '_SubviewGroupContainer {'
    f'    border-left: {_CHILD_RAIL_PX}px solid palette(dark);'
    '}'
    '_SubviewGroupContainer QToolButton { background: transparent; border: none; }'
    '_SubviewGroupContainer QToolButton:hover { background-color: palette(midlight); }'
    '_SubviewGroupContainer QToolButton:checked {'
    '    background-color: palette(highlight);'
    '    color: palette(highlighted-text);'
    '}'
)


class _SubviewGroupContainer(QWidget):
    """The child buttons of one navigation panel, indented beneath it."""

    def __init__(
        self,
        child_actions: tuple[QAction, ...],
        *,
        icon_size: QSize,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # A plain QWidget paints no stylesheet border without this, so the rail in
        # _SUBVIEW_GROUP_STYLE would not be drawn.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet(_SUBVIEW_GROUP_STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(_CHILD_INDENT_PX, 0, 0, 0)
        layout.setSpacing(0)

        self._buttons: dict[QAction, QToolButton] = {}
        for action in child_actions:
            btn = QToolButton(self)
            btn.setDefaultAction(action)
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            btn.setIconSize(icon_size)
            layout.addWidget(btn)
            self._buttons[action] = btn

    def child_button(self, action: QAction) -> QToolButton | None:
        return self._buttons.get(action)

    def set_child_button_visible(self, action: QAction, visible: bool) -> None:
        btn = self._buttons.get(action)
        if btn is not None:
            btn.setVisible(visible)


@dataclass(frozen=True)
class NavigationSubviewGroup:
    child_actions: tuple[QAction, ...]
    container: _SubviewGroupContainer


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
        child_actions: tuple[QAction, ...],
        *,
        insert_before: QAction,
        child_icon_size: QSize,
    ) -> NavigationSubviewGroup:
        """Re-home panels already added by add_panel as children of the panel above."""
        for child in child_actions:
            self.tool_bar.removeAction(child)
        container = _SubviewGroupContainer(child_actions, icon_size=child_icon_size)
        self.tool_bar.insertWidget(insert_before, container)
        group = NavigationSubviewGroup(child_actions=child_actions, container=container)
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

        if not top_level_buttons and not child_buttons:
            return

        # Children sit behind the rail and one indent in, so they need that much less
        # width to line their right edges up with the top-level buttons.
        child_inset = _CHILD_RAIL_PX + _CHILD_INDENT_PX
        target_width = max(
            max((btn.sizeHint().width() for btn in top_level_buttons), default=0),
            max((btn.sizeHint().width() + child_inset for btn in child_buttons), default=0),
        )
        for btn in top_level_buttons:
            btn.setFixedWidth(target_width)
        for btn in child_buttons:
            btn.setFixedWidth(target_width - child_inset)
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

        self.diffraction_view = DiffractionView()
        self.diffraction_image_view = DiffractionImageView()
        self.diffraction_action = self.navigation.add_panel(
            QIcon(':/icons/patterns'),
            'Diffraction',
            left=self.diffraction_view,
            right=self.diffraction_image_view,
        )

        self.product_view = ProductView()

        if is_developer_mode_enabled:
            self.product_visualization_view = ProductVisualizationView()
            self.product_right_view = ProductRightPanelView(self.product_visualization_view)
        else:
            self.product_right_view = ProductRightPanelView()

        self.product_action = self.navigation.add_panel(
            QIcon(':/icons/products'),
            'Products',
            left=self.product_view,
            right=self.product_right_view,
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

        self.fluorescence_view = FluorescenceView()
        self.fluorescence_image_view = ImageView()
        self.fluorescence_action = self.navigation.add_panel(
            QIcon(':/icons/fluorescence'),
            'Fluorescence',
            left=self.fluorescence_view,
            right=self.fluorescence_image_view,
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

        # Children of Products.
        self.navigation.add_subview_group(
            child_actions=(
                self.positions_action,
                self.probe_action,
                self.object_action,
            ),
            insert_before=self.processing_action,
            child_icon_size=QSize(24, 24),
        )
        # Children of Processing.
        self.navigation.add_subview_group(
            child_actions=(self.globus_action, self.genesis_action, self.automation_action),
            insert_before=self.fluorescence_action,
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
