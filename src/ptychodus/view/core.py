from __future__ import annotations
from dataclasses import dataclass
import logging

from PyQt5.QtCore import PYQT_VERSION_STR, QSize, QT_VERSION_STR, Qt
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
    QWidget,
)

from . import resources  # noqa
from .agent import AgentView, AgentChatView
from .diffraction import PatternsImageView, PatternsView
from .image import ImageView
from .product import ProductView, ProductVisualizationView
from .processing import ProcessingStatusView
from .repository import RepositoryTableView, RepositoryTreeView
from .probe_positions import ProbePositionsPlotView
from .settings import SettingsView

logger = logging.getLogger(__name__)


_SUBVIEW_BUTTON_STYLE = (
    'QToolButton { padding-left: 12px; background-color: palette(mid); }'
    'QToolButton:hover { background-color: palette(midlight); }'
    'QToolButton:checked {'
    '    background-color: palette(highlight);'
    '    color: palette(highlighted-text);'
    '}'
)


@dataclass(frozen=True)
class NavigationSubviewGroup:
    parent_action: QAction
    child_actions: tuple[QAction, ...]
    top_separator: QAction
    bottom_separator: QAction


class NavigationPanel:
    def __init__(self) -> None:
        self.tool_bar = QToolBar()
        self.action_group = QActionGroup(self.tool_bar)
        self.left_stack = QStackedWidget()
        self.right_stack = QStackedWidget()
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
        return action

    def add_subview_group(
        self,
        parent_action: QAction,
        child_actions: tuple[QAction, ...],
        *,
        top_separator_before: QAction,
        bottom_separator_before: QAction,
    ) -> NavigationSubviewGroup:
        group = NavigationSubviewGroup(
            parent_action=parent_action,
            child_actions=child_actions,
            top_separator=self.tool_bar.insertSeparator(top_separator_before),
            bottom_separator=self.tool_bar.insertSeparator(bottom_separator_before),
        )
        self.subview_groups.append(group)
        return group

    def set_current_index(self, index: int) -> None:
        self.left_stack.setCurrentIndex(index)
        self.right_stack.setCurrentIndex(index)


class ViewCore(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
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

        self.patterns_view = PatternsView()
        self.patterns_image_view = PatternsImageView()
        self.patterns_action = self.navigation.add_panel(
            QIcon(':/icons/patterns'),
            'Patterns',
            left=self.patterns_view,
            right=self.patterns_image_view,
        )

        self.product_view = ProductView()
        self.product_visualization_view = ProductVisualizationView()
        self.product_action = self.navigation.add_panel(
            QIcon(':/icons/products'),
            'Products',
            left=self.product_view,
            right=self.product_visualization_view,
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

        self.agent_view = AgentView()
        self.agent_chat_view = AgentChatView()
        self.agent_action = self.navigation.add_panel(
            QIcon(':/icons/sparkles'),
            'Agent',
            left=self.agent_view,
            right=self.agent_chat_view,
        )

        self.navigation.add_subview_group(
            parent_action=self.product_action,
            child_actions=(self.positions_action, self.probe_action, self.object_action),
            top_separator_before=self.positions_action,
            bottom_separator_before=self.processing_action,
        )
        self.navigation.add_subview_group(
            parent_action=self.processing_action,
            child_actions=(self.globus_action, self.genesis_action, self.automation_action),
            top_separator_before=self.globus_action,
            bottom_separator_before=self.agent_action,
        )

        #####

        self.setWindowIcon(QIcon(':/icons/ptychodus'))

        self.navigation.tool_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.navigation.tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.navigation.tool_bar.setIconSize(QSize(32, 32))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.navigation.tool_bar)

        for group in self.navigation.subview_groups:
            for action in group.child_actions:
                btn = self.navigation.tool_bar.widgetForAction(action)
                if isinstance(btn, QToolButton):
                    btn.setIconSize(QSize(24, 24))
                    btn.setStyleSheet(_SUBVIEW_BUTTON_STYLE)

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
