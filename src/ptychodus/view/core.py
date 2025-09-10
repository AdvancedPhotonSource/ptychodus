from __future__ import annotations
import logging

from PyQt5.QtCore import PYQT_VERSION_STR, QSize, QT_VERSION_STR, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QActionGroup,
    QApplication,
    QLCDNumber,
    QMainWindow,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableView,
    QToolBar,
    QWidget,
)

from . import resources  # noqa
from .agent import AgentView, AgentChatView
from .automation import AutomationView
from .image import ImageView
from .diffraction import PatternsView
from .product import ProductView
from .reconstructor import ReconstructorView, ReconstructorPlotView
from .repository import RepositoryTableView, RepositoryTreeView
from .scan import ScanPlotView
from .settings import SettingsView
from .workflow import WorkflowParametersView

logger = logging.getLogger(__name__)


class ViewCore(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        logger.info(f'PyQt {PYQT_VERSION_STR}')
        logger.info(f'Qt {QT_VERSION_STR}')

        self.navigation_tool_bar = QToolBar()
        self.navigation_action_group = QActionGroup(self.navigation_tool_bar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = QStackedWidget()
        self.right_panel = QStackedWidget()
        self.memory_widget = QLCDNumber()

        self.settings_action = self.navigation_tool_bar.addAction(
            QIcon(':/icons/settings'), 'Settings'
        )
        self.settings_view = SettingsView()
        self.settings_table_view = QTableView()

        self.patterns_action = self.navigation_tool_bar.addAction(
            QIcon(':/icons/patterns'), 'Patterns'
        )
        self.patterns_view = PatternsView()
        self.patterns_image_view = ImageView()

        self.product_action = self.navigation_tool_bar.addAction(
            QIcon(':/icons/products'), 'Products'
        )
        self.product_view = ProductView()
        self.product_diagram_view = QWidget()

        self.scan_action = self.navigation_tool_bar.addAction(QIcon(':/icons/scan'), 'Positions')
        self.scan_view = RepositoryTableView()
        self.scan_plot_view = ScanPlotView.create_instance()

        self.probe_action = self.navigation_tool_bar.addAction(QIcon(':/icons/probe'), 'Probe')
        self.probe_view = RepositoryTreeView()
        self.probe_image_view = ImageView()

        self.object_action = self.navigation_tool_bar.addAction(QIcon(':/icons/object'), 'Object')
        self.object_view = RepositoryTreeView()
        self.object_image_view = ImageView()

        self.reconstructor_action = self.navigation_tool_bar.addAction(
            QIcon(':/icons/reconstructor'), 'Reconstructor'
        )
        self.reconstructor_view = ReconstructorView()
        self.reconstructor_plot_view = ReconstructorPlotView()

        self.workflow_action = self.navigation_tool_bar.addAction(
            QIcon(':/icons/workflow'), 'Workflow'
        )
        self.workflow_parameters_view = WorkflowParametersView.create_instance()
        self.workflow_table_view = QTableView()

        self.automation_action = self.navigation_tool_bar.addAction(
            QIcon(':/icons/automate'), 'Automation'
        )
        self.automation_view = AutomationView.create_instance()
        self.automation_widget = QWidget()

        self.agent_action = self.navigation_tool_bar.addAction(
            QIcon(':/icons/sparkles'),
            'Agent',
        )
        self.agent_view = AgentView()
        self.agent_chat_view = AgentChatView()

        #####

        self.setWindowIcon(QIcon(':/icons/ptychodus'))

        self.navigation_tool_bar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.navigation_tool_bar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.navigation_tool_bar.setIconSize(QSize(32, 32))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.navigation_tool_bar)

        for index, action in enumerate(self.navigation_tool_bar.actions()):
            action.setCheckable(True)
            action.setData(index)
            self.navigation_action_group.addAction(action)

        # maintain same order as navigationToolBar buttons
        self.left_panel.addWidget(self.settings_view)
        self.left_panel.addWidget(self.patterns_view)
        self.left_panel.addWidget(self.product_view)
        self.left_panel.addWidget(self.scan_view)
        self.left_panel.addWidget(self.probe_view)
        self.left_panel.addWidget(self.object_view)
        self.left_panel.addWidget(self.reconstructor_view)
        self.left_panel.addWidget(self.workflow_parameters_view)
        self.left_panel.addWidget(self.automation_view)
        self.left_panel.addWidget(self.agent_view)
        self.left_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.splitter.addWidget(self.left_panel)

        # maintain same order as navigationToolBar buttons
        self.right_panel.addWidget(self.settings_table_view)
        self.right_panel.addWidget(self.patterns_image_view)
        self.right_panel.addWidget(self.product_diagram_view)
        self.right_panel.addWidget(self.scan_plot_view)
        self.right_panel.addWidget(self.probe_image_view)
        self.right_panel.addWidget(self.object_image_view)
        self.right_panel.addWidget(self.reconstructor_plot_view)
        self.right_panel.addWidget(self.workflow_table_view)
        self.right_panel.addWidget(self.automation_widget)
        self.right_panel.addWidget(self.agent_chat_view)
        self.right_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.splitter.addWidget(self.right_panel)

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
