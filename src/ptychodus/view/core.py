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
from .patterns import PatternsView
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

        self.navigationToolBar = QToolBar()
        self.navigation_action_group = QActionGroup(self.navigationToolBar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = QStackedWidget()
        self.right_panel = QStackedWidget()
        self.memory_widget = QLCDNumber()

        self.settingsAction = self.navigationToolBar.addAction(
            QIcon(':/icons/settings'), 'Settings'
        )
        self.settings_view = SettingsView.create_instance()
        self.settings_table_view = QTableView()

        self.patterns_action = self.navigationToolBar.addAction(
            QIcon(':/icons/patterns'), 'Patterns'
        )
        self.patterns_view = PatternsView()
        self.patterns_image_view = ImageView.create_instance()

        self.productAction = self.navigationToolBar.addAction(QIcon(':/icons/products'), 'Products')
        self.product_view = ProductView()
        self.productDiagramView = QWidget()

        self.scanAction = self.navigationToolBar.addAction(QIcon(':/icons/scan'), 'Positions')
        self.scan_view = RepositoryTableView()
        self.scan_plot_view = ScanPlotView.create_instance()

        self.probeAction = self.navigationToolBar.addAction(QIcon(':/icons/probe'), 'Probe')
        self.probe_view = RepositoryTreeView()
        self.probe_image_view = ImageView.create_instance()

        self.objectAction = self.navigationToolBar.addAction(QIcon(':/icons/object'), 'Object')
        self.object_view = RepositoryTreeView()
        self.object_image_view = ImageView.create_instance()

        self.reconstructorAction = self.navigationToolBar.addAction(
            QIcon(':/icons/reconstructor'), 'Reconstructor'
        )
        self.reconstructor_view = ReconstructorView()
        self.reconstructor_plot_view = ReconstructorPlotView()

        self.workflow_action = self.navigationToolBar.addAction(
            QIcon(':/icons/workflow'), 'Workflow'
        )
        self.workflow_parameters_view = WorkflowParametersView.create_instance()
        self.workflow_table_view = QTableView()

        self.automationAction = self.navigationToolBar.addAction(
            QIcon(':/icons/automate'), 'Automation'
        )
        self.automation_view = AutomationView.create_instance()
        self.automationWidget = QWidget()

        self.agent_action = self.navigationToolBar.addAction(
            QIcon(':/icons/sparkles'),
            'Agent',
        )
        self.agent_view = AgentView()
        self.agent_chat_view = AgentChatView()

        #####

        self.setWindowIcon(QIcon(':/icons/ptychodus'))

        self.navigationToolBar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.navigationToolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.navigationToolBar.setIconSize(QSize(32, 32))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.navigationToolBar)

        for index, action in enumerate(self.navigationToolBar.actions()):
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
        self.right_panel.addWidget(self.productDiagramView)
        self.right_panel.addWidget(self.scan_plot_view)
        self.right_panel.addWidget(self.probe_image_view)
        self.right_panel.addWidget(self.object_image_view)
        self.right_panel.addWidget(self.reconstructor_plot_view)
        self.right_panel.addWidget(self.workflow_table_view)
        self.right_panel.addWidget(self.automationWidget)
        self.right_panel.addWidget(self.agent_chat_view)
        self.right_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.splitter.addWidget(self.right_panel)

        self.setCentralWidget(self.splitter)

        desktop_size = QApplication.desktop().availableGeometry().size()
        preferred_height = desktop_size.height() * 2 // 3
        preferred_width = min(desktop_size.width() * 2 // 3, 2 * preferred_height)
        self.resize(preferred_width, preferred_height)

        self.statusBar().addPermanentWidget(self.memory_widget)
