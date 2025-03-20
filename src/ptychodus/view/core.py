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
        self.navigationActionGroup = QActionGroup(self.navigationToolBar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = QStackedWidget()
        self.right_panel = QStackedWidget()
        self.memory_widget = QLCDNumber()

        self.settingsAction = self.navigationToolBar.addAction(
            QIcon(':/icons/settings'), 'Settings'
        )
        self.settingsView = SettingsView.create_instance()
        self.settingsTableView = QTableView()

        self.patternsAction = self.navigationToolBar.addAction(
            QIcon(':/icons/patterns'), 'Patterns'
        )
        self.patternsView = PatternsView()
        self.patternsImageView = ImageView.create_instance()

        self.productAction = self.navigationToolBar.addAction(QIcon(':/icons/products'), 'Products')
        self.productView = ProductView()
        self.productDiagramView = QWidget()

        self.scanAction = self.navigationToolBar.addAction(QIcon(':/icons/scan'), 'Positions')
        self.scanView = RepositoryTableView()
        self.scanPlotView = ScanPlotView.create_instance()

        self.probeAction = self.navigationToolBar.addAction(QIcon(':/icons/probe'), 'Probe')
        self.probeView = RepositoryTreeView()
        self.probeImageView = ImageView.create_instance()

        self.objectAction = self.navigationToolBar.addAction(QIcon(':/icons/object'), 'Object')
        self.objectView = RepositoryTreeView()
        self.objectImageView = ImageView.create_instance()

        self.reconstructorAction = self.navigationToolBar.addAction(
            QIcon(':/icons/reconstructor'), 'Reconstructor'
        )
        self.reconstructorView = ReconstructorView()
        self.reconstructorPlotView = ReconstructorPlotView()

        self.workflowAction = self.navigationToolBar.addAction(
            QIcon(':/icons/workflow'), 'Workflow'
        )
        self.workflowParametersView = WorkflowParametersView.create_instance()
        self.workflowTableView = QTableView()

        self.automationAction = self.navigationToolBar.addAction(
            QIcon(':/icons/automate'), 'Automation'
        )
        self.automationView = AutomationView.create_instance()
        self.automationWidget = QWidget()

        self.agentAction = self.navigationToolBar.addAction(
            QIcon(':/icons/sparkles'),
            'Agent',
        )
        self.agentView = AgentView()
        self.agentChatView = AgentChatView()

        #####

        self.setWindowIcon(QIcon(':/icons/ptychodus'))

        self.navigationToolBar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self.navigationToolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        self.navigationToolBar.setIconSize(QSize(32, 32))
        self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.navigationToolBar)

        for index, action in enumerate(self.navigationToolBar.actions()):
            action.setCheckable(True)
            action.setData(index)
            self.navigationActionGroup.addAction(action)

        # maintain same order as navigationToolBar buttons
        self.left_panel.addWidget(self.settingsView)
        self.left_panel.addWidget(self.patternsView)
        self.left_panel.addWidget(self.productView)
        self.left_panel.addWidget(self.scanView)
        self.left_panel.addWidget(self.probeView)
        self.left_panel.addWidget(self.objectView)
        self.left_panel.addWidget(self.reconstructorView)
        self.left_panel.addWidget(self.workflowParametersView)
        self.left_panel.addWidget(self.automationView)
        self.left_panel.addWidget(self.agentView)
        self.left_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.splitter.addWidget(self.left_panel)

        # maintain same order as navigationToolBar buttons
        self.right_panel.addWidget(self.settingsTableView)
        self.right_panel.addWidget(self.patternsImageView)
        self.right_panel.addWidget(self.productDiagramView)
        self.right_panel.addWidget(self.scanPlotView)
        self.right_panel.addWidget(self.probeImageView)
        self.right_panel.addWidget(self.objectImageView)
        self.right_panel.addWidget(self.reconstructorPlotView)
        self.right_panel.addWidget(self.workflowTableView)
        self.right_panel.addWidget(self.automationWidget)
        self.right_panel.addWidget(self.agentChatView)
        self.right_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        self.splitter.addWidget(self.right_panel)

        self.setCentralWidget(self.splitter)

        desktop_size = QApplication.desktop().availableGeometry().size()
        preferred_height = desktop_size.height() * 2 // 3
        preferred_width = min(desktop_size.width() * 2 // 3, 2 * preferred_height)
        self.resize(preferred_width, preferred_height)

        self.statusBar().addPermanentWidget(self.memory_widget)
