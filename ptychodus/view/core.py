from __future__ import annotations
import logging

from PyQt5.QtCore import PYQT_VERSION_STR, QSize, QT_VERSION_STR, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QActionGroup, QApplication, QMainWindow, QProgressBar, QSizePolicy,
                             QSplitter, QStackedWidget, QTableView, QToolBar, QWidget)

from . import resources  # noqa
from .automation import AutomationView
from .image import ImageView
from .patterns import PatternsView
from .product import ProductView
from .reconstructor import ReconstructorParametersView, ReconstructorPlotView
from .repository import RepositoryTableView, RepositoryTreeView
from .scan import ScanPlotView
from .settings import SettingsParametersView
from .workflow import WorkflowParametersView

logger = logging.getLogger(__name__)


class ViewCore(QMainWindow):

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)

        self.navigationToolBar = QToolBar()
        self.navigationActionGroup = QActionGroup(self.navigationToolBar)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.parametersWidget = QStackedWidget()
        self.contentsWidget = QStackedWidget()
        self.memoryProgressBar = QProgressBar()

        self.settingsAction = self.navigationToolBar.addAction(QIcon(':/icons/settings'),
                                                               'Settings')
        self.settingsParametersView = SettingsParametersView.createInstance()
        self.settingsEntryView = QTableView()

        self.patternsAction = self.navigationToolBar.addAction(QIcon(':/icons/patterns'),
                                                               'Patterns')
        self.patternsView = PatternsView.createInstance()
        self.patternsImageView = ImageView.createInstance()

        self.productAction = self.navigationToolBar.addAction(QIcon(':/icons/products'),
                                                              'Products')
        self.productView = ProductView.createInstance()
        self.productDiagramView = QWidget()

        self.scanAction = self.navigationToolBar.addAction(QIcon(':/icons/scan'), 'Scan')
        self.scanView = RepositoryTableView.createInstance()
        self.scanPlotView = ScanPlotView.createInstance()

        self.probeAction = self.navigationToolBar.addAction(QIcon(':/icons/probe'), 'Probe')
        self.probeView = RepositoryTreeView.createInstance()
        self.probeImageView = ImageView.createInstance()

        self.objectAction = self.navigationToolBar.addAction(QIcon(':/icons/object'), 'Object')
        self.objectView = RepositoryTreeView.createInstance()
        self.objectImageView = ImageView.createInstance()

        self.reconstructorAction = self.navigationToolBar.addAction(QIcon(':/icons/reconstructor'),
                                                                    'Reconstructor')
        self.reconstructorParametersView = ReconstructorParametersView.createInstance()
        self.reconstructorPlotView = ReconstructorPlotView.createInstance()

        self.workflowAction = self.navigationToolBar.addAction(QIcon(':/icons/workflow'),
                                                               'Workflow')
        self.workflowParametersView = WorkflowParametersView.createInstance()
        self.workflowTableView = QTableView()

        self.automationAction = self.navigationToolBar.addAction(QIcon(':/icons/automate'),
                                                                 'Automation')
        self.automationView = AutomationView.createInstance()
        self.automationWidget = QWidget()

    @classmethod
    def createInstance(cls,
                       isDeveloperModeEnabled: bool,
                       parent: QWidget | None = None) -> ViewCore:
        logger.info(f'PyQt {PYQT_VERSION_STR}')
        logger.info(f'Qt {QT_VERSION_STR}')

        view = cls(parent)
        view.navigationToolBar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        view.addToolBar(Qt.ToolBarArea.LeftToolBarArea, view.navigationToolBar)
        view.navigationToolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        view.navigationToolBar.setIconSize(QSize(32, 32))

        for index, action in enumerate(view.navigationToolBar.actions()):
            action.setCheckable(True)
            action.setData(index)
            view.navigationActionGroup.addAction(action)

        view.settingsAction.setChecked(True)

        # maintain same order as navigationToolBar buttons
        view.parametersWidget.addWidget(view.settingsParametersView)
        view.parametersWidget.addWidget(view.patternsView)
        view.parametersWidget.addWidget(view.productView)
        view.parametersWidget.addWidget(view.scanView)
        view.parametersWidget.addWidget(view.probeView)
        view.parametersWidget.addWidget(view.objectView)
        view.parametersWidget.addWidget(view.reconstructorParametersView)
        view.parametersWidget.addWidget(view.workflowParametersView)
        view.parametersWidget.addWidget(view.automationView)
        view.parametersWidget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        view.splitter.addWidget(view.parametersWidget)

        # maintain same order as navigationToolBar buttons
        view.contentsWidget.addWidget(view.settingsEntryView)
        view.contentsWidget.addWidget(view.patternsImageView)
        view.contentsWidget.addWidget(view.productDiagramView)
        view.contentsWidget.addWidget(view.scanPlotView)
        view.contentsWidget.addWidget(view.probeImageView)
        view.contentsWidget.addWidget(view.objectImageView)
        view.contentsWidget.addWidget(view.reconstructorPlotView)
        view.contentsWidget.addWidget(view.workflowTableView)
        view.contentsWidget.addWidget(view.automationWidget)
        view.contentsWidget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        view.splitter.addWidget(view.contentsWidget)

        view.setCentralWidget(view.splitter)

        # TODO make visible when complete
        view.automationAction.setVisible(isDeveloperModeEnabled)

        desktopSize = QApplication.desktop().availableGeometry().size()
        preferredHeight = desktopSize.height() * 2 // 3
        preferredWidth = min(desktopSize.width() * 2 // 3, 2 * preferredHeight)
        view.resize(preferredWidth, preferredHeight)

        view.memoryProgressBar.setSizePolicy(QSizePolicy.Policy.Minimum,
                                             QSizePolicy.Policy.Preferred)
        view.statusBar().addPermanentWidget(view.memoryProgressBar)
        view.statusBar().showMessage('Ready')

        return view
