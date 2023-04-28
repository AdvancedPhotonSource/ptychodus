from __future__ import annotations
from pathlib import Path
from typing import Optional
import logging

from PyQt5.QtCore import PYQT_VERSION_STR, QSize, QT_VERSION_STR, Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (QActionGroup, QApplication, QMainWindow, QMenu, QSizePolicy,
                             QSplitter, QStackedWidget, QTableView, QToolBar, QToolButton,
                             QTreeView, QWidget)

from . import resources
from .automation import *
from .data import DataParametersView
from .detector import *
from .image import ImageView
from .monitor import *
from .object import *
from .probe import *
from .reconstructor import *
from .scan import *
from .settings import *
from .workflow import *

logger = logging.getLogger(__name__)


class ViewCore(QMainWindow):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)

        self.navigationToolBar = QToolBar()
        self.navigationActionGroup = QActionGroup(self.navigationToolBar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.parametersWidget = QStackedWidget()
        self.contentsWidget = QStackedWidget()

        self.settingsAction = self.navigationToolBar.addAction(QIcon(':/icons/settings'),
                                                               'Settings')
        self.settingsParametersView = SettingsParametersView.createInstance()
        self.settingsEntryView = QTableView()

        self.dataAction = self.navigationToolBar.addAction(QIcon(':/icons/dataset'), 'Dataset')
        self.dataParametersView = DataParametersView.createInstance()
        self.dataTableView = QTableView()

        self.detectorAction = self.navigationToolBar.addAction(QIcon(':/icons/detector'),
                                                               'Detector')
        self.detectorParametersView = DetectorParametersView.createInstance()
        self.detectorImageView = ImageView.createInstance()

        self.scanAction = self.navigationToolBar.addAction(QIcon(':/icons/scan'), 'Scan')
        self.scanView = ScanView.createInstance()
        self.scanPlotView = ScanPlotView.createInstance()

        self.probeAction = self.navigationToolBar.addAction(QIcon(':/icons/probe'), 'Probe')
        self.probeView = ProbeView.createInstance()
        self.probeImageView = ImageView.createInstance()

        self.objectAction = self.navigationToolBar.addAction(QIcon(':/icons/object'), 'Object')
        self.objectView = ObjectView.createInstance()
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

        self.monitorAction = self.navigationToolBar.addAction(QIcon(':/icons/monitor'), 'Monitor')
        self.monitorProbeView = MonitorProbeView.createInstance()
        self.monitorObjectView = MonitorObjectView.createInstance()

    @classmethod
    def createInstance(cls,
                       isDeveloperModeEnabled: bool,
                       parent: Optional[QWidget] = None) -> ViewCore:
        logger.info(f'PyQt {PYQT_VERSION_STR}')
        logger.info(f'Qt {QT_VERSION_STR}')

        view = cls(parent)
        view.navigationToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        view.addToolBar(Qt.LeftToolBarArea, view.navigationToolBar)
        view.navigationToolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        view.navigationToolBar.setIconSize(QSize(32, 32))

        for index, action in enumerate(view.navigationToolBar.actions()):
            action.setCheckable(True)
            action.setData(index)
            view.navigationActionGroup.addAction(action)

        view.settingsAction.setChecked(True)

        # maintain same order as navigationToolBar buttons
        view.parametersWidget.addWidget(view.settingsParametersView)
        view.parametersWidget.addWidget(view.dataParametersView)
        view.parametersWidget.addWidget(view.detectorParametersView)
        view.parametersWidget.addWidget(view.scanView)
        view.parametersWidget.addWidget(view.probeView)
        view.parametersWidget.addWidget(view.objectView)
        view.parametersWidget.addWidget(view.reconstructorParametersView)
        view.parametersWidget.addWidget(view.workflowParametersView)
        view.parametersWidget.addWidget(view.automationView)
        view.parametersWidget.addWidget(view.monitorProbeView)
        view.parametersWidget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        view.splitter.addWidget(view.parametersWidget)

        # maintain same order as navigationToolBar buttons
        view.contentsWidget.addWidget(view.settingsEntryView)
        view.contentsWidget.addWidget(view.dataTableView)
        view.contentsWidget.addWidget(view.detectorImageView)
        view.contentsWidget.addWidget(view.scanPlotView)
        view.contentsWidget.addWidget(view.probeImageView)
        view.contentsWidget.addWidget(view.objectImageView)
        view.contentsWidget.addWidget(view.reconstructorPlotView)
        view.contentsWidget.addWidget(view.workflowTableView)
        view.contentsWidget.addWidget(view.automationWidget)
        view.contentsWidget.addWidget(view.monitorObjectView)
        view.contentsWidget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        view.splitter.addWidget(view.contentsWidget)

        view.setCentralWidget(view.splitter)

        # TODO make visible when complete
        view.automationAction.setVisible(isDeveloperModeEnabled)

        desktopSize = QApplication.desktop().availableGeometry().size()
        preferredHeight = desktopSize.height() * 2 // 3
        preferredWidth = min(desktopSize.width() * 2 // 3, 2 * preferredHeight)
        view.resize(preferredWidth, preferredHeight)
        view.statusBar().showMessage('Ready')  # TODO make better use of the statusBar

        return view
