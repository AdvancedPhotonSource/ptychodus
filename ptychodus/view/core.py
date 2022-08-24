from __future__ import annotations
from pathlib import Path
from typing import Optional

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (QActionGroup, QApplication, QHeaderView, QMainWindow, QMenu,
                             QSplitter, QStyle, QTableView, QToolBar, QToolButton, QTreeView)

from .data import DataParametersView
from .detector import *
from .monitor import *
from .object import *
from .probe import *
from .reconstructor import *
from .scan import *
from .settings import *
from .workflow import *


class ViewCore(QMainWindow):

    def __init__(self, parent: Optional[QWidget]) -> None:
        super().__init__(parent)

        pixmapi = getattr(QStyle, 'SP_FileIcon')
        fileIcon = self.style().standardIcon(pixmapi)

        self.navigationToolBar = QToolBar()
        self.navigationActionGroup = QActionGroup(self.navigationToolBar)

        self.splitter = QSplitter(Qt.Horizontal)
        self.parametersWidget = QStackedWidget()
        self.contentsWidget = QStackedWidget()

        self.settingsAction = self.navigationToolBar.addAction(fileIcon, 'Settings')
        self.settingsParametersView = SettingsParametersView.createInstance()
        self.settingsEntryView = QTableView()

        self.dataAction = self.navigationToolBar.addAction(fileIcon, 'Data')
        self.dataParametersView = DataParametersView.createInstance()
        self.dataTableView = QTableView()

        self.detectorAction = self.navigationToolBar.addAction(fileIcon, 'Detector')
        self.detectorParametersView = DetectorParametersView.createInstance()
        self.detectorImageView = ImageView.createInstance()

        self.scanAction = self.navigationToolBar.addAction(fileIcon, 'Scan')
        self.scanParametersView = ScanParametersView.createInstance()
        self.scanPlotView = ScanPlotView.createInstance()

        self.probeAction = self.navigationToolBar.addAction(fileIcon, 'Probe')
        self.probeParametersView = ProbeParametersView.createInstance()
        self.probeImageView = ImageView.createInstance()

        self.objectAction = self.navigationToolBar.addAction(fileIcon, 'Object')
        self.objectParametersView = ObjectParametersView.createInstance()
        self.objectImageView = ImageView.createInstance()

        self.reconstructorAction = self.navigationToolBar.addAction(fileIcon, 'Reconstructor')
        self.reconstructorParametersView = ReconstructorParametersView.createInstance()
        self.reconstructorPlotView = ReconstructorPlotView.createInstance()

        self.workflowAction = self.navigationToolBar.addAction(fileIcon, 'Workflow')
        self.workflowParametersView = WorkflowParametersView.createInstance()
        self.workflowTableView = QTableView()

        self.monitorAction = self.navigationToolBar.addAction(fileIcon, 'Monitor')
        self.monitorProbeView = MonitorProbeView.createInstance()
        self.monitorObjectView = MonitorObjectView.createInstance()

    @classmethod
    def createInstance(cls,
                       isDeveloperModeEnabled: bool,
                       parent: Optional[QWidget] = None) -> ViewCore:
        view = cls(parent)

        view.navigationToolBar.setContextMenuPolicy(Qt.PreventContextMenu)
        view.addToolBar(Qt.LeftToolBarArea, view.navigationToolBar)
        view.navigationToolBar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)

        for index, action in enumerate(view.navigationToolBar.actions()):
            action.setCheckable(True)
            action.setData(index)
            view.navigationActionGroup.addAction(action)

        view.settingsAction.setChecked(True)

        view.settingsEntryView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        view.workflowTableView.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)

        # maintain same order as navigationToolBar buttons
        view.parametersWidget.addWidget(view.settingsParametersView)
        view.parametersWidget.addWidget(view.dataParametersView)
        view.parametersWidget.addWidget(view.detectorParametersView)
        view.parametersWidget.addWidget(view.scanParametersView)
        view.parametersWidget.addWidget(view.probeParametersView)
        view.parametersWidget.addWidget(view.objectParametersView)
        view.parametersWidget.addWidget(view.reconstructorParametersView)
        view.parametersWidget.addWidget(view.workflowParametersView)
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
        view.contentsWidget.addWidget(view.monitorObjectView)
        view.contentsWidget.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
        view.splitter.addWidget(view.contentsWidget)

        view.setCentralWidget(view.splitter)

        # TODO make visible when complete
        view.workflowAction.setVisible(isDeveloperModeEnabled)

        desktopSize = QApplication.desktop().availableGeometry().size()
        view.resize(desktopSize.width() * 2 // 3, desktopSize.height() * 2 // 3)
        view.statusBar().showMessage('Ready')  # TODO make better use of the statusBar

        return view
