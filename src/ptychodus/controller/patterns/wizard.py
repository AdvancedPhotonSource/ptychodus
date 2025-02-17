from __future__ import annotations
from pathlib import Path
import logging
import re

from PyQt5.QtCore import Qt, QDir, QFileInfo, QModelIndex, QSortFilterProxyModel
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileSystemModel,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QWizard,
)

from ptychodus.api.observer import Observable, Observer

from ...model.metadata import MetadataPresenter
from ...model.patterns import PatternSettings, PatternSizer, PatternsAPI
from ...view.patterns import (
    OpenDatasetWizardFilesPage,
    OpenDatasetWizardMetadataPage,
    OpenDatasetWizardPage,
)

from ..data import FileDialogFactory
from ..parametric import (
    CheckableGroupBoxParameterViewController,
    ParameterViewController,
    PathParameterViewController,
    SpinBoxParameterViewController,
)

logger = logging.getLogger(__name__)

__all__ = [
    'OpenDatasetWizardController',
]


class OpenDatasetWizardFilesController(Observer):
    def __init__(
        self,
        api: PatternsAPI,
        page: OpenDatasetWizardFilesPage,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        super().__init__()
        self._api = api
        self._page = page
        self._fileDialogFactory = fileDialogFactory

        self._fileSystemModel = QFileSystemModel()
        self._fileSystemModel.setFilter(QDir.Filter.AllEntries | QDir.Filter.AllDirs)
        self._fileSystemModel.setNameFilterDisables(False)

        self._fileSystemProxyModel = QSortFilterProxyModel()
        self._fileSystemProxyModel.setSourceModel(self._fileSystemModel)

        page.directoryComboBox.addItem(str(fileDialogFactory.getOpenWorkingDirectory()))
        page.directoryComboBox.addItem(str(Path.home()))
        page.directoryComboBox.setEditable(True)
        page.directoryComboBox.textActivated.connect(self._handleDirectoryComboBoxActivated)

        page.fileSystemTableView.setModel(self._fileSystemProxyModel)
        page.fileSystemTableView.setSortingEnabled(True)
        page.fileSystemTableView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        page.fileSystemTableView.verticalHeader().hide()
        page.fileSystemTableView.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        page.fileSystemTableView.doubleClicked.connect(self._handleFileSystemTableDoubleClicked)
        page.fileSystemTableView.selectionModel().currentChanged.connect(self._checkIfComplete)

        for fileFilter in api.getOpenFileFilterList():
            page.fileTypeComboBox.addItem(fileFilter)

        page.fileTypeComboBox.textActivated.connect(self._setNameFiltersInFileSystemModel)

        self._setRootPath(fileDialogFactory.getOpenWorkingDirectory())
        self._syncModelToView()
        api.addObserver(self)

    def _setRootPath(self, rootPath: Path) -> None:
        index = self._fileSystemModel.setRootPath(str(rootPath))
        proxyIndex = self._fileSystemProxyModel.mapFromSource(index)
        self._page.fileSystemTableView.setRootIndex(proxyIndex)
        self._page.directoryComboBox.setCurrentText(str(rootPath))
        self._fileDialogFactory.setOpenWorkingDirectory(rootPath)

    def _handleDirectoryComboBoxActivated(self, text: str) -> None:
        fileInfo = QFileInfo(text)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def _handleFileSystemTableDoubleClicked(self, proxyIndex: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        fileInfo = self._fileSystemModel.fileInfo(index)

        if fileInfo.isDir():
            self._setRootPath(Path(fileInfo.canonicalFilePath()))

    def openDataset(self) -> None:
        proxyIndex = self._page.fileSystemTableView.currentIndex()
        index = self._fileSystemProxyModel.mapToSource(proxyIndex)
        filePath = Path(self._fileSystemModel.filePath(index))
        self._fileDialogFactory.setOpenWorkingDirectory(filePath.parent)

        fileFilter = self._page.fileTypeComboBox.currentText()
        self._api.openPatterns(filePath, fileType=fileFilter)

    def _checkIfComplete(self, current: QModelIndex, previous: QModelIndex) -> None:
        index = self._fileSystemProxyModel.mapToSource(current)
        fileInfo = self._fileSystemModel.fileInfo(index)
        self._page._setComplete(fileInfo.isFile())

    def _setNameFiltersInFileSystemModel(self, currentText: str) -> None:
        z = re.search(r'\((.+)\)', currentText)

        if z:
            nameFilters = z.group(1).split()
            logger.debug(f'Dataset File Name Filters: {nameFilters}')
            self._fileSystemModel.setNameFilters(nameFilters)

    def _syncModelToView(self) -> None:
        self._page.fileTypeComboBox.setCurrentText(self._api.getOpenFileFilter())

    def update(self, observable: Observable) -> None:
        if observable is self._api:
            self._syncModelToView()


class OpenDatasetWizardMetadataViewController(Observer):
    def __init__(
        self,
        presenter: MetadataPresenter,
        page: OpenDatasetWizardMetadataPage,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._page = page

        presenter.addObserver(self)
        self._syncModelToView()
        page._setComplete(True)

    def importMetadata(self) -> None:
        if self._page.detectorPixelCountCheckBox.isChecked():
            self._presenter.syncDetectorPixelCount()

        if self._page.detectorPixelSizeCheckBox.isChecked():
            self._presenter.syncDetectorPixelSize()

        if self._page.detectorBitDepthCheckBox.isChecked():
            self._presenter.syncDetectorBitDepth()

        if self._page.detectorDistanceCheckBox.isChecked():
            self._presenter.syncDetectorDistance()

        self._presenter.syncPatternCrop(
            syncCenter=self._page.patternCropCenterCheckBox.isChecked(),
            syncExtent=self._page.patternCropExtentCheckBox.isChecked(),
        )

        if self._page.probeEnergyCheckBox.isChecked():
            self._presenter.syncProbeEnergy()

    def _syncModelToView(self) -> None:
        canSyncDetectorPixelCount = self._presenter.canSyncDetectorPixelCount()
        self._page.detectorPixelCountCheckBox.setVisible(canSyncDetectorPixelCount)
        self._page.detectorPixelCountCheckBox.setChecked(canSyncDetectorPixelCount)

        canSyncDetectorPixelSize = self._presenter.canSyncDetectorPixelSize()
        self._page.detectorPixelSizeCheckBox.setVisible(canSyncDetectorPixelSize)
        self._page.detectorPixelSizeCheckBox.setChecked(canSyncDetectorPixelSize)

        canSyncDetectorBitDepth = self._presenter.canSyncDetectorBitDepth()
        self._page.detectorBitDepthCheckBox.setVisible(canSyncDetectorBitDepth)
        self._page.detectorBitDepthCheckBox.setChecked(canSyncDetectorBitDepth)

        canSyncDetectorDistance = self._presenter.canSyncDetectorDistance()
        self._page.detectorDistanceCheckBox.setVisible(canSyncDetectorDistance)
        self._page.detectorDistanceCheckBox.setChecked(canSyncDetectorDistance)

        canSyncPatternCropCenter = self._presenter.canSyncPatternCropCenter()
        self._page.patternCropCenterCheckBox.setVisible(canSyncPatternCropCenter)
        self._page.patternCropCenterCheckBox.setChecked(canSyncPatternCropCenter)

        canSyncPatternCropExtent = self._presenter.canSyncPatternCropExtent()
        self._page.patternCropExtentCheckBox.setVisible(canSyncPatternCropExtent)
        self._page.patternCropExtentCheckBox.setChecked(canSyncPatternCropExtent)

        canSyncProbeEnergy = self._presenter.canSyncProbeEnergy()
        self._page.probeEnergyCheckBox.setVisible(canSyncProbeEnergy)
        self._page.probeEnergyCheckBox.setChecked(canSyncProbeEnergy)

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class PatternLoadViewController(ParameterViewController):
    def __init__(self, settings: PatternSettings) -> None:
        super().__init__()
        self._viewController = SpinBoxParameterViewController(
            settings.numberOfDataThreads,
        )
        self._widget = QGroupBox('Load')

        layout = QFormLayout()
        layout.addRow('Number of Data Threads:', self._viewController.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PatternMemoryMapViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: PatternSettings, fileDialogFactory: FileDialogFactory) -> None:
        super().__init__(settings.memmapEnabled, 'Memory Map Diffraction Data')
        self._viewController = PathParameterViewController.createDirectoryChooser(
            settings.scratchDirectory, fileDialogFactory
        )

        layout = QFormLayout()
        layout.addRow('Scratch Directory:', self._viewController.getWidget())
        self.getWidget().setLayout(layout)


class PatternCropViewController(CheckableGroupBoxParameterViewController, Observer):
    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.cropEnabled, 'Crop')
        self._settings = settings
        self._sizer = sizer

        self._centerXSpinBox = QSpinBox()
        self._centerYSpinBox = QSpinBox()
        self._widthSpinBox = QSpinBox()
        self._heightSpinBox = QSpinBox()

        layout = QGridLayout()
        layout.addWidget(QLabel('Center [px]:'), 0, 0)
        layout.addWidget(self._centerXSpinBox, 0, 1)
        layout.addWidget(self._centerYSpinBox, 0, 2)
        layout.addWidget(QLabel('Extent [px]:'), 1, 0)
        layout.addWidget(self._widthSpinBox, 1, 1)
        layout.addWidget(self._heightSpinBox, 1, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.getWidget().setLayout(layout)

        self._syncModelToView()

        self._centerXSpinBox.valueChanged.connect(settings.cropCenterXInPixels.setValue)
        self._centerYSpinBox.valueChanged.connect(settings.cropCenterYInPixels.setValue)
        self._widthSpinBox.valueChanged.connect(settings.cropWidthInPixels.setValue)
        self._heightSpinBox.valueChanged.connect(settings.cropHeightInPixels.setValue)

        sizer.addObserver(self)

    def _syncModelToView(self) -> None:
        center_x = self._sizer.axis_x.get_crop_center()
        center_y = self._sizer.axis_y.get_crop_center()
        width = self._sizer.axis_x.get_crop_size()
        height = self._sizer.axis_y.get_crop_size()

        center_x_limits = self._sizer.axis_x.get_crop_center_limits()
        center_y_limits = self._sizer.axis_y.get_crop_center_limits()
        width_limits = self._sizer.axis_x.get_crop_size_limits()
        height_limits = self._sizer.axis_y.get_crop_size_limits()

        self._centerXSpinBox.blockSignals(True)
        self._centerXSpinBox.setRange(center_x_limits.lower, center_x_limits.upper)
        self._centerXSpinBox.setValue(center_x)
        self._centerXSpinBox.blockSignals(False)

        self._centerYSpinBox.blockSignals(True)
        self._centerYSpinBox.setRange(center_y_limits.lower, center_y_limits.upper)
        self._centerYSpinBox.setValue(center_y)
        self._centerYSpinBox.blockSignals(False)

        self._widthSpinBox.blockSignals(True)
        self._widthSpinBox.setRange(width_limits.lower, width_limits.upper)
        self._widthSpinBox.setValue(width)
        self._widthSpinBox.blockSignals(False)

        self._heightSpinBox.blockSignals(True)
        self._heightSpinBox.setRange(height_limits.lower, height_limits.upper)
        self._heightSpinBox.setValue(height)
        self._heightSpinBox.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._syncModelToView()


class OpenDatasetWizardPatternTransformView(QGroupBox):  # FIXME
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__('Transform', parent)
        self.valueLowerBoundCheckBox = QCheckBox('Value Lower Bound:')
        self.valueLowerBoundSpinBox = QSpinBox()
        self.valueUpperBoundCheckBox = QCheckBox('Value Upper Bound:')
        self.valueUpperBoundSpinBox = QSpinBox()
        self.axesLabel = QLabel('Axes:')
        self.flipXCheckBox = QCheckBox('Flip X')
        self.flipYCheckBox = QCheckBox('Flip Y')

    @classmethod
    def createInstance(cls, parent: QWidget | None = None) -> OpenDatasetWizardPatternTransformView:
        view = cls(parent)

        layout = QGridLayout()
        layout.addWidget(view.valueLowerBoundCheckBox, 0, 0)
        layout.addWidget(view.valueLowerBoundSpinBox, 0, 1, 1, 2)
        layout.addWidget(view.valueUpperBoundCheckBox, 1, 0)
        layout.addWidget(view.valueUpperBoundSpinBox, 1, 1, 1, 2)
        layout.addWidget(view.axesLabel, 2, 0)
        layout.addWidget(view.flipXCheckBox, 2, 1, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(view.flipYCheckBox, 2, 2, Qt.AlignmentFlag.AlignHCenter)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)
        view.setLayout(layout)

        return view


class PatternTransformViewController(Observer):
    def __init__(
        self,
        presenter: DiffractionPatternPresenter,
        view: OpenDatasetWizardPatternTransformView,
    ) -> None:
        super().__init__()
        self._presenter = presenter
        self._view = view

    @classmethod
    def createInstance(
        cls,
        presenter: DiffractionPatternPresenter,
        view: OpenDatasetWizardPatternTransformView,
    ) -> PatternTransformController:
        controller = cls(presenter, view)
        presenter.addObserver(controller)

        view.valueLowerBoundCheckBox.toggled.connect(presenter.setValueLowerBoundEnabled)
        view.valueLowerBoundSpinBox.valueChanged.connect(presenter.setValueLowerBound)
        view.valueUpperBoundCheckBox.toggled.connect(presenter.setValueUpperBoundEnabled)
        view.valueUpperBoundSpinBox.valueChanged.connect(presenter.setValueUpperBound)
        view.flipXCheckBox.toggled.connect(presenter.setFlipXEnabled)
        view.flipYCheckBox.toggled.connect(presenter.setFlipYEnabled)

        controller._syncModelToView()
        return controller

    def _syncModelToView(self) -> None:
        self._view.valueLowerBoundCheckBox.setChecked(self._presenter.isValueLowerBoundEnabled())

        self._view.valueLowerBoundSpinBox.blockSignals(True)
        self._view.valueLowerBoundSpinBox.setRange(
            self._presenter.getValueLowerBoundLimits().lower,
            self._presenter.getValueLowerBoundLimits().upper,
        )
        self._view.valueLowerBoundSpinBox.setValue(self._presenter.getValueLowerBound())
        self._view.valueLowerBoundSpinBox.blockSignals(False)

        self._view.valueUpperBoundCheckBox.setChecked(self._presenter.isValueUpperBoundEnabled())

        self._view.valueUpperBoundSpinBox.blockSignals(True)
        self._view.valueUpperBoundSpinBox.setRange(
            self._presenter.getValueUpperBoundLimits().lower,
            self._presenter.getValueUpperBoundLimits().upper,
        )
        self._view.valueUpperBoundSpinBox.setValue(self._presenter.getValueUpperBound())
        self._view.valueUpperBoundSpinBox.blockSignals(False)

        self._view.flipXCheckBox.setChecked(self._presenter.isFlipXEnabled())
        self._view.flipYCheckBox.setChecked(self._presenter.isFlipYEnabled())

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._syncModelToView()


class OpenDatasetWizardPatternsViewController(ParameterViewController):
    def __init__(self, settings: PatternSettings, sizer: PatternSizer) -> None:
        self._loadViewController = PatternLoadViewController(settings)
        self._memoryMapViewController = PatternMemoryMapViewController(settings)
        self._cropViewController = PatternCropViewController(settings, sizer)
        self._paddingViewController = PatternPaddingViewController(settings)
        self._binningViewController = PatternBinningViewController(settings)
        self._transformViewController = PatternTransformViewController(settings)

        layout = QVBoxLayout()
        layout.addWidget(self._loadViewController.getWidget())
        layout.addWidget(self._memoryMapViewController.getWidget())
        layout.addWidget(self._cropViewController.getWidget())
        layout.addWidget(self._paddingViewController.getWidget())
        layout.addWidget(self._binningViewController.getWidget())
        layout.addStretch()

        self._widget = OpenDatasetWizardPage()
        self._widget.setTitle('Pattern Processing')
        self._widget._setComplete(True)  # FIXME ???
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class OpenDatasetWizard(QWizard):
    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.filesPage = OpenDatasetWizardFilesPage.createInstance()
        self.metadataPage = OpenDatasetWizardMetadataPage.createInstance()
        self.patternsPage = OpenDatasetWizardPatternsPage.createInstance()

        self.setWindowTitle('Open Dataset')
        self.addPage(self.filesPage)
        self.addPage(self.metadataPage)
        self.addPage(self.patternsPage)


class OpenDatasetWizardController:
    def __init__(
        self,
        api: PatternsAPI,
        metadataPresenter: DiffractionMetadataPresenter,
        datasetPresenter: DiffractionDatasetPresenter,
        patternPresenter: DiffractionPatternPresenter,
        wizard: OpenDatasetWizard,
        fileDialogFactory: FileDialogFactory,
    ) -> None:
        self._api = api
        self._wizard = wizard
        self._filesController = OpenDatasetWizardFilesController.createInstance(
            api, wizard.filesPage, fileDialogFactory
        )
        self._metadataController = OpenDatasetWizardMetadataController.createInstance(
            metadataPresenter, wizard.metadataPage
        )
        self._patternsController = OpenDatasetWizardPatternsController.createInstance(
            api,
            datasetPresenter,
            patternPresenter,
            wizard.patternsPage,
            fileDialogFactory,
        )

    @classmethod
    def createInstance(
        cls,
        api: PatternsAPI,
        metadataPresenter: DiffractionMetadataPresenter,
        datasetPresenter: DiffractionDatasetPresenter,
        patternPresenter: DiffractionPatternPresenter,
        wizard: OpenDatasetWizard,
        fileDialogFactory: FileDialogFactory,
    ) -> OpenDatasetWizardController:
        controller = cls(
            api,
            metadataPresenter,
            datasetPresenter,
            patternPresenter,
            wizard,
            fileDialogFactory,
        )
        wizard.button(QWizard.WizardButton.NextButton).clicked.connect(
            controller._executeNextButtonAction
        )
        wizard.button(QWizard.WizardButton.FinishButton).clicked.connect(
            controller._executeFinishButtonAction
        )
        return controller

    def _executeNextButtonAction(self) -> None:
        page = self._wizard.currentPage()

        if page is self._wizard.metadataPage:
            self._filesController.openDataset()
        elif page is self._wizard.patternsPage:
            self._metadataController.importMetadata()

    def _executeFinishButtonAction(self) -> None:
        self._api.startAssemblingDiffractionPatterns()

    def openDataset(self) -> None:
        self._api.stopAssemblingDiffractionPatterns(finishAssembling=False)
        self._wizard.restart()
        self._wizard.show()
