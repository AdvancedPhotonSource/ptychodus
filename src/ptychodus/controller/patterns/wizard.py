from __future__ import annotations
from pathlib import Path
from typing import Final
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
    QWizardPage,
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
    CheckBoxParameterViewController,
    CheckableGroupBoxParameterViewController,
    ParameterViewController,
    PathParameterViewController,
    SpinBoxParameterViewController,
)

logger = logging.getLogger(__name__)

__all__ = [
    'OpenDatasetWizardController',
]


class OpenDatasetWizardFilesViewController(Observer):
    def __init__(self, api: PatternsAPI, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__()
        self._api = api
        self._page = OpenDatasetWizardFilesPage()
        self._file_dialog_factory = file_dialog_factory

        self._fileSystemModel = QFileSystemModel()
        self._fileSystemModel.setFilter(QDir.Filter.AllEntries | QDir.Filter.AllDirs)
        self._fileSystemModel.setNameFilterDisables(False)

        self._fileSystemProxyModel = QSortFilterProxyModel()
        self._fileSystemProxyModel.setSourceModel(self._fileSystemModel)

        self._page.directoryComboBox.addItem(str(file_dialog_factory.getOpenWorkingDirectory()))
        self._page.directoryComboBox.addItem(str(Path.home()))
        self._page.directoryComboBox.setEditable(True)
        self._page.directoryComboBox.textActivated.connect(self._handleDirectoryComboBoxActivated)

        self._page.fileSystemTableView.setModel(self._fileSystemProxyModel)
        self._page.fileSystemTableView.setSortingEnabled(True)
        self._page.fileSystemTableView.sortByColumn(0, Qt.SortOrder.AscendingOrder)
        self._page.fileSystemTableView.verticalHeader().hide()
        self._page.fileSystemTableView.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._page.fileSystemTableView.doubleClicked.connect(
            self._handleFileSystemTableDoubleClicked
        )
        self._page.fileSystemTableView.selectionModel().currentChanged.connect(
            self._checkIfComplete
        )

        for fileFilter in api.getOpenFileFilterList():
            self._page.fileTypeComboBox.addItem(fileFilter)

        self._page.fileTypeComboBox.textActivated.connect(self._setNameFiltersInFileSystemModel)

        self._setRootPath(file_dialog_factory.getOpenWorkingDirectory())
        self._sync_model_to_view()
        api.addObserver(self)

    def _setRootPath(self, rootPath: Path) -> None:
        index = self._fileSystemModel.setRootPath(str(rootPath))
        proxyIndex = self._fileSystemProxyModel.mapFromSource(index)
        self._page.fileSystemTableView.setRootIndex(proxyIndex)
        self._page.directoryComboBox.setCurrentText(str(rootPath))
        self._file_dialog_factory.setOpenWorkingDirectory(rootPath)

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
        self._file_dialog_factory.setOpenWorkingDirectory(filePath.parent)

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

    def _sync_model_to_view(self) -> None:
        self._page.fileTypeComboBox.setCurrentText(self._api.getOpenFileFilter())

    def update(self, observable: Observable) -> None:
        if observable is self._api:
            self._sync_model_to_view()

    def getWidget(self) -> QWizardPage:
        return self._page


class OpenDatasetWizardMetadataViewController(Observer):
    def __init__(self, presenter: MetadataPresenter) -> None:
        super().__init__()
        self._presenter = presenter
        self._page = OpenDatasetWizardMetadataPage()

        presenter.addObserver(self)
        self._sync_model_to_view()
        self._page._setComplete(True)

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

    def _sync_model_to_view(self) -> None:
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

    def getWidget(self) -> QWizardPage:
        return self._page

    def update(self, observable: Observable) -> None:
        if observable is self._presenter:
            self._sync_model_to_view()


class PatternLoadViewController(ParameterViewController):
    def __init__(self, settings: PatternSettings) -> None:
        super().__init__()
        self._view_controller = SpinBoxParameterViewController(
            settings.numberOfDataThreads,
        )
        self._widget = QGroupBox('Load')

        layout = QFormLayout()
        layout.addRow('Number of Data Threads:', self._view_controller.getWidget())
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class PatternMemoryMapViewController(CheckableGroupBoxParameterViewController):
    def __init__(self, settings: PatternSettings, file_dialog_factory: FileDialogFactory) -> None:
        super().__init__(settings.memmapEnabled, 'Memory Map Diffraction Data')
        self._view_controller = PathParameterViewController.createDirectoryChooser(
            settings.scratchDirectory, file_dialog_factory
        )

        layout = QFormLayout()
        layout.addRow('Scratch Directory:', self._view_controller.getWidget())
        self.getWidget().setLayout(layout)


class PatternCropViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.cropEnabled, 'Crop')
        self._settings = settings
        self._sizer = sizer

        self._center_x_spin_box = QSpinBox()
        self._center_y_spin_box = QSpinBox()
        self._width_spin_box = QSpinBox()
        self._height_spin_box = QSpinBox()
        self._flip_x_check_box = QCheckBox('Flip X')
        self._flip_y_check_box = QCheckBox('Flip Y')

        layout = QGridLayout()
        layout.addWidget(QLabel('Center:'), 0, 0)
        layout.addWidget(self._center_x_spin_box, 0, 1)
        layout.addWidget(self._center_y_spin_box, 0, 2)
        layout.addWidget(QLabel('Extent:'), 1, 0)
        layout.addWidget(self._width_spin_box, 1, 1)
        layout.addWidget(self._height_spin_box, 1, 2)
        layout.addWidget(QLabel('Axes:'), 2, 0)
        layout.addWidget(self._flip_x_check_box, 2, 1, Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self._flip_y_check_box, 2, 2, Qt.AlignmentFlag.AlignHCenter)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.getWidget().setLayout(layout)

        self._sync_model_to_view()

        self._center_x_spin_box.valueChanged.connect(settings.cropCenterXInPixels.setValue)
        self._center_y_spin_box.valueChanged.connect(settings.cropCenterYInPixels.setValue)
        self._width_spin_box.valueChanged.connect(settings.cropWidthInPixels.setValue)
        self._height_spin_box.valueChanged.connect(settings.cropHeightInPixels.setValue)
        self._flip_x_check_box.toggled.connect(settings.flipXEnabled.setValue)
        self._flip_y_check_box.toggled.connect(settings.flipYEnabled.setValue)

        sizer.addObserver(self)

    def _sync_model_to_view(self) -> None:
        center_x = self._sizer.axis_x.get_crop_center()
        center_y = self._sizer.axis_y.get_crop_center()
        width = self._sizer.axis_x.get_crop_size()
        height = self._sizer.axis_y.get_crop_size()

        center_x_limits = self._sizer.axis_x.get_crop_center_limits()
        center_y_limits = self._sizer.axis_y.get_crop_center_limits()
        width_limits = self._sizer.axis_x.get_crop_size_limits()
        height_limits = self._sizer.axis_y.get_crop_size_limits()

        self._center_x_spin_box.blockSignals(True)
        self._center_x_spin_box.setRange(center_x_limits.lower, center_x_limits.upper)
        self._center_x_spin_box.setValue(center_x)
        self._center_x_spin_box.blockSignals(False)

        self._center_y_spin_box.blockSignals(True)
        self._center_y_spin_box.setRange(center_y_limits.lower, center_y_limits.upper)
        self._center_y_spin_box.setValue(center_y)
        self._center_y_spin_box.blockSignals(False)

        self._width_spin_box.blockSignals(True)
        self._width_spin_box.setRange(width_limits.lower, width_limits.upper)
        self._width_spin_box.setValue(width)
        self._width_spin_box.blockSignals(False)

        self._height_spin_box.blockSignals(True)
        self._height_spin_box.setRange(height_limits.lower, height_limits.upper)
        self._height_spin_box.setValue(height)
        self._height_spin_box.blockSignals(False)

        self._flip_x_check_box.setChecked(self._settings.flipXEnabled.getValue())
        self._flip_y_check_box.setChecked(self._settings.flipYEnabled.getValue())

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super().update(observable)


class PatternBinningViewController(CheckableGroupBoxParameterViewController):
    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.binningEnabled, 'Bin Pixels')
        self._settings = settings
        self._sizer = sizer

        self._bin_size_x_spin_box = QSpinBox()
        self._bin_size_y_spin_box = QSpinBox()

        layout = QGridLayout()
        layout.addWidget(QLabel('Bin Size:'), 0, 0)
        layout.addWidget(self._bin_size_x_spin_box, 0, 1)
        layout.addWidget(self._bin_size_y_spin_box, 0, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.getWidget().setLayout(layout)

        self._sync_model_to_view()

        self._bin_size_x_spin_box.valueChanged.connect(settings.binSizeX.setValue)
        self._bin_size_y_spin_box.valueChanged.connect(settings.binSizeY.setValue)

        sizer.addObserver(self)

    def _sync_model_to_view(self) -> None:
        bin_size_x = self._sizer.axis_x.get_bin_size()
        bin_size_y = self._sizer.axis_y.get_bin_size()

        bin_size_x_limits = self._sizer.axis_x.get_bin_size_limits()
        bin_size_y_limits = self._sizer.axis_y.get_bin_size_limits()

        self._bin_size_x_spin_box.blockSignals(True)
        self._bin_size_x_spin_box.setRange(bin_size_x_limits.lower, bin_size_x_limits.upper)
        self._bin_size_x_spin_box.setValue(bin_size_x)
        self._bin_size_x_spin_box.blockSignals(False)

        self._bin_size_y_spin_box.blockSignals(True)
        self._bin_size_y_spin_box.setRange(bin_size_y_limits.lower, bin_size_y_limits.upper)
        self._bin_size_y_spin_box.setValue(bin_size_y)
        self._bin_size_y_spin_box.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super().update(observable)


class PatternPaddingViewController(CheckableGroupBoxParameterViewController):
    MAX_INT: Final[int] = 0x7FFFFFFF

    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
    ) -> None:
        super().__init__(settings.paddingEnabled, 'Pad')
        self._settings = settings
        self._sizer = sizer

        self._pad_x_spin_box = QSpinBox()
        self._pad_y_spin_box = QSpinBox()

        layout = QGridLayout()
        layout.addWidget(QLabel('Padding:'), 0, 0)
        layout.addWidget(self._pad_x_spin_box, 0, 1)
        layout.addWidget(self._pad_y_spin_box, 0, 2)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self.getWidget().setLayout(layout)

        self._sync_model_to_view()

        self._pad_x_spin_box.valueChanged.connect(settings.padX.setValue)
        self._pad_y_spin_box.valueChanged.connect(settings.padY.setValue)

        sizer.addObserver(self)

    def _sync_model_to_view(self) -> None:
        pad_x = self._sizer.axis_x.get_pad_size()
        pad_y = self._sizer.axis_y.get_pad_size()

        self._pad_x_spin_box.blockSignals(True)
        self._pad_x_spin_box.setRange(0, self.MAX_INT)
        self._pad_x_spin_box.setValue(pad_x)
        self._pad_x_spin_box.blockSignals(False)

        self._pad_y_spin_box.blockSignals(True)
        self._pad_y_spin_box.setRange(0, self.MAX_INT)
        self._pad_y_spin_box.setValue(pad_y)
        self._pad_y_spin_box.blockSignals(False)

    def update(self, observable: Observable) -> None:
        if observable is self._sizer:
            self._sync_model_to_view()
        else:
            super().update(observable)


class PatternTransformViewController:
    def __init__(self, settings: PatternSettings) -> None:
        self._lower_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.valueLowerBoundEnabled, 'Value Lower Bound:'
        )
        self._lower_bound_view_controller = SpinBoxParameterViewController(settings.valueLowerBound)
        self._upper_bound_enabled_view_controller = CheckBoxParameterViewController(
            settings.valueUpperBoundEnabled, 'Value upper Bound:'
        )
        self._upper_bound_view_controller = SpinBoxParameterViewController(settings.valueUpperBound)

        layout = QGridLayout()
        layout.addWidget(self._lower_bound_enabled_view_controller.getWidget(), 0, 0)
        layout.addWidget(self._lower_bound_view_controller.getWidget(), 0, 1, 1, 2)
        layout.addWidget(self._upper_bound_view_controller.getWidget(), 1, 0)
        layout.addWidget(self._upper_bound_view_controller.getWidget(), 1, 1, 1, 2)
        layout.setColumnStretch(2, 1)
        layout.setColumnStretch(3, 1)

        self._widget = QGroupBox('Transform')
        self._widget.setLayout(layout)

    def getWidget(self) -> QWidget:
        return self._widget


class OpenDatasetWizardPatternsViewController(ParameterViewController):
    def __init__(
        self, settings: PatternSettings, sizer: PatternSizer, file_dialog_factory: FileDialogFactory
    ) -> None:
        self._loadViewController = PatternLoadViewController(settings)
        self._memoryMapViewController = PatternMemoryMapViewController(
            settings, file_dialog_factory
        )
        self._cropViewController = PatternCropViewController(settings, sizer)
        self._binningViewController = PatternBinningViewController(settings, sizer)
        self._paddingViewController = PatternPaddingViewController(settings, sizer)
        self._transformViewController = PatternTransformViewController(settings)

        layout = QVBoxLayout()
        layout.addWidget(self._loadViewController.getWidget())
        layout.addWidget(self._memoryMapViewController.getWidget())
        layout.addWidget(self._cropViewController.getWidget())
        layout.addWidget(self._binningViewController.getWidget())
        layout.addWidget(self._paddingViewController.getWidget())
        layout.addWidget(self._transformViewController.getWidget())
        layout.addStretch()

        self._page = OpenDatasetWizardPage()
        self._page.setTitle('Pattern Processing')
        self._page._setComplete(True)  # FIXME why???
        self._page.setLayout(layout)

    def getWidget(self) -> QWizardPage:
        return self._page


class OpenDatasetWizardController:
    def __init__(
        self,
        settings: PatternSettings,
        sizer: PatternSizer,
        api: PatternsAPI,
        metadata_presenter: MetadataPresenter,
        file_dialog_factory: FileDialogFactory,
    ) -> None:
        self._api = api
        self._file_view_controller = OpenDatasetWizardFilesViewController(
            self._api, file_dialog_factory
        )
        self._metadata_view_controller = OpenDatasetWizardMetadataViewController(metadata_presenter)
        self._patterns_view_controller = OpenDatasetWizardPatternsViewController(
            settings, sizer, file_dialog_factory
        )

        self._wizard = QWizard()
        self._wizard.setWindowTitle('Open Dataset')
        self._wizard.addPage(self._file_view_controller.getWidget())
        self._wizard.addPage(self._metadata_view_controller.getWidget())
        self._wizard.addPage(self._patterns_view_controller.getWidget())

        self._wizard.button(QWizard.WizardButton.NextButton).clicked.connect(
            self._executeNextButtonAction
        )
        self._wizard.button(QWizard.WizardButton.FinishButton).clicked.connect(
            self._executeFinishButtonAction
        )

    def _executeNextButtonAction(self) -> None:
        page = self._wizard.currentPage()

        if page is self._metadata_view_controller.getWidget():
            self._file_view_controller.openDataset()
        elif page is self._patterns_view_controller.getWidget():
            self._metadata_view_controller.importMetadata()

    def _executeFinishButtonAction(self) -> None:
        self._api.startAssemblingDiffractionPatterns()  # FIXME

    def openDataset(self) -> None:
        self._api.stopAssemblingDiffractionPatterns(finishAssembling=False)  # FIXME
        self._wizard.restart()
        self._wizard.show()
